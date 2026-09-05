"""Guarded reset of local SQLite data and storage contents."""

import os
import shutil
from pathlib import Path

import click
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

from .server_lock import lock_instance


def _strict_child(value, parent, label):
    """Require a symlink-free absolute path strictly below ``parent``."""
    if not isinstance(value, (str, os.PathLike)) or not value:
        raise click.ClickException(f"Onveilige configuratie: {label} ontbreekt.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise click.ClickException(
            f"Onveilige configuratie: {label} moet een absoluut pad zijn."
        )
    path = Path(os.path.abspath(path))
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError) as exc:
        raise click.ClickException(
            f"Onveilige configuratie: {label} kan niet veilig worden opgelost."
        ) from exc
    if resolved != path:
        raise click.ClickException(
            f"Onveilige configuratie: {label} loopt via een symlink of junction: {path}."
        )
    try:
        relative = resolved.relative_to(parent)
    except ValueError as exc:
        raise click.ClickException(
            f"Onveilige configuratie: {label} {resolved} ligt niet onder {parent}."
        ) from exc
    if not relative.parts:
        raise click.ClickException(
            f"Onveilige configuratie: {label} mag niet de instance-directory zelf zijn."
        )
    return resolved


def _instance_path(application):
    """Resolve the instance directory and reject the filesystem root."""
    try:
        path = Path(application.instance_path).expanduser()
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        raise click.ClickException(
            "Onveilige configuratie: de instance-directory kan niet worden opgelost."
        ) from exc
    if not path.is_absolute():
        raise click.ClickException(
            "Onveilige configuratie: de instance-directory moet een absoluut pad zijn."
        )
    path = path.resolve()
    if path == Path(path.anchor):
        raise click.ClickException(
            "Onveilige configuratie: de instance-directory is geen veilige directory."
        )
    return path


def _database_path(config, instance):
    """Resolve the configured local SQLite database."""
    try:
        database = make_url(config.get("SQLALCHEMY_DATABASE_URI"))
    except (ArgumentError, TypeError, ValueError) as exc:
        raise click.ClickException(
            "Onveilige configuratie: DATABASE_URL is geen geldige SQLite-URL."
        ) from exc
    if database.drivername not in {"sqlite", "sqlite+pysqlite"}:
        raise click.ClickException(
            "reset-local-data verwijdert uitsluitend een lokale SQLite-bestandsdatabase; "
            "de geconfigureerde database is extern of onverwacht."
        )
    if database.query:
        raise click.ClickException(
            "Onverwachte configuratie: SQLite-URL's met queryparameters zijn niet veilig "
            "voor reset-local-data."
        )
    if not database.database or database.database in {":memory:", "file::memory:"}:
        raise click.ClickException(
            "reset-local-data vereist een lokale SQLite-bestandsdatabase, geen geheugen-"
            "of URI-database."
        )
    path = Path(database.database).expanduser()
    if not path.is_absolute():
        path = instance / path
    path = _strict_child(path, instance, "de SQLite-database")
    if path.exists() and not path.is_file():
        raise click.ClickException(
            f"Onverwachte configuratie: de SQLite-doelnaam is geen bestand: {path}."
        )
    return path


def _targets(application):
    """Resolve and validate the database and storage deletion targets."""
    config = application.config
    if config.get("ALIEXPRESS_ENV") != "local":
        raise click.ClickException(
            "reset-local-data werkt uitsluitend met ALIEXPRESS_ENV=local."
        )
    instance = _instance_path(application)
    storage = _strict_child(config.get("STORAGE_DIR"), instance, "STORAGE_DIR")
    if storage.exists() and not storage.is_dir():
        raise click.ClickException(
            f"Onverwachte configuratie: STORAGE_DIR is geen directory: {storage}."
        )
    database = _database_path(config, instance)
    if database.is_relative_to(storage) or storage.is_relative_to(database):
        raise click.ClickException(
            "Onveilige configuratie: database en storage overlappen."
        )
    return database, storage


def _database_files(database):
    """Return SQLite's database and transaction sidecars."""
    return tuple(
        Path(f"{database}{suffix}") for suffix in ("", "-wal", "-shm", "-journal")
    )


def _validated_database_files(database):
    """Return existing database files after validating every target up front."""
    files = []
    for path in _database_files(database):
        if not path.exists() and not path.is_symlink():
            continue
        if path.is_symlink() or not path.is_file():
            raise click.ClickException(
                f"Onverwacht doel bij de SQLite-database; niets verwijderd: {path}."
            )
        files.append(path)
    return tuple(files)


def _has_data(database, storage):
    """Return whether a reset target exists or contains data."""
    return any(
        path.exists() or path.is_symlink() for path in _database_files(database)
    ) or (storage.exists() and any(storage.iterdir()))


def _remove_database(files):
    """Remove database files that were validated before deletion started."""
    for path in files:
        path.unlink()


def _remove_storage_contents(storage):
    """Remove storage children while preserving the configured storage directory."""
    if not storage.exists():
        return
    for path in storage.iterdir():
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def reset_local_data(application, *, yes=False):
    """Remove local SQLite data and storage contents after safety checks."""
    database, storage = _targets(application)
    if not _has_data(database, storage):
        click.echo("De lokale omgeving is al leeg.")
        return
    with lock_instance(application.instance_path):
        # The contents may have changed while waiting to acquire the instance.
        if not _has_data(database, storage):
            click.echo("De lokale omgeving is al leeg.")
            return
        database_files = _validated_database_files(database)
        click.echo("Dit verwijdert definitief:")
        for path in database_files:
            click.echo(f"  {path}")
        click.echo(f"  de inhoud van {storage}")
        if not yes and not click.confirm("Doorgaan?", default=False):
            click.echo("Geannuleerd.")
            return
        _remove_database(database_files)
        _remove_storage_contents(storage)
        click.echo("Lokale database en storage-inhoud zijn verwijderd.")
