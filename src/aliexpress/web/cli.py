"""Flask CLI commands for school and admin management.

Register via ``app.cli.add_command(schools)`` in ``app.py``. Commands run
inside an application context automatically, so ``db.session`` is available
without an explicit ``with app.app_context():``.
"""

import os
import secrets
import shutil
import string

import click
from flask import current_app
from flask.cli import with_appcontext
from werkzeug.security import generate_password_hash

from .extensions import db
from .models import School

_PASSWORD_ALPHABET = string.ascii_letters + string.digits


def _generate_temp_password(length=16):
    return "".join(secrets.choice(_PASSWORD_ALPHABET) for _ in range(length))


@click.group()
def schools():
    """School management commands."""


@schools.command("add")
@click.argument("schoolcode")
@click.option(
    "--naam", prompt="Naam van de school", help="Volledige naam van de school"
)
@with_appcontext
def add_school(schoolcode, naam):
    """Add a school with a temporary password the school must change on first login."""
    if db.session.get(School, schoolcode):
        raise click.ClickException(f"School '{schoolcode}' bestaat al.")
    temp_password = _generate_temp_password()
    school = School(
        schoolcode=schoolcode,
        naam=naam,
        password_hash=generate_password_hash(temp_password),
        must_change_password=True,
    )
    db.session.add(school)
    db.session.commit()
    click.echo(f"School '{schoolcode}' ({naam}) aangemaakt.")
    click.echo(f"Tijdelijk wachtwoord (eenmalig zichtbaar): {temp_password}")
    click.echo("De school wordt gevraagd dit te wijzigen bij de eerste login.")


@schools.command("delete")
@click.argument("schoolcode")
@with_appcontext
def delete_school(schoolcode):
    """Delete a school and all its data (including saved files)"""
    school = db.session.get(School, schoolcode)
    if school is None:
        raise click.ClickException(f"School '{schoolcode}' bestaat niet.")
    click.confirm(
        f"Weet je zeker dat je school '{schoolcode}' ({school.naam}) wilt verwijderen? "
        "Dit verwijdert ook alle processen en resultaten.",
        abort=True,
    )
    db.session.delete(school)
    db.session.commit()
    # Resolve via STORAGE_DIR so that a custom storage location (e.g. in tests) is
    # honoured consistently with storage.py, which derives all paths from this key.
    storage_dir = os.path.join(current_app.config["STORAGE_DIR"], schoolcode)
    if os.path.isdir(storage_dir):
        shutil.rmtree(storage_dir)
        click.echo(f"Opgeslagen bestanden verwijderd: {storage_dir}")
    click.echo(f"School '{schoolcode}' verwijderd.")
