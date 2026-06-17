"""Flask CLI commands for school and admin management.

Register via ``app.cli.add_command(schools)`` and ``app.cli.add_command(admins)``
in ``app.py``. Commands run inside an application context automatically, so
``db.session`` is available without an explicit ``with app.app_context():``.
"""

import os
import secrets
import shutil
import string

import click
from flask import current_app
from werkzeug.security import generate_password_hash

from .extensions import db
from .models import Admin, School

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
def delete_school(schoolcode):
    """Verwijder een school en al haar gegevens (inclusief opgeslagen bestanden)."""
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
    storage_dir = os.path.join(current_app.instance_path, "storage", schoolcode)
    if os.path.isdir(storage_dir):
        shutil.rmtree(storage_dir)
        click.echo(f"Opgeslagen bestanden verwijderd: {storage_dir}")
    click.echo(f"School '{schoolcode}' verwijderd.")


@click.group()
def admins():
    """Admin account management commands."""


@admins.command("add")
@click.argument("username")
def add_admin(username):
    """Maak een admin-account aan."""
    if Admin.query.filter_by(username=username).first():
        raise click.ClickException(f"Admin '{username}' bestaat al.")
    password = click.prompt("Wachtwoord", hide_input=True, confirmation_prompt=True)
    admin = Admin(
        username=username,
        password_hash=generate_password_hash(password),
    )
    db.session.add(admin)
    db.session.commit()
    click.echo(f"Admin '{username}' aangemaakt.")
