"""Flask CLI commands for school management.

Register via ``app.cli.add_command(schools)`` in ``app.py``. Commands run inside an
application context automatically, so ``db.session`` is available without an explicit
``with app.app_context():``.
"""

import click
from werkzeug.security import generate_password_hash

from .extensions import db
from .models import School


@click.group()
def schools():
    """School management commands."""


@schools.command("add")
@click.argument("schoolcode")
@click.option(
    "--naam", prompt="Naam van de school", help="Volledige naam van de school"
)
def add_school(schoolcode, naam):
    """Voeg een school toe met een gehashed wachtwoord."""
    password = click.prompt("Wachtwoord", hide_input=True, confirmation_prompt=True)
    if db.session.get(School, schoolcode):
        raise click.ClickException(f"School '{schoolcode}' bestaat al.")
    school = School(
        schoolcode=schoolcode,
        naam=naam,
        password_hash=generate_password_hash(password),
    )
    db.session.add(school)
    db.session.commit()
    click.echo(f"School '{schoolcode}' ({naam}) aangemaakt.")
