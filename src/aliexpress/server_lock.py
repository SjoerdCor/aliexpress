"""Cross-platform exclusion between the local server and destructive maintenance."""

from contextlib import contextmanager
from pathlib import Path

import click
from filelock import FileLock, Timeout

INSTANCE_LOCK_FILENAME = ".ali-express-instance.lock"


def instance_lock_path(instance_path):
    """Return the lock-file path for an absolute application instance path."""
    path = Path(instance_path).expanduser()
    if not path.is_absolute():
        raise click.ClickException(
            "Onveilige configuratie: de instance-directory moet een absoluut pad zijn."
        )
    return path.resolve() / INSTANCE_LOCK_FILENAME


@contextmanager
def lock_instance(instance_path):
    """Exclusively hold an instance while serving or resetting it."""
    path = instance_lock_path(instance_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(path)
        lock.acquire(timeout=0)
    except Timeout as exc:
        raise click.ClickException(
            "De lokale instance is actief; stop de server of wacht op de lopende "
            "onderhoudsopdracht."
        ) from exc
    except OSError as exc:
        raise click.ClickException(
            f"Kan de lokale instance niet veilig vergrendelen via {path}."
        ) from exc
    try:
        yield
    finally:
        lock.release()
