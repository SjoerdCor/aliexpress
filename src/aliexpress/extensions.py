"""Flask extension singletons, instantiated here and initialised on the app.

Keeping the ``db`` object in its own module (rather than in ``app.py``) lets the model
modules import it without importing the application, which avoids a circular import once
``app.py`` imports those models.
"""

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
