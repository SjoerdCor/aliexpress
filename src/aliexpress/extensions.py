"""Flask extension singletons, instantiated here and initialised on the app.

Keeping these objects in their own module (rather than in ``app.py``) lets the model
modules import them without importing the application, which avoids a circular import once
``app.py`` imports those models.
"""

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
login_manager = LoginManager()
limiter = Limiter(get_remote_address, storage_uri="memory://")
