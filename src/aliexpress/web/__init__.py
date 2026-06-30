"""Flask application layer: blueprints, models, extensions, CLI, and request infrastructure.

``routes/`` contains the blueprints; ``extensions`` wires up Flask-SQLAlchemy, Flask-Login,
and Flask-Limiter; ``models`` defines the database schema; ``storage`` manages the
per-process file layout; ``cli`` adds the ``schools`` and ``admins`` management commands.
"""
