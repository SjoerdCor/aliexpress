"""Configuration settings for different application environments."""

import os


# pylint: disable=too-few-public-methods  # Flask config classes expose settings via class attributes, not methods
class Config:
    """Base configuration with default settings."""

    SECRET_KEY = os.getenv("SECRET_KEY")
    DEBUG = False
    TESTING = False

    # Small, structured state (job status, log lines; later schools/login) lives in a
    # SQLite database. A relative SQLite path is resolved against Flask's instance folder,
    # so the default becomes ``instance/app.db``. The URI is overridable via the
    # ``DATABASE_URL`` environment variable, which is how the EU server later points at
    # PostgreSQL and how the tests point at a throwaway database.
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", "sqlite:///app.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Session security. SECURE is off in the base config (no HTTPS locally/in tests);
    # ProductionConfig flips it on. HTTPONLY and SAMESITE are always on. The login session
    # is a non-permanent cookie (see login_user(..., remember=False)): it lasts until the
    # browser closes, which is the right default for a shared school computer.
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = False


class DevelopmentConfig(Config):
    """Development configuration with debug mode enabled."""

    SECRET_KEY = os.getenv("SECRET_KEY", "dev-fallback-secret")
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration with debug mode disabled."""

    DEBUG = False
    SESSION_COOKIE_SECURE = True
