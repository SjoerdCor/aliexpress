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
    # so this becomes ``instance/app.db``. Swapping to PostgreSQL on the EU server later is
    # only a change of this URI.
    SQLALCHEMY_DATABASE_URI = "sqlite:///app.db"
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class DevelopmentConfig(Config):
    """Development configuration with debug mode enabled."""

    SECRET_KEY = os.getenv("SECRET_KEY", "dev-fallback-secret")
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration with debug mode disabled."""

    DEBUG = False
