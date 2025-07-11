"""Configuration settings for different application environments."""

import os


class Config:
    """Base configuration with default settings."""

    SECRET_KEY = os.getenv("SECRET_KEY")
    DEBUG = False
    TESTING = False


class DevelopmentConfig(Config):
    """Development configuration with debug mode enabled."""

    SECRET_KEY = os.getenv("SECRET_KEY", "dev-fallback-secret")
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration with debug mode disabled."""

    DEBUG = False
