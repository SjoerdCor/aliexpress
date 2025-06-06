import os


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY")
    DEBUG = False
    TESTING = False


class DevelopmentConfig(Config):
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-fallback-secret")
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False
