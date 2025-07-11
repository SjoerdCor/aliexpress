"""Custom exception classes for handling readable application errors."""


class ReadableError(Exception):
    """Base exception with user-friendly and technical error details."""

    def __init__(self, code, context=None, technical_message=None):
        super().__init__(technical_message or code)
        self.code = code
        self.context = context or {}
        self.technical_message = technical_message


class ValidationError(ReadableError):
    """Raised when input validation fails."""


class FeasibilityError(ReadableError):
    """Raised when a feasibility check fails."""


class CouldNotReadFileError(ReadableError):
    """Generic error, raised when a file cannot be read for unknown reason."""
