"""Custom exception classes for handling readable application errors."""


class ReadableError(Exception):
    """Base exception with user-friendly and technical error details."""

    def __init__(self, code, context=None, technical_message=None):
        super().__init__(technical_message or code)
        self.code = code
        self.context = context or {}
        self.technical_message = technical_message


class DuplicateNameError(ReadableError):
    """Raised when duplicate names are inevitable"""


class DuplicateGroupError(ReadableError):
    """Raised when duplicate groups are detected"""


class ValidationError(ReadableError):
    """Raised when input validation fails."""


class FeasibilityError(ReadableError):
    """Raised when a feasibility check fails."""


class CouldNotReadFileError(ReadableError):
    """Generic error, raised when a file cannot be read for unknown reason."""


class SolverError(Exception):
    """Raised when the LP solver does not reach optimality."""


class StageInfeasible(Exception):
    """Raised by ``solver.strategies.solve_stage`` when a stage is proven
    infeasible — the distinct counterpart of ``SolverError`` (which covers any
    other non-optimal status). A caller that gives infeasibility a concrete
    meaning catches this; anywhere else it propagates like ``SolverError``."""
