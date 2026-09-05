"""Validation and comparison rules for names that become filesystem segments.

Schoolcodes and process names are user input, database identifiers, and directory
names at the same time.  Keeping their rules here prevents POSIX from accepting a
name that later fails on Windows (or aliases to a different name on a case-insensitive
filesystem).
"""

import unicodedata

IDENTIFIER_MAX_LENGTH = 64

_RESERVED_WINDOWS_NAMES = {
    "aux",
    "clock$",
    "con",
    "nul",
    "prn",
}


class IdentifierError(ValueError):
    """Raised when a user-visible identifier is not portable."""


def normalize_identifier(value):
    """Return the canonical Unicode form used for storage and display."""
    if not isinstance(value, str):
        return value
    return unicodedata.normalize("NFC", value)


def identifier_key(value):
    """Return the cross-platform equality key for an identifier.

    NFKC handles compatibility-equivalent Unicode forms and ``casefold`` mirrors the
    case-insensitive behaviour of the filesystems in the support contract.
    """
    if not isinstance(value, str):
        raise TypeError("identifier must be a string")
    return unicodedata.normalize("NFKC", normalize_identifier(value)).casefold()


normalized_identifier_key = identifier_key


def _invalid_characters_message():
    return "Alleen letters, cijfers, spaties, - en _ toegestaan"


def _reserved_name(value):
    """Return the reserved Windows basename, or ``None``."""
    key = identifier_key(value)
    if key in _RESERVED_WINDOWS_NAMES:
        return key
    if len(key) == 4 and key[:3] in {"com", "lpt"} and key[3] in "123456789":
        return key
    return None


def validate_identifier(value, *, label="identifier"):
    """Validate and return a canonical, portable single path segment.

    The accepted alphabet intentionally matches the existing process-name contract:
    Unicode letters/digits, spaces, ``-`` and ``_``.  The conservative alphabet also
    excludes every path separator, drive marker, control character and Windows device
    name without having to branch on the host operating system.
    """
    if not isinstance(value, str) or not value:
        raise IdentifierError(f"{label} is verplicht.")

    normalized = normalize_identifier(value)
    if not normalized or normalized != normalized.strip():
        raise IdentifierError(_invalid_characters_message())
    if len(normalized) > IDENTIFIER_MAX_LENGTH:
        raise IdentifierError(
            f"{label} mag maximaal {IDENTIFIER_MAX_LENGTH} tekens bevatten."
        )

    if any(not (character.isalnum() or character in "-_ ") for character in normalized):
        raise IdentifierError(_invalid_characters_message())

    reserved = _reserved_name(normalized)
    if reserved is not None:
        raise IdentifierError(
            f"{label} '{normalized}' is niet toegestaan op alle platformen "
            f"(gereserveerde Windows-naam: {reserved.upper()})."
        )

    return normalized
