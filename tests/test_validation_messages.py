"""Unit tests for aliexpress.validation_messages — Dutch error-message formatters.

These pin the Dutch strings shown to teachers when uploads fail; a refactor must not
silently change the UI text.
"""

import numpy as np
import pandas as pd
import pandera as pa
from werkzeug.exceptions import RequestEntityTooLarge

from aliexpress.errors import ValidationError
from aliexpress.validation_messages import (
    readableerror_to_validation_message,
    schemaerror_to_validation_message,
    to_validation_message,
)


class TestErrorMessages:
    """Unit tests for the user-facing Dutch error messages produced by app.py helpers.

    These functions are critical because they determine what Dutch text teachers see
    when their uploads fail. Tests here pin the Dutch strings so a refactor cannot
    silently change what the UI shows.
    """

    def test_unknown_exception_returns_generic_fallback(self):
        """A generic exception not in any known category returns the Dutch fallback."""
        msg = to_validation_message(RuntimeError("anything"))
        assert "onverwachts" in msg

    def test_request_entity_too_large_returns_size_message(self):
        """An oversized upload (HTTP 413) yields the friendly 'too large' message."""
        msg = to_validation_message(RequestEntityTooLarge())
        assert "te groot" in msg

    def test_readable_error_with_known_code_returns_dutch_template(self):
        """'wrong_columns_preferences' ValidationError returns the Dutch column-error template."""
        exc = ValidationError(
            "wrong_columns_preferences", {"wrong_columns": "Kolom A, Kolom B"}
        )
        msg = readableerror_to_validation_message(exc)
        assert "verkeerde kolommen" in msg
        assert "Kolom A, Kolom B" in msg

    def test_readable_error_with_unknown_code_returns_fallback(self):
        """A ValidationError with an unrecognised code falls back to the generic Dutch message."""
        exc = ValidationError("some_unknown_code", {})
        msg = readableerror_to_validation_message(exc)
        assert "onverwachts" in msg

    def test_too_few_students_not_together_returns_correct_dutch_text(self):
        """'too_few_students_not_together' error mentions the rule index and student minimum."""
        exc = ValidationError("too_few_students_not_together", {"rule_index": 2})
        msg = readableerror_to_validation_message(exc)
        assert "Niet-samen-regel 2" in msg
        assert "minstens 2 leerlingen" in msg

    def test_unknown_student_not_together_returns_student_name(self):
        """'unknown_student_not_together' error includes the unknown student names."""
        exc = ValidationError(
            "unknown_student_not_together", {"unknown_students": "Jan Jansen"}
        )
        msg = readableerror_to_validation_message(exc)
        assert "Jan Jansen" in msg

    def test_too_many_niet_in_form_names_student_and_cap(self):
        """'too_many_niet_in_form' mentions the student and the maximum exclusions."""
        exc = ValidationError(
            "too_many_niet_in_form",
            {"leerling": "Jan", "max_niet_in": 2, "n_groepen": 3},
        )
        msg = readableerror_to_validation_message(exc)
        assert "Jan" in msg and "2" in msg and "3" in msg

    def test_invalid_gewicht_form_names_student_and_weight(self):
        """'invalid_gewicht_form' mentions the student and the offending weight."""
        exc = ValidationError(
            "invalid_gewicht_form", {"leerling": "Jan", "gewicht": 0.0}
        )
        msg = readableerror_to_validation_message(exc)
        assert "Jan" in msg and "groter dan 0" in msg

    def test_invalid_min_tevredenheid_form_names_student(self):
        """'invalid_min_tevredenheid_form' mentions the student and the <= 1 bound."""
        exc = ValidationError(
            "invalid_min_tevredenheid_form",
            {"leerling": "Jan", "minimale_tevredenheid": 1.5},
        )
        msg = readableerror_to_validation_message(exc)
        assert "Jan" in msg and "1" in msg

    @staticmethod
    def _nulls_schema_error():
        """A real pandera SERIES_CONTAINS_NULLS error, as a missing value would raise."""
        schema = pa.DataFrameSchema({"Waarde": pa.Column(str, nullable=False)})
        try:
            schema.validate(pd.DataFrame({"Waarde": [np.nan]}))
        except pa.errors.SchemaError as exc:
            exc.filetype = "voorkeuren"
            return exc
        raise AssertionError("expected a SchemaError")

    def test_missing_value_names_the_student_as_entered(self):
        """A missing required value (e.g. a Gewicht without a wish) yields a friendly
        message naming the student as entered, instead of a 500 (regression test)."""
        exc = self._nulls_schema_error()
        # datareader attaches the offending students by the name as entered.
        exc.offending_students = ["Bob B"]
        msg = schemaerror_to_validation_message(exc)
        assert "Bob B" in msg
        assert "voorkeuren" in msg

    def test_missing_value_without_student_context_does_not_crash(self):
        """Without attached students the message still renders (no KeyError/500)."""
        msg = schemaerror_to_validation_message(self._nulls_schema_error())
        assert "voorkeuren" in msg
