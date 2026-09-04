"""Unit tests for aliexpress.validation_messages — Dutch error-message formatters.

These pin the Dutch strings shown to teachers when uploads fail; a refactor must not
silently change the UI text.
"""

# The detailed payload is repeated in the web-flow and logging invariant tests so
# each public boundary can be verified independently.
# pylint: disable=duplicate-code

# pylint: disable=protected-access  # tests call _validate_input directly (same as test_datareader)

from unittest.mock import patch

import numpy as np
import pandas as pd
import pandera as pa
import pytest
from werkzeug.exceptions import RequestEntityTooLarge

from aliexpress.data import datareader
from aliexpress.errors import FeasibilityError, ValidationError
from aliexpress.web.validation_messages import (
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
        """'invalid_min_tevredenheid_form' mentions the student and the 100% bound."""
        exc = ValidationError(
            "invalid_min_tevredenheid_form",
            {"leerling": "Jan", "minimale_tevredenheid": 1.5},
        )
        msg = readableerror_to_validation_message(exc)
        assert "Jan" in msg and "100%" in msg

    def test_self_preference_form_names_student(self):
        """A self-target gets a direct, actionable form message."""
        exc = ValidationError("self_preference_form", {"leerling": "Anna Bos"})
        msg = readableerror_to_validation_message(exc)

        assert "Anna Bos" in msg
        assert "zichzelf" in msg

    def test_balance_caps_too_tight_single_change_is_natural_dutch(self):
        """A single proposed increase gets a natural singular message."""
        exc = FeasibilityError(
            "balance_caps_too_tight",
            context={
                "suggestion": {
                    "diff_total": {"current": 4, "suggested": 5},
                }
            },
        )

        msg = readableerror_to_validation_message(exc)

        assert msg == (
            "Met deze grenzen is geen geldige indeling mogelijk. Een mogelijke minimale "
            "verruiming is: verhoog ‘Groepsgrootte totaal’ van 4 naar 5 (+1). "
            "Mogelijk werkt ook een andere combinatie."
        )

    def test_balance_caps_too_tight_multiple_changes_explain_the_joint_set(self):
        """Multiple increases use 'én' and explicitly belong together."""
        exc = FeasibilityError(
            "balance_caps_too_tight",
            context={
                "suggestion": {
                    "diff_total": {"current": 4, "suggested": 5},
                    "gender_total": {"current": 3, "suggested": 5},
                }
            },
        )

        msg = readableerror_to_validation_message(exc)

        assert "‘Groepsgrootte totaal’ van 4 naar 5 (+1)" in msg
        assert "én ‘Jongens/meisjes totaal’ van 3 naar 5 (+2)" in msg
        assert "Deze aanpassingen horen bij elkaar." in msg
        assert "Mogelijk werkt ook een andere combinatie." in msg

    def test_balance_caps_too_tight_uses_all_dutch_balance_names(self):
        """Every internal family is rendered with its Dutch UI label."""
        labels = {
            "diff_year": "Groepsgrootte per jaarlaag",
            "diff_total": "Groepsgrootte totaal",
            "gender_year": "Jongens/meisjes per jaarlaag",
            "gender_total": "Jongens/meisjes totaal",
            "clique": "Zelfde stamgroep totaal",
            "clique_sex": "Zelfde stamgroep per sekse",
        }
        exc = FeasibilityError(
            "balance_caps_too_tight",
            context={
                "suggestion": {
                    family: {"current": 1, "suggested": 2} for family in labels
                }
            },
        )

        msg = readableerror_to_validation_message(exc)

        for label in labels.values():
            assert f"‘{label}’" in msg

    def test_infeasible_preferences_uses_valid_grouping_language(self):
        """The fallback describes validity, not an unexplained balance objective."""
        exc = FeasibilityError(
            "infeasible_preferences", context={"case": "min_satisfaction"}
        )

        msg = readableerror_to_validation_message(exc)

        assert msg.startswith(
            "Met deze voorkeuren bestaat geen geldige groepsindeling."
        )
        assert "evenwichtige groepsindeling" not in msg

    def test_detailed_small_conflict_has_neutral_fixed_order_and_context(self):
        """A small core explains floors, raw wishes, rules and merged exclusions."""
        exc = FeasibilityError(
            "infeasible_preferences",
            context={
                "case": "detailed",
                "conflict": {
                    "conditions": [
                        {
                            "type": "forbidden_group",
                            "student": "Piet",
                            "group": "Blauw",
                        },
                        {
                            "type": "not_together",
                            "rule_index": 2,
                            "students": ["Piet", "Sam", "Noor"],
                            "max_together": 1,
                        },
                        {
                            "type": "forbidden_group",
                            "student": "Piet",
                            "group": "Rood",
                        },
                        {
                            "type": "minimum_satisfaction",
                            "student": "Piet",
                            "floor": 1.0,
                            "preferences": [
                                {
                                    "kind": "Graag met",
                                    "target": "Sam",
                                    "weight": 1.0,
                                },
                                {
                                    "kind": "Liever niet met",
                                    "target": "Noor",
                                    "weight": -2.0,
                                },
                            ],
                        },
                    ]
                },
            },
        )

        msg = readableerror_to_validation_message(exc)

        assert msg.startswith(
            "Met deze voorkeuren bestaat geen geldige groepsindeling."
        )
        assert "Piet" in msg
        assert "Alle voorkeuren gehonoreerd" in msg
        assert "Graag met Sam" in msg
        assert "Liever niet met Noor" in msg
        assert "Niet-samen-regel 2" in msg
        assert "maximaal 1" in msg
        assert "Piet mag niet in Blauw en Rood" in msg
        assert msg.index("extra zekerheid") < msg.index("Niet-samen-regel 2")
        assert msg.index("Niet-samen-regel 2") < msg.index("Piet mag niet in Blauw")
        assert "niet tegelijk uitvoerbaar" in msg
        assert "oorzaak" not in msg.lower()

    def test_detailed_large_conflict_is_a_concrete_inventory(self):
        """A large core names involved inputs without inventing a causal explanation."""
        conditions = [
            {
                "type": "minimum_satisfaction",
                "student": student,
                "floor": 1.0,
                "preferences": [],
            }
            for student in ["Anna", "Bram", "Claire", "Daan", "Eva"]
        ]
        conditions += [
            {
                "type": "forbidden_group",
                "student": student,
                "group": "Blauw",
            }
            for student in ["Anna", "Bram", "Claire", "Daan"]
        ]
        msg = readableerror_to_validation_message(
            FeasibilityError(
                "infeasible_preferences",
                context={"case": "detailed", "conflict": {"conditions": conditions}},
            )
        )

        assert "te groot" in msg
        assert "extra zekerheid" in msg
        assert "Anna" in msg and "Eva" in msg
        assert "Niet in" in msg
        assert "Graag met" not in msg

    def test_detailed_conflict_shows_form_label_for_negative_preference(self):
        """Negative form preferences keep their user-facing type in the context."""
        msg = readableerror_to_validation_message(
            FeasibilityError(
                "infeasible_preferences",
                context={
                    "case": "detailed",
                    "conflict": {
                        "conditions": [
                            {
                                "type": "minimum_satisfaction",
                                "student": "Piet",
                                "floor": 0.5,
                                "preferences": [
                                    {
                                        "kind": "Liever niet met",
                                        "target": "Sam",
                                        "weight": 1.0,
                                    }
                                ],
                            }
                        ]
                    },
                },
            )
        )

        assert "Minstens tevreden" in msg
        assert "Liever niet met Sam" in msg

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


class TestSchemaErrorMessages:
    """Tests for schemaerror_to_validation_message branches not covered by TestErrorMessages.

    Each test triggers a REAL pandera SchemaError through the actual datareader validation
    path (approach B), so the test verifies the real contract between datareader (which
    raises) and validation_messages (which translates). Building synthetic SchemaErrors
    would drift from reality and give false confidence.
    """

    @staticmethod
    def _valid_voorkeuren_df():
        """Minimal valid wide-format voorkeuren DataFrame, same structure as datareader expects."""
        header = [
            ("MinimaleTevredenheid", np.nan, np.nan),
            ("Jongen/meisje", np.nan, np.nan),
            ("Stamgroep", np.nan, np.nan),
            ("Graag met", 1.0, "Waarde"),
            ("Graag met", 1.0, "Gewicht"),
            ("Graag met", 2.0, "Waarde"),
            ("Graag met", 2.0, "Gewicht"),
            ("Graag met", 3.0, "Waarde"),
            ("Graag met", 3.0, "Gewicht"),
            ("Graag met", 4.0, "Waarde"),
            ("Graag met", 4.0, "Gewicht"),
            ("Graag met", 5.0, "Waarde"),
            ("Graag met", 5.0, "Gewicht"),
            ("Liever niet met", 1.0, "Waarde"),
            ("Liever niet met", 1.0, "Gewicht"),
            ("Niet in", 1.0, "Waarde"),
            ("Niet in", 2.0, "Waarde"),
        ]
        columns = pd.MultiIndex.from_tuples(
            header, names=["TypeWens", "Nr", "TypeWaarde"]
        )
        data = [
            [
                np.nan,
                "Jongen",
                "a",
                "jane",
                1,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            [
                np.nan,
                "Meisje",
                "a",
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
        ]
        return pd.DataFrame(
            data,
            columns=columns,
            index=pd.Index(["john", "jane"], name="Leerling"),
        )

    def test_column_not_in_schema_returns_wrong_columns_message(self):
        """An extra column in the groepen file raises COLUMN_NOT_IN_SCHEMA; the message
        names the filetype and says 'verkeerde kolommen'.

        This is the first thing a teacher sees when they upload the wrong file (e.g. the
        voorkeuren sheet instead of groepen). The message must be clear about which file
        is wrong and what to check.
        """
        with patch("aliexpress.data.datareader.pd.read_excel") as mock_read:
            mock_read.return_value = pd.DataFrame(
                {"Groepen": ["Rood"], "Jongens": [5], "Meisjes": [6], "ExtraKolom": [0]}
            )
            try:
                datareader.read_groups_excel("groepen.xlsx")
            except pa.errors.SchemaError as exc:
                msg = schemaerror_to_validation_message(exc)
                assert "groepen" in msg
                assert "verkeerde kolommen" in msg
            else:
                pytest.fail("expected a SchemaError for extra column")

    def test_datatype_coercion_names_the_column_and_filetype(self):
        """A non-coercible cell in a typed column raises DATATYPE_COERCION; the message
        names the column and the filetype.

        Happens when e.g. MinimaleTevredenheid contains text that cannot be parsed as a
        float. The message tells the teacher which column to fix.
        """
        df = self._valid_voorkeuren_df().copy().astype(object)
        df.loc["john", ("MinimaleTevredenheid", np.nan, np.nan)] = "tekst"
        processor = datareader.VoorkeurenProcessor.__new__(
            datareader.VoorkeurenProcessor
        )
        try:
            processor._validate_input(df)
        except pa.errors.SchemaError as exc:
            msg = schemaerror_to_validation_message(exc)
            assert "MinimaleTevredenheid" in msg
            assert "voorkeuren" in msg
        else:
            pytest.fail("expected a SchemaError for wrong datatype")

    def test_empty_voorkeuren_yields_empty_df_message(self):
        """An empty voorkeuren sheet raises DATAFRAME_CHECK/empty_df; message names the filetype.

        This branch matters because teachers sometimes upload an empty template; they must
        get a clear Dutch message rather than a cryptic 500.
        """
        df = self._valid_voorkeuren_df().iloc[:0]  # empty, but valid columns
        processor = datareader.VoorkeurenProcessor.__new__(
            datareader.VoorkeurenProcessor
        )
        try:
            processor._validate_input(df)
        except pa.errors.SchemaError as exc:
            msg = schemaerror_to_validation_message(exc)
            assert "leeg" in msg
            assert "voorkeuren" in msg
        else:
            pytest.fail("expected a SchemaError for empty df")

    def test_empty_groepen_yields_empty_df_message(self):
        """An empty groepen sheet raises DATAFRAME_CHECK/empty_df; message names the filetype.

        Same check as voorkeuren but for the groepen file — verifies the branch works for
        both filetypes, not just voorkeuren.
        """
        with patch("aliexpress.data.datareader.pd.read_excel") as mock_read:
            mock_read.return_value = pd.DataFrame(
                columns=["Groepen", "Jongens", "Meisjes"]
            )
            try:
                datareader.read_groups_excel("groepen.xlsx")
            except pa.errors.SchemaError as exc:
                msg = schemaerror_to_validation_message(exc)
                assert "leeg" in msg
                assert "groepen" in msg
            else:
                pytest.fail("expected a SchemaError for empty groepen df")

    def test_wrong_sex_value_names_the_student(self):
        """A non-'Jongen'/'Meisje' sex value raises DATAFRAME_CHECK/isin on the Jongen/meisje
        column; the message must name the offending student(s).

        This is the most common upload mistake: teachers who fill in 'M'/'V' or 'man'/'vrouw'
        instead of the expected Dutch labels get a clear hint.
        """
        df = self._valid_voorkeuren_df().copy()
        df.loc["john", ("Jongen/meisje", np.nan, np.nan)] = "Alien"
        processor = datareader.VoorkeurenProcessor.__new__(
            datareader.VoorkeurenProcessor
        )
        try:
            processor._validate_input(df)
        except pa.errors.SchemaError as exc:
            msg = schemaerror_to_validation_message(exc)
            assert "john" in msg
            assert "geslacht" in msg.lower()
        else:
            pytest.fail("expected a SchemaError for wrong sex value")

    def test_duplicated_values_preferences_returns_friendly_message(self):
        """Listing the same classmate twice for one student raises
        DATAFRAME_CHECK/duplicated_values_preferences; the message must not crash and
        must mention 'voorkeuren' and 'dubbel'.

        The check function returns a single bool so pandera cannot expose the offending
        student names via failure_cases — the message is therefore generic but still
        actionable. This test guards against the 'bool is not subscriptable' crash that
        would otherwise produce a 500 in the teacher's browser.
        """
        df = self._valid_voorkeuren_df().copy().astype(object)
        # Make john list 'jane' twice: slot 1 and slot 2 both point to jane.
        df.loc["john", ("Graag met", 2.0, "Waarde")] = "jane"
        df.loc["john", ("Graag met", 2.0, "Gewicht")] = 1.0

        processor = datareader.VoorkeurenProcessor.__new__(
            datareader.VoorkeurenProcessor
        )
        processor.input = df
        processor.df = df.copy()
        processor.restructure()

        try:
            processor.validate_preferences(all_to_groups=["blauw"])
        except pa.errors.SchemaError as exc:
            msg = schemaerror_to_validation_message(exc)
            assert "dubbel" in msg.lower()
            assert "voorkeuren" in msg
        else:
            pytest.fail("expected a SchemaError for duplicated preference values")

    def test_invalid_values_preferences_names_the_unknown_target(self):
        """A wish aimed at an unknown student/group raises
        DATAFRAME_CHECK/invalid_values_preferences; the message names the unknown value.

        When a teacher types a wrong name (typo, old classmate) the message must show
        which value failed so they know exactly what to fix.
        """
        df = self._valid_voorkeuren_df().copy()
        # john wishes for 'onbekend' — not in the known students or groups list.
        df.loc["john", ("Graag met", 1.0, "Waarde")] = "onbekend"

        processor = datareader.VoorkeurenProcessor.__new__(
            datareader.VoorkeurenProcessor
        )
        processor.input = df
        processor.df = df.copy()
        processor.restructure()

        try:
            # 'jane' is a known student; 'onbekend' is not a student or group.
            processor.validate_preferences(all_to_groups=["blauw"])
        except pa.errors.SchemaError as exc:
            msg = schemaerror_to_validation_message(exc)
            assert "onbekend" in msg
        else:
            pytest.fail("expected a SchemaError for invalid preference target")

    def test_duplicate_index_voorkeuren_names_the_duplicate(self):
        """Two rows with the same student name in voorkeuren raises
        SERIES_CONTAINS_DUPLICATES (voorkeuren branch); the message names the duplicate.

        Teachers who copy a row by accident get a message naming the repeated student so
        they know which duplicate to remove.
        """
        df = self._valid_voorkeuren_df()
        df_dup = pd.concat([df, df.iloc[:1]])  # john appears twice
        processor = datareader.VoorkeurenProcessor.__new__(
            datareader.VoorkeurenProcessor
        )
        try:
            processor._validate_input(df_dup)
        except pa.errors.SchemaError as exc:
            msg = schemaerror_to_validation_message(exc)
            assert "john" in msg
            assert "uniek" in msg.lower() or "niet uniek" in msg.lower()
        else:
            pytest.fail("expected a SchemaError for duplicated student index")

    def test_negative_gewicht_returns_friendly_message(self):
        """A negative weight in a Gewicht column raises DATAFRAME_CHECK/greater_than;
        the message must mention 'gewichten' and 'voorkeurenbestand'.

        When a teacher types a negative weight (e.g. '-1') in the Gewicht column the
        formatter must give a clear Dutch message rather than crashing.
        """
        df = self._valid_voorkeuren_df().copy()
        df.loc["john", ("Graag met", 1.0, "Gewicht")] = -1

        processor = datareader.VoorkeurenProcessor.__new__(
            datareader.VoorkeurenProcessor
        )
        processor.input = df
        processor.df = df.copy()
        processor.student_display = {"john": "john", "jane": "jane"}
        processor.restructure()

        try:
            processor.validate_preferences(all_to_groups=["blauw"])
        except pa.errors.SchemaError as exc:
            msg = schemaerror_to_validation_message(exc)
            assert "gewichten" in msg.lower()
            assert "voorkeurenbestand" in msg.lower()
        else:
            pytest.fail("expected a SchemaError for negative gewicht")

    def test_duplicate_group_name_in_groepen_returns_non_voorkeuren_message(self):
        """A duplicate group name in the groepen file raises SERIES_CONTAINS_DUPLICATES
        (non-voorkeuren branch); the message names the filetype and the duplicate column.

        The groepen file has unique=True on the Groepen column. If a teacher lists
        the same class name twice the message must not say 'voorkeuren' (wrong file).
        """
        with patch("aliexpress.data.datareader.pd.read_excel") as mock_read:
            mock_read.return_value = pd.DataFrame(
                {"Groepen": ["Rood", "Rood"], "Jongens": [5, 6], "Meisjes": [6, 7]}
            )
            try:
                datareader.read_groups_excel("groepen.xlsx")
            except pa.errors.SchemaError as exc:
                msg = schemaerror_to_validation_message(exc)
                assert "groepen" in msg
                assert "voorkeuren" not in msg
                assert "dubbeling" in msg.lower() or "Groepen" in msg
            else:
                pytest.fail("expected a SchemaError for duplicate group names")
