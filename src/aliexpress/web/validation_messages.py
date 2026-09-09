"""User-facing Dutch error messages for upload validation failures.

Pure text formatters — no Flask or logging dependencies. Called from app.py by
``_flash_upload_error`` and ``_handle_failure``.
"""

import numpy as np
import pandera as pa
from werkzeug.exceptions import RequestEntityTooLarge

from ..errors import CouldNotReadFileError, FeasibilityError, ValidationError


def to_validation_message(exc: Exception) -> str:
    """Convert a validation exception to a user-friendly message"""
    if isinstance(exc, RequestEntityTooLarge):
        return "Het bestand is te groot om te uploaden. Kies een kleiner bestand."
    if isinstance(exc, pa.errors.SchemaError):
        return schemaerror_to_validation_message(exc)
    if isinstance(exc, (ValidationError, CouldNotReadFileError, FeasibilityError)):
        return readableerror_to_validation_message(exc)
    return (
        "Er is iets onverwachts misgegaan. Het probleem is gelogd. "
        "Laat de maker dit onderzoeken."
    )


def _format_infeasible_preferences(context: dict) -> str:
    """Compose the Dutch message for infeasible preferences (ADR-0008).

    ``context["case"]`` names the family that must give, found robustly at family level
    (not a single arbitrary student/rule from a degenerate minimum). Each case states the
    constraint family to relax — the extra zekerheid (minimal satisfaction) and/or the
    niet-samen rules — without pointing at individual students.
    """
    conflict = context.get("conflict")
    if isinstance(conflict, dict) and isinstance(conflict.get("conditions"), list):
        return _format_detailed_conflict(conflict["conditions"])

    case = context.get("case", "fundamental")
    header = "Met deze voorkeuren bestaat geen geldige groepsindeling."

    verlaag_zekerheid = (
        "verlaag de extra zekerheid een stap "
        "bij de leerlingen waar je die hebt ingesteld"
    )
    versoepel_regel = (
        "versoepel een niet-samen-regel "
        "(sta meer leerlingen samen toe, of haal er een uit)"
    )

    if case == "min_satisfaction":
        return (
            f"{header} De gevraagde extra zekerheid is te streng: {verlaag_zekerheid}."
        )
    if case == "not_together":
        return f"{header} De niet-samen-regels zijn te streng: {versoepel_regel}."
    if case == "either":
        return (
            f"{header} Je kunt het op twee manieren oplossen — één is genoeg: "
            f"{verlaag_zekerheid}, óf {versoepel_regel}."
        )
    if case == "both":
        return (
            f"{header} De extra zekerheid en de niet-samen-regels botsen samen; versoepel "
            f"ze allebei: {verlaag_zekerheid}, en {versoepel_regel}."
        )
    return (
        f"{header} Het lukt ook niet door de extra zekerheid of de niet-samen-regels te "
        "versoepelen. Waarschijnlijk botsen de 'Niet in'-uitsluitingen: controleer of "
        "leerlingen niet uit te veel groepen geweigerd worden."
    )


def _format_detailed_conflict(conditions: list[dict], small_core_limit: int = 8) -> str:
    """Format a proven detail core without adding causal or repair advice.

    Cores up to ``small_core_limit`` are written condition by condition.  Larger valid
    cores use the compact inventory format below.
    """
    if len(conditions) > small_core_limit:
        return _format_large_conflict(conditions)

    minimums = [c for c in conditions if c.get("type") == "minimum_satisfaction"]
    rules = [c for c in conditions if c.get("type") == "not_together"]
    forbidden = [c for c in conditions if c.get("type") == "forbidden_group"]
    lines = ["Met deze voorkeuren bestaat geen geldige groepsindeling."]

    for condition in minimums:
        lines.append(
            f"- {condition['student']} heeft {_floor_label(condition['floor'])}."
        )
        preferences = condition.get("preferences", [])
        lines.append("  Gewone voorkeuren die de tevredenheid bepalen:")
        if preferences:
            for preference in preferences:
                lines.append(
                    "    - "
                    f"{preference.get('kind', 'Voorkeur')} "
                    f"{preference.get('target')} "
                    f"(gewicht {preference.get('weight')})."
                )
        else:
            lines.append("    - Geen gewone voorkeuren.")

    for condition in rules:
        students = _join_dutch(condition.get("students", []))
        lines.append(
            f"- Niet-samen-regel {condition['rule_index']}: maximaal "
            f"{condition['max_together']} van {students} samen in één groep."
        )

    exclusions = {}
    for condition in forbidden:
        exclusions.setdefault(condition["student"], []).append(condition["group"])
    for student, groups in exclusions.items():
        lines.append(f"- {student} mag niet in {_join_dutch(groups)}.")

    lines.append(
        "Deze voorwaarden zijn niet tegelijk uitvoerbaar. Er kunnen nog andere conflicten bestaan."
    )
    return "\n".join(lines)


def _format_large_conflict(conditions: list[dict]) -> str:
    """Format a valid large core as a concrete inventory of involved input."""
    rules = sorted(
        {
            condition["rule_index"]
            for condition in conditions
            if condition.get("type") == "not_together"
        }
    )
    minimum_students = list(
        dict.fromkeys(
            condition["student"]
            for condition in conditions
            if condition.get("type") == "minimum_satisfaction"
        )
    )
    forbidden_students = list(
        dict.fromkeys(
            condition["student"]
            for condition in conditions
            if condition.get("type") == "forbidden_group"
        )
    )
    inventory = []
    if rules:
        inventory.append(
            "niet-samen-regel " + _join_dutch([str(rule_index) for rule_index in rules])
        )
    if minimum_students:
        inventory.append("extra zekerheid van " + _join_dutch(minimum_students))
    if forbidden_students:
        inventory.append("Niet in-uitsluitingen van " + _join_dutch(forbidden_students))
    return (
        "Het gevonden conflict is te groot om volledig uit te schrijven. "
        f"Het betreft {_join_dutch(inventory)}. "
        "Deze voorwaarden zijn niet tegelijk uitvoerbaar. Er kunnen nog andere conflicten bestaan."
    )


def _floor_label(value) -> str:
    """Use the labels shown by the form for the two supported floor levels."""
    if float(value) >= 1.0:
        return "de extra zekerheid 'Alle voorkeuren gehonoreerd'"
    return "de extra zekerheid 'Minstens tevreden'"


def _join_dutch(values) -> str:
    """Join already formatted values with Dutch punctuation."""
    values = [str(value) for value in values]
    if len(values) < 2:
        return values[0] if values else "geen leerlingen"
    return ", ".join(values[:-1]) + " en " + values[-1]


_BALANCE_CAP_LABELS = {
    "diff_year": "Groepsgrootte per jaarlaag",
    "diff_total": "Groepsgrootte totaal",
    "gender_year": "Jongens/meisjes per jaarlaag",
    "gender_total": "Jongens/meisjes totaal",
    "clique": "Zelfde stamgroep totaal",
    "clique_sex": "Zelfde stamgroep per sekse",
}


def _format_balance_caps_too_tight(context: dict) -> str:
    """Compose the Dutch message for a joint balance-cap suggestion."""
    changes = []
    for family, label in _BALANCE_CAP_LABELS.items():
        change = context.get("suggestion", {}).get(family)
        if change is None:
            continue
        current = change["current"]
        suggested = change["suggested"]
        changes.append(
            f"‘{label}’ van {current} naar {suggested} (+{suggested - current})"
        )

    header = "Met deze grenzen is geen geldige indeling mogelijk."
    if len(changes) == 1:
        return (
            f"{header} Een mogelijke minimale verruiming is: verhoog {changes[0]}. "
            "Mogelijk werkt ook een andere combinatie."
        )
    if changes:
        return (
            f"{header} Een mogelijke minimale verruiming is: verhoog {' én '.join(changes)}. "
            "Deze aanpassingen horen bij elkaar. Mogelijk werkt ook een andere combinatie."
        )
    return (
        f"{header} Mogelijk zijn de ingestelde balansgrenzen te krap. "
        "Mogelijk werkt ook een andere combinatie."
    )


def readableerror_to_validation_message(exc: Exception) -> str:
    """Convert a validation exception to a user-friendly message"""
    if exc.code == "infeasible_preferences":
        return _format_infeasible_preferences(exc.context)
    if exc.code == "balance_caps_too_tight":
        return _format_balance_caps_too_tight(exc.context)
    friendly_templates = {
        "wrong_columns_preferences": (
            "Het voorkeuren-bestand heeft de verkeerde kolommen. Controleer of je het goede"
            " bestand hebt geupload en het meest recente template hebt gebruikt. "
            "\n{wrong_columns}"
        ),
        "infeasible_problem": (
            "Met deze vereiste klassenbalans en verdeling van leerlingen die overgaan is het"
            "niet mogelijk. Overweeg de volgende versoepelingen om het probleem wel op te "
            "lossen:\n {possible_improvement}"
        ),
        "internal_error": (
            "Er is iets onverwachts misgegaan. Het probleem is gelogd. "
            "Laat de maker dit onderzoeken."
        ),
        "duplicate_student_not_together": (
            "Deze leerling is al aan deze spreiding toegevoegd. Kies een andere leerling."
        ),
        "missing_max_samen_not_together": ("Vul het maximum in voor deze spreiding."),
        "too_few_students_not_together": (
            "Voeg minimaal twee leerlingen aan deze spreiding toe."
        ),
        "invalid_max_samen_not_together": ("Het maximum moet minimaal 1 zijn."),
        "unknown_student_not_together": (
            "Kies leerlingen uit de deelnemers aan deze groepsindeling."
        ),
        "too_strict_not_together": (
            "Deze spreiding is niet uitvoerbaar: {n_students} leerlingen kunnen niet over "
            "{n_groups} nieuwe groepen worden verdeeld met maximaal {max_samen} van hen "
            "samen. Verhoog het maximum of pas de leerlingen aan."
        ),
        "invalid_max_samen_type_not_together": (
            "Vul een geheel getal van minimaal 1 in als maximum."
        ),
        "too_many_niet_in_form": (
            "{leerling} mag niet in te veel groepen geweigerd worden: met {n_groepen} "
            "groepen kun je er maximaal {max_niet_in} uitsluiten, anders is er geen "
            "groep meer over."
        ),
        # Used by the route layer (Stap 3) to validate the raw form input before a
        # Preference is constructed; the dataclass itself also rejects a weight <= 0.
        "invalid_gewicht_form": (
            "{leerling} heeft een voorkeur met gewicht {gewicht}. Een gewicht moet groter "
            "dan 0 zijn."
        ),
        "invalid_min_tevredenheid_form": (
            "{leerling} heeft een te hoge minimale tevredenheid. "
            "Die mag hoogstens 100% zijn."
        ),
        "self_preference_form": (
            "{leerling} kan geen voorkeur voor zichzelf opgeven. Kies een andere "
            "leerling of groep."
        ),
        # Used by the roster step ("Leerlingen controleren") when validating hand-added students.
        "incomplete_new_student": ("Vul de voornaam, achternaam en het geslacht in."),
        "open_new_student": (
            "Bevestig de leerling met ‘Leerling aan de lijst toevoegen’ of verwijder de invoer."
        ),
        "no_students_selected": ("Selecteer ten minste één leerling die doorgaat."),
        "duplicate_new_student": (
            "Er staat al een leerling met de naam ‘{name}’ in de lijst."
        ),
        "missing_groep_new_student": ("Kies de huidige groep, of kies ‘Anders’."),
        "invalid_groep_new_student": (
            "Kies een huidige groep uit de lijst, of kies ‘Anders’."
        ),
        "missing_jaargroep_new_student": ("Kies ook de huidige jaarlaag."),
        "duplicate_group_names": (
            "Groepsnamen moeten uniek zijn. Dubbel: {duplicates}."
        ),
        "missing_balance_maximum": (
            "Vul bij elke klassenbalans-grens een geheel getal van minimaal 1 in, "
            "of vink Onbeperkt aan."
        ),
        "invalid_balance_maximum": (
            "Een klassenbalans-grens moet een geheel getal van minimaal 1 zijn, "
            "of Onbeperkt."
        ),
    }

    template = friendly_templates.get(exc.code, None)
    if template:
        return template.format(**exc.context)
    return (
        "Er is iets onverwachts misgegaan. Het probleem is gelogd. "
        "Laat de maker dit onderzoeken."
    )


# Deliberately overruling pylint here; we need a branch per validation
# pylint: disable=too-many-return-statements, too-many-branches
def schemaerror_to_validation_message(exc: pa.errors.SchemaError) -> str:
    """Convert a pandera SchemaError to a user-friendly message

    This SchemaError must have been modified to contain a 'filetype' attribute.
    """
    if exc.reason_code in (
        pa.errors.SchemaErrorReason.COLUMN_NOT_IN_SCHEMA,
        pa.errors.SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME,
    ):
        return (
            f"Het {exc.filetype}-bestand heeft de verkeerde kolommen. Controleer of je het goede"
            " bestand hebt geupload en het meest recente template hebt gebruikt. "
            f"\n{exc.failure_cases}"
        )
    if exc.reason_code == pa.errors.SchemaErrorReason.DATATYPE_COERCION:
        return (
            f"Ongeldige waarden gevonden in kolom {exc.schema.name} "
            f"van het {exc.filetype}-bestand"
        )
    if exc.reason_code == pa.errors.SchemaErrorReason.SERIES_CONTAINS_NULLS:
        students = getattr(exc, "offending_students", [])
        if students:
            return (
                f"In het {exc.filetype}-bestand mist een waarde bij: "
                f"{', '.join(students)}. Vul bij elke voorkeur een naam of groep in, of haal "
                "het bijbehorende gewicht weg als er geen voorkeur is."
            )
        return (
            f"In het {exc.filetype}-bestand zijn niet alle verplichte velden gevuld "
            f"(kolom {exc.column_name})."
        )
    if exc.reason_code == pa.errors.SchemaErrorReason.SERIES_CONTAINS_DUPLICATES:
        if exc.filetype == "voorkeuren":
            duplicates = ", ".join(exc.failure_cases["failure_case"])
            return (
                f"In voorkeuren is de volgende naam/namen niet uniek: {duplicates}\n"
                "Voeg de eerste letter van de achternaam toe om de leerlingen van "
                "elkaar te onderscheiden."
            )
        return (
            f"In het {exc.filetype}-bestand zijn dubbelingen ingevuld "
            f"in kolom {exc.column_name}"
        )

    if exc.reason_code == pa.errors.SchemaErrorReason.DATAFRAME_CHECK:
        if exc.check.name == "empty_df":
            return (
                f"Het {exc.filetype}-bestand was helemaal leeg. Daardoor kan er "
                "geen groepsindeling worden berekend"
            )
        if exc.column_name == ("Jongen/meisje", np.nan, np.nan):
            return f"Verkeerd ingevuld geslacht voor {', '.join(exc.failure_cases['index'])}"
        if exc.check.name == "greater_than" and "Gewicht" in exc.column_name:
            return "Er zijn negatieve gewichten in het voorkeurenbestand."
        if exc.check.name == "duplicated_values_preferences":
            # The check function returns a single bool (not per-row), so pandera
            # stores the bool as failure_cases — we cannot extract student names.
            return (
                "In het voorkeuren-bestand is een leerling of groep gevonden die "
                "dubbel voorkomt. Tel ze op of streep ze tegen elkaar weg om "
                "dubbelingen te voorkomen."
            )
        if exc.check.name == "invalid_values_preferences":
            invalid_values = ", ".join(
                set(
                    exc.failure_cases.loc[
                        lambda df: df["column"] == "Waarde", "failure_case"
                    ]
                )
            )
            return f"Onbekende leerling of groep in categorie: {invalid_values}"
    return (
        f"Er is iets onverwachts misgegaan bij het lezen van {exc.filetype}. "
        "Controleer het bestand goed en of je het meest recente template hebt gebruikt. "
        "Als het probleem blijft bestaan, laat de maker dit onderzoeken."
    )
