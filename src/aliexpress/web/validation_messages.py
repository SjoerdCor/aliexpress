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
    case = context.get("case", "fundamental")
    header = "Met deze voorkeuren lukt geen evenwichtige groepsindeling."

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


def readableerror_to_validation_message(exc: Exception) -> str:
    """Convert a validation exception to a user-friendly message"""
    if exc.code == "infeasible_preferences":
        return _format_infeasible_preferences(exc.context)
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
            "Niet-samen-regel {rule_index} bevat dezelfde leerling meerdere keren."
        ),
        "missing_max_samen_not_together": (
            "Vul het maximale aantal samen in voor regel {rule_index}."
        ),
        "too_few_students_not_together": (
            "Niet-samen-regel {rule_index} heeft minder dan 2 leerlingen. "
            "Voeg minstens 2 leerlingen toe."
        ),
        "invalid_max_samen_not_together": (
            "Niet-samen-regel {rule_index}: het maximale aantal samen moet minstens 1 zijn."
        ),
        "unknown_student_not_together": (
            "In de niet-samen-regels staan onbekende leerlingen: {unknown_students}. "
            "Controleer of de namen overeenkomen met het voorkeuren-bestand."
        ),
        "too_strict_not_together": (
            "Niet-samen-regel {rule_index}: met {n_groups} groepen is het niet mogelijk om "
            "{n_students} leerlingen te verdelen met maximaal {max_samen} bij elkaar."
        ),
        "invalid_max_samen_type_not_together": (
            "Niet-samen-regel {rule_index}: het maximale aantal samen moet een geheel getal zijn."
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
        # Used by the roster step ("Wie gaat mee") when validating hand-added students.
        "incomplete_new_student": (
            "Maak elke nieuwe leerling af: vul voornaam, achternaam én geslacht in."
        ),
        "duplicate_new_student": (
            'Er bestaat al een leerling "{naam}". Geef een onderscheidende naam.'
        ),
        "missing_jaargroep_new_student": (
            "Geef bij elke nieuwe leerling ook de jaargroep aan."
        ),
        "duplicate_group_names": (
            "Groepsnamen moeten uniek zijn. Dubbel: {duplicates}."
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
