"""User-facing Dutch error messages for upload validation failures.

Pure text formatters — no Flask or logging dependencies. Called from app.py by
``_flash_upload_error`` and ``_handle_failure``.
"""

import numpy as np
import pandera as pa
from werkzeug.exceptions import RequestEntityTooLarge

from aliexpress.errors import CouldNotReadFileError, FeasibilityError, ValidationError


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


def readableerror_to_validation_message(exc: Exception) -> str:
    """Convert a validation exception to a user-friendly message"""
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
                f"{', '.join(students)}. Vul bij elke wens een naam of groep in, of haal "
                "het bijbehorende gewicht weg als er geen wens is."
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
            students_with_duplicates = ", ".join(
                set(exc.failure_cases["index"].get_level_values("Leerling"))
            )
            return (
                "In het voorkeuren-bestand is voor "
                f"{students_with_duplicates} een leerling of groep gevonden die "
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
        if exc.check.name == "isin" and exc.filetype == "niet_samen":
            unknown_students = ", ".join(exc.failure_cases["failure_case"].astype(str))
            return (
                f"In het niet-samen-bestand komt {unknown_students} voor, "
                "die niet in het voorkeurenbestand voorkomt"
            )
        if exc.check.name == "duplicated_students_not_together":
            rows = ", ".join(set(exc.failure_cases["index"].add(1).astype(str)))
            duplicated_students = ", ".join(
                exc.failure_cases.groupby("index")["failure_case"].apply(
                    lambda s: s[s.duplicated()]
                )
            )
            return (
                f"In het niet-samen-bestand wordt in de {rows}e "
                f"groep dezelfde leerling meerdere keren genoemd: {duplicated_students}"
            )
        if exc.check.name == "too_strict_not_together":
            rows = ", ".join(set(exc.failure_cases["index"].add(1).astype(str)))
            max_samen = ", ".join(
                exc.failure_cases.loc[
                    lambda df: df["column"] == "Max aantal samen", "failure_case"
                ].astype(str)
            )
            nr_students = ", ".join(
                exc.failure_cases.groupby("index").size().sub(1).astype(str)
            )

            return (
                f"In het niet-samen-bestand op de {rows}e rij is de maximale "
                f"groepsgrootte te klein: met dit aantal groepen lukt het niet om {nr_students} "
                f"leerlingen te verdelen met maximaal {max_samen} bij elkaar."
            )

    return (
        f"Er is iets onverwachts misgegaan bij het lezen van {exc.filetype}. "
        "Controleer het bestand goed en of je het meest recente template hebt gebruikt. "
        "Als het probleem blijft bestaan, laat de maker dit onderzoeken."
    )
