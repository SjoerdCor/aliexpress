"""Functions to select students and groups to be selected for distribution"""

import pandas as pd

from ..errors import DuplicateNameError
from .datareader import display_name, matching_key

CANDIDATE_FIELDS = [
    "key",
    "roepnaam",
    "achternaam",
    "groepsnaam",
    "geslacht",
    "jaargroep",
]


def get_candidates(df: pd.DataFrame, jaargroep: int) -> list:
    """Return list of candidates for the given jaargroep."""
    df_current = df[df["jaargroep"] == jaargroep]
    if df_current.empty:
        return []
    relevant_columns = CANDIDATE_FIELDS

    return (
        df_current.reset_index()
        .sort_values(["groepsnaam", "roepnaam", "achternaam"])[relevant_columns]
        .to_dict(orient="records")
    )


def get_groups_from(df: pd.DataFrame, jaargroep: int) -> list:
    """Return unique group names in the current jaargroep plus 'Anders'."""
    df_current = df[df["jaargroep"] == jaargroep]
    return df_current["groepsnaam"].unique().tolist() + ["Anders"]


def get_groups_to(df: pd.DataFrame, jaargroep: int) -> dict:
    """Return dictionary of groups for the next jaargroep with blijft_in_groep flag."""
    next_jaargroep = jaargroep + 1
    groupnames_to = (
        df.loc[df["jaargroep"] == next_jaargroep, "groepsnaam"].unique().tolist()
    )

    df_next = df[df["groepsnaam"].isin(groupnames_to)].copy()
    if df_next.empty:
        return {}

    max_jaargroep_per_group = df_next.groupby("groepsnaam")["jaargroep"].transform(
        "max"
    )
    df_next["blijft_in_groep"] = df_next["jaargroep"] < max_jaargroep_per_group

    return (
        df_next.sort_values(["groepsnaam", "jaargroep", "geslacht"])
        .groupby("groepsnaam")
        .apply(
            lambda g: g[
                ["roepnaam", "achternaam", "geslacht", "jaargroep", "blijft_in_groep"]
            ].to_dict(orient="records")
        )
        .to_dict()
    )


def handle_edexml_upload(df: pd.DataFrame, jaargroep: int):
    """Process uploaded EDEXML and render candidates + groups."""
    candidates = get_candidates(df, jaargroep)
    groups_from = get_groups_from(df, jaargroep)
    groups_to = get_groups_to(df, jaargroep)
    return candidates, groups_from, groups_to


def get_candidates_herindelen(df: pd.DataFrame, group_names: list[str]) -> list:
    """Return candidates for herindelen: all students in the selected groups."""
    df_selected = df[df["groepsnaam"].isin(group_names)]
    if df_selected.empty:
        return []
    return (
        df_selected.reset_index()
        .sort_values(["groepsnaam", "roepnaam", "achternaam"])[CANDIDATE_FIELDS]
        .to_dict(orient="records")
    )


def handle_edexml_upload_herindelen(df: pd.DataFrame, group_names: list[str]):
    """Process uploaded EDEXML for herindelen: redistribute students within selected groups.

    All students in the selected groups become candidates; the destination groups are the
    same groups with zero occupancy (no fixed students, so the solver controls placement
    entirely).
    """
    candidates = get_candidates_herindelen(df, group_names)
    groups_from = list(group_names) + ["Anders"]
    groups_to = {g: [] for g in group_names}
    return candidates, groups_from, groups_to


def students_df(students: pd.DataFrame) -> pd.DataFrame:
    """Assign a unique display name per leerling and sort for the prefilled Excel template."""
    return students.assign(uniekenaam=create_unique_name).sort_values(
        ["groepsnaam", "uniekenaam"]
    )


def students_df_from_records(students: list[dict]) -> pd.DataFrame:
    """Build the prefilled-Excel DataFrame from a list of student dicts (roster participants)."""
    return students_df(pd.DataFrame(students))


def unique_display_names(students: list[dict]) -> dict[str, str]:
    """Map each participant's full display name (``roepnaam achternaam``) to a short unique
    name (``roepnaam`` plus as few surname letters as needed to stay unique).

    Display-only: the full name remains the stored, matched identity (see ADR 0007); this
    just gives the overview a shorter, still-unambiguous label.
    """
    uniek = create_unique_name(pd.DataFrame(students).copy())
    return {
        f"{s['roepnaam']} {s['achternaam']}": uniek.iloc[i]
        for i, s in enumerate(students)
    }


def create_unique_name(df: pd.DataFrame) -> pd.Series:
    """Find a unique display name per leerling, from roepnaam and achternaam.

    Names are kept as entered (only edge-stripped, see ``display_name``); uniqueness is
    decided on the ``matching_key`` so case/space variants count as the same person.
    Because display-equal names are necessarily key-equal, unique keys guarantee unique
    display names.
    """
    df["roepnaam"] = df["roepnaam"].apply(display_name)
    df["achternaam"] = df["achternaam"].apply(display_name)

    duplicated = (
        df["roepnaam"].apply(matching_key) + df["achternaam"].apply(matching_key)
    ).duplicated()
    if duplicated.any():
        raise DuplicateNameError(
            "duplicate_names",
            {
                "duplicate_names": df.loc[
                    duplicated, ["roepnaam", "achternaam"]
                ].to_dict(orient="records")
            },
            f"Can not create unique names for {df[duplicated]}",
        )
    unique_names = df["roepnaam"] + " "

    n_letters_added = 0
    while unique_names.apply(matching_key).duplicated().any():
        for ix in unique_names[
            unique_names.apply(matching_key).duplicated(keep=False)
        ].index:
            unique_names[ix] += df.loc[ix, "achternaam"][n_letters_added]
        n_letters_added += 1
    return unique_names.apply(display_name)
