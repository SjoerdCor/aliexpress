"""Functions to select students and groups to be selected for distribution"""

import pandas as pd

from .datareader import display_name, matching_key
from .errors import DuplicateNameError


def get_candidates(df: pd.DataFrame, jaargroep: int) -> list:
    """Return list of candidates for the given jaargroep."""
    df_current = df[df["jaargroep"] == jaargroep]
    if df_current.empty:
        return []
    relevant_columns = ["key", "roepnaam", "achternaam", "groepsnaam", "geslacht"]

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


def combine_students(
    candidates: dict, selected_ids: list[int], new_students: dict
) -> pd.DataFrame:
    """Combine selected and new students into a single DataFrame"""
    if candidates:
        df_original = pd.DataFrame(candidates).set_index("key").loc[selected_ids]
    else:
        df_original = pd.DataFrame(columns=["roepnaam", "achternaam", "groepsnaam"])
    df_new = pd.DataFrame(new_students)
    return (
        pd.concat([df_original, df_new])
        .assign(uniekenaam=create_unique_name)
        .sort_values(["groepsnaam", "uniekenaam"])
    )


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
