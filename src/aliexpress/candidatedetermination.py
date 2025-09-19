"""Functions to select students and groups to be selected for distribution"""

from collections import Counter

import pandas as pd


def handle_edexml_upload(df, jaargroep):
    """Process uploaded edexml and render candidates + groups"""

    candidates = (
        df.loc[lambda df: df["jaargroep"] == jaargroep]
        .sort_values(["groepsnaam", "roepnaam", "achternaam"])
        .reset_index()
        .filter(
            ["key", "roepnaam", "achternaam", "groepsnaam", "geslacht"], axis="columns"
        )
        .to_dict(orient="records")
    )

    groups_from = (
        df.loc[lambda df: df["jaargroep"] == jaargroep, "groepsnaam"].unique().tolist()
    ) + ["Anders"]

    groupnames_to = (
        df.loc[lambda df: df["jaargroep"] == jaargroep + 1, "groepsnaam"]
        .unique()
        .tolist()
    )

    groups_to = (
        df.loc[lambda df: df["groepsnaam"].isin(groupnames_to)]
        .assign(
            blijft_in_groep=lambda df: df["jaargroep"]
            < df.groupby("groepsnaam")["jaargroep"].transform("max")
        )
        .sort_values(["groepsnaam", "jaargroep", "geslacht"])
        .groupby("groepsnaam")
        .apply(
            lambda g: g[
                ["roepnaam", "achternaam", "geslacht", "jaargroep", "blijft_in_groep"]
            ].to_dict(orient="records")
        )
        .to_dict()
    )
    return candidates, groups_from, groups_to


def handle_form_submission(
    existing_groups, new_groups, candidates, new_students, selected_ids
):
    """Process the form after candidates have been selected and groups defined"""

    groups_to = _build_groups_summary(existing_groups, new_groups)
    df_total = _combine_students(candidates, selected_ids, new_students)
    return groups_to, df_total


def _build_groups_summary(existing_groups, new_groups):
    """Build a summary of groups with counts of boys and girls"""
    groups_to = {}
    for g, lst in existing_groups.items():
        c = Counter(lst)
        groups_to[g] = {"Jongens": c.get("Jongen", 0), "Meisjes": c.get("Meisje", 0)}

    for g in new_groups:
        groups_to.setdefault(g, {"Jongens": 0, "Meisjes": 0})

    return groups_to


def _combine_students(candidates, selected_ids, new_students):
    """Combine selected and new students into a single DataFrame"""
    df_original = pd.DataFrame(candidates).set_index("key").loc[selected_ids]
    df_new = pd.DataFrame(new_students)
    return (
        pd.concat([df_original, df_new])
        .assign(uniekenaam=create_unique_name)
        .sort_values(["groepsnaam", "uniekenaam"])
    )


def create_unique_name(df: pd.DataFrame) -> pd.Series:
    """Find unique name per leerling. Needs roepnaam and achternaam"""
    unique_names = df["roepnaam"] + " "

    n_letters_added = 0
    while unique_names.duplicated().any():
        for ix in unique_names[unique_names.duplicated(keep=False)].index:
            unique_names[ix] += df.loc[ix, "achternaam"][n_letters_added]
        n_letters_added += 1
    return unique_names.str.strip()
