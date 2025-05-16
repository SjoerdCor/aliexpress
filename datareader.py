"""Read and transform the input sheet to a workable DataFrame"""

import warnings
import pandas as pd


def toggle_negative_weights(df: pd.DataFrame, mask="Gewicht") -> pd.DataFrame:
    """Adjusts 'Liever niet met'/'Graag met' category by negating weight and renaming.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe containing preferences in long-form, with right indexes
        annd Gewicht as column

    Returns
    -------
        pd.DataFrame
            Of the same shape, but with negated Gewicht and TypeWens
    """
    df = df.reset_index()
    # TODO: Make this more readable
    if mask == "Gewicht":
        mask = df["Gewicht"] < 0
    elif mask == "Liever niet met":
        mask = df["TypeWens"] == "Liever niet met"
    df.loc[mask, "Gewicht"] = -df["Gewicht"]
    df.loc[mask, "TypeWens"] = df.loc[mask, "TypeWens"].map(
        {"Graag met": "Liever niet met", "Liever niet met": "Graag met"}
    )

    df["Nr"] = df.groupby(["Leerling", "TypeWens"]).cumcount() + 1
    df = df.set_index(["Leerling", "TypeWens", "Nr"])
    return df


class VoorkeurenProcessor:
    """Read and transform the input sheet to a workable DataFrame"""

    def __init__(self, filename: str = "voorkeuren.xlsx"):
        self.filename = filename
        self.input = self._read_voorkeuren()
        self.df = self.input.copy()

    def _read_voorkeuren(self) -> pd.DataFrame:
        """Reads and processes the voorkeuren file into a structured DataFrame."""
        df = pd.read_excel(self.filename, header=None, index_col=0).rename_axis(
            "Leerling"
        )

        df.iloc[0] = df.iloc[0].ffill()
        df.iloc[1] = df.iloc[1].ffill()
        df.iloc[2] = df.iloc[2].replace(
            {"Naam (leerling of stamgroep)": "Waarde", "Stamgroep": "Waarde"},
        )
        df.columns = pd.MultiIndex.from_arrays(
            [df.iloc[0], df.iloc[1], df.iloc[2]], names=["TypeWens", "Nr", "TypeWaarde"]
        )

        df = df.iloc[3:]

        if df.index.duplicated().any():
            raise RuntimeError("Non-unique leerlingen detected in input data.")
        return df

    def restructure(self) -> None:
        """Restructures voorkeuren DataFrame from wide to long format with default values."""
        self.df = self.df.stack(["TypeWens", "Nr"]).fillna({"Gewicht": 1})

    def validate(self, all_to_groups=None) -> None:
        """Validates voorkeuren DataFrame structure and values."""
        if self.df.index.names != ["Leerling", "TypeWens", "Nr"]:
            raise ValueError(
                "Invalid index names. Expected ['Leerling', 'TypeWens', 'Nr']."
            )

        if list(self.df.columns) != ["Gewicht", "Waarde"]:
            raise ValueError("Invalid columns! Expected ['Gewicht', 'Waarde'].")

        if (self.df["Gewicht"] <= 0).any():
            raise ValueError("All 'Gewicht' values must be positive.")

        all_leerlingen = self.input.index.get_level_values("Leerling").unique().tolist()
        accepted_values = {
            "Niet in": all_to_groups or [],
            "Graag met": all_leerlingen + (all_to_groups or []),
            "Liever niet met": all_leerlingen + (all_to_groups or []),
        }

        for wishtype, allowed_values in accepted_values.items():
            try:
                # The default value in the lambda prevents pylint cell-var-from-loop
                invalid_values = self.df.xs(wishtype, level="TypeWens")["Waarde"].loc[
                    lambda x, allowed_values=allowed_values: ~x.isin(allowed_values)
                ]
                if not invalid_values.empty:
                    raise ValueError(
                        f"Invalid values in '{wishtype}':\n{invalid_values}"
                    )
            except KeyError:
                warnings.warn(f"No entries found for wish type '{wishtype}'")

    def process(self, all_to_groups: list) -> pd.DataFrame:
        """Runs the full processing pipeline.

        Parameters
        ----------
        all_to_groups : list
            The groups to which students can be sent. This is necessary to validate the
            input

        """

        self.restructure()
        self.validate(all_to_groups)
        self.df = toggle_negative_weights(self.df, mask="Liever niet met")
        return self.df

    def get_students_meta_info(self) -> dict:
        """Get all meta information about each student

        This can be useful to balance new groups

        Returns
        -------
        dict
            Per student all known information
        """
        meta_info_cols = ["Jongen/meisje", "Stamgroep"]
        return self.input[meta_info_cols].droplevel([1, 2], "columns").to_dict("index")

    def get_students_per_old_group(self) -> dict:
        """Get per group the current student names

        Returns
        -------
        dict
            Per group the current student names
        """
        return (
            self.input["Stamgroep"]
            .squeeze()
            .rename("Stamgroep")
            .reset_index()
            .groupby("Stamgroep")["Leerling"]
            .agg(list)
            .to_dict()
        )
