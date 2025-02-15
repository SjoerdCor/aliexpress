"""Read and transform the input sheet to a workable DataFrame"""

import warnings
import pandas as pd


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

    def handle_liever_niet(self) -> None:
        """Adjusts 'Liever niet met' category by negating weight and renaming."""
        df = self.df.reset_index()
        mask = df["TypeWens"] == "Liever niet met"
        df.loc[mask, "Gewicht"] = -df["Gewicht"]
        df.loc[mask, "TypeWens"] = "Graag met"

        df["Nr"] = df.groupby(["Leerling", "TypeWens"]).cumcount() + 1
        self.df = df.set_index(["Leerling", "TypeWens", "Nr"])

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
        self.handle_liever_niet()
        return self.df

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
