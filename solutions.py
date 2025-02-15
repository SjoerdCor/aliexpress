import pandas as pd
import pulp
from openpyxl.utils import get_column_letter
from openpyxl.styles import numbers

from problemsolver import get_satisfaction_integral


class SolutionAnalyzer:
    def __init__(
        self,
        prob: pulp.LpProblem,
        preferences: pd.DataFrame,
        old_preferences: pd.DataFrame,
        input_sheet: pd.DataFrame,
    ):
        self.prob = prob
        self.preferences = preferences
        self.old_preferences = old_preferences
        self.input_sheet = input_sheet
        self.groepsindeling = self._get_outcome()
        self.ll_performance = self._calculate_performance_per_leerling()
        self.solution_performance = self._calculate_solution_performance()
        self.satisfied_preferences_original_index = (
            self.determine_satisfied_wishes_leerlingindex()
        )

    def _get_outcome(self) -> pd.DataFrame:
        """
        Restructure the Problem Variables in a nice DataFrame

        Parameters
        ----------
        vars: list of pulp.LpVariables
            The result of prob.variables()
        """
        chosen_groups = [
            v.name
            for v in self.prob.variables()
            if v.value() == 1 and v.name.startswith("group")
        ]
        df = pd.DataFrame(chosen_groups)
        df[["Naam", "Group"]] = df[0].str.extract(r"group_\('(.*)',_'(.*)'\)")
        return df.drop(columns=[0])

    def display_groepsindeling(self):
        """
        Transform DataFrame so that leerlingen are grouped by the group in which they are placed
        """
        df = (
            self.groepsindeling.assign(
                nr=lambda df: df.groupby("Group").cumcount().add(1)
            )
            .set_index(["Group", "nr"])["Naam"]
            .unstack("Group", fill_value="")
        )
        return df

    @staticmethod
    def _probvars_to_series(prob, name: str, not_in_name: str) -> pd.Series:
        """
        Extract (accounted) preferences from problem to a series

        Will extract leerling name and preference number as index, and whether accounted for as value
        Parameters
        ----------
        name: str
            The beginning of the variable name, will also be the name of the Series
        not_in_name: str

        """
        constraints = {
            v.name: v.value()
            for v in prob.variables()
            if v.name.startswith(name) and not not_in_name in v.name
        }
        series = pd.Series(constraints, name=name)
        ix = (
            series.index.to_series()
            .str.extract(rf"{name}_\('(?P<ll>.*)',_(?P<Nr>.*)\)")
            .set_index(["ll", "Nr"])
            .index
        )

        series.index = ix
        return series

    def _calculate_satisfied_constraints(self) -> pd.DataFrame:
        """
        Calculate which constraints and for whom are accommodated

        Parameters
        ----------
        prob: pulp.LpProblem
            The result of prob.variables()

        Returns
        -------
            pd.DataFrame with Satisfied and WeightedSatisfied preferences
        """
        satisfied = self._probvars_to_series(
            self.prob, "Satisfied", "per_group"
        ).astype("boolean")
        weighted_satisfied = self._probvars_to_series(
            self.prob, "WeightedSatisfied", "per_group"
        )
        df = pd.concat([satisfied, weighted_satisfied], axis="columns")
        df.index = df.index.set_levels(pd.to_numeric(df.index.levels[1]), level=1)
        return df

    def _calculate_performance_per_leerling(self):
        """
        Calculate basic performance metrics per leerling

        Performance is better when more preferences are more accommodated

        Parameters
        ----------
        satisfied_constraints: pd.DataFrame
            The output of calculate_satisfied_constraints
        """
        df = (
            self._calculate_satisfied_constraints()
            .groupby("ll")
            .agg(
                NrPreferences=("Satisfied", "count"),
                AccountedPreferences=("Satisfied", "sum"),
                PctAccounted=("Satisfied", "mean"),
                AccountedWeightedPreferences=("WeightedSatisfied", "sum"),
            )
            .assign(
                NrWeightedPreferences=self.preferences.xs("Graag met", level="TypeWens")
                .loc[lambda df: df["Gewicht"].gt(0)]
                .groupby("Leerling")["Gewicht"]
                .sum(),
                PctWeightedAccounted=lambda df: df["AccountedWeightedPreferences"]
                / df["NrWeightedPreferences"],
                PossibleSatisfaction=lambda df: df["NrWeightedPreferences"].map(
                    lambda x: get_satisfaction_integral(0, x)
                ),
                ActualSatisfaction=lambda df: df["AccountedWeightedPreferences"].map(
                    lambda x: get_satisfaction_integral(0, x)
                ),
                RelativeSatisfaction=lambda df: df["ActualSatisfaction"]
                / df["PossibleSatisfaction"],
            )
        )
        return df

    def display_leerling_performance(self):
        cols = {
            "RelativeSatisfaction": "Tevredenheid",
            "AccountedPreferences": "Aantal gehonoreerde wensen",
            "NrPreferences": "Aantal wensen",
        }
        return (
            self.ll_performance.rename_axis("Leerling")
            .loc[:, list(cols.keys())]
            .rename(columns=cols)
            .style.background_gradient(
                "RdYlGn", vmin=0, vmax=1, subset=["Tevredenheid"]
            )
            .format({"Tevredenheid": "{:.2%}"})
        )

    def _calculate_solution_performance(self):
        """
        Calculate the performance of the general model

        Parameters
        ----------
        ll_performance: pd.DataFrame
            The output of calculate_performance_per_leerling
        """
        cols = [
            "NrPreferences",
            "AccountedPreferences",
            "NrWeightedPreferences",
            "AccountedWeightedPreferences",
            "PossibleSatisfaction",
            "ActualSatisfaction",
        ]
        solution_performance = (
            self.ll_performance[cols]
            .sum()
            .to_frame()
            .transpose()
            .assign(
                PctAccountedPreferences=lambda df: df["AccountedPreferences"]
                / df["NrPreferences"],
                PctAccountedWeightedPreferences=lambda df: df[
                    "AccountedWeightedPreferences"
                ]
                / df["NrWeightedPreferences"],
                RelativeSatisfaction=lambda df: df["ActualSatisfaction"]
                / df["PossibleSatisfaction"],
            )
        ).to_dict("records")[0]
        return solution_performance

    def determine_satisfied_wishes_leerlingindex(self) -> pd.DataFrame:
        """Get the satisfied wishes, but change the index so that it matches the input

        This is useful so that we can match the original file whether a wish is satisfied
        And is used in coloring the output

        Parameters
        ----------
        voorkeuren : pd.DataFrame
            The DataFrame that has the voorkeuren with altered index

        voorkeuren_old : _type_
            The DataFrame that has the wishes and is used as input value

        satisfied_constraints : pd.DataFrame
            Whether the a constraint (with index belonging to `voorkeuren`) is satisfied

        Returns
        -------
        df
        """

        mapping = {}
        for i in range(len(self.preferences)):
            if self.preferences.index[i][1] == "Graag met":
                mapping[self.preferences.reset_index("TypeWens").index[i]] = (
                    self.old_preferences.index[i]
                )

        df = pd.DataFrame(mapping, index=["Leerling", "TypeWens", "Nr"]).transpose()
        df.index.names = ["ll", "Nr"]

        satisfied_constraints = self._calculate_satisfied_constraints()
        df = (
            df.join(satisfied_constraints)
            .reset_index(drop=True)
            .set_index(["Leerling", "TypeWens", "Nr"])
        )
        return df

    def _display_satisfied_preferences(self):
        """
        Determine the background property based on whether a wish is satisfied
        """
        df_style = pd.DataFrame(
            "background-color: white",
            index=self.input_sheet.index,
            columns=self.input_sheet.columns,
        )

        for idx in df_style.index:
            for col in df_style.columns:
                original_idx = (idx, col[0], col[1])
                try:
                    if self.satisfied_preferences_original_index.loc[
                        original_idx, "Satisfied"
                    ]:
                        df_style.loc[idx, col] = "background-color: green"
                    else:
                        df_style.loc[idx, col] = "background-color: red"
                except KeyError:  # Not a preference -> leave background as is
                    continue

        return df_style

    def display_satisfied_preferences(self):
        return self.input_sheet.style.apply(
            self._display_satisfied_preferences, axis=None
        )

    @staticmethod
    def _autoscale_column_width(sheet):
        for column in sheet.columns:
            max_length = 0
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(cell.value)
                except TypeError:
                    pass
            adjusted_width = (max_length + 2) * 1.2
            sheet.column_dimensions[get_column_letter(column[0].column)].width = (
                adjusted_width
            )

    def to_excel(self):
        # https://github.com/PyCQA/pylint/issues/3060 pylint: disable=abstract-class-instantiated
        with pd.ExcelWriter("groepsindeling2.xlsx", engine="openpyxl") as writer:

            self.display_groepsindeling().to_excel(
                writer, "Groepsindeling", index=False
            )

            self.display_leerling_performance().to_excel(writer, "Leerlingtevredenheid")
            sheet = writer.book.worksheets[1]
            for cell in sheet["B"]:
                cell.number_format = numbers.FORMAT_PERCENTAGE
            self._autoscale_column_width(sheet)

            self.display_satisfied_preferences().to_excel(writer, "VervuldeWensen")
