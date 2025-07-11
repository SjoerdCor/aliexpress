""" "Create test data for the app"""

import math
import os
import random
import string

import pandas as pd
import numpy as np

random.seed(42)

FOLDER = "testdata"


def generate_groups(n_groups=4) -> pd.DataFrame:
    """Create the groups-DataFrame"""
    assert 1 < n_groups <= 10
    sample_group_names = [
        "Beren",
        "Otters",
        "Panda's",
        "Flamingo's",
        "Alpaca's",
        "Pinguins",
        "Vossen",
        "Zebras",
        "Giraffen",
        "Stokstaartjes",
    ]
    selected_names = sample_group_names[:n_groups]

    rows = {}
    for gr in selected_names:
        n_leerlingen = random.randint(12, 18)
        pct_jongens = 0.3 + 0.4 * random.random()
        n_jongens = int(pct_jongens * n_leerlingen)
        rows[gr] = {"Jongens": n_jongens, "Meisjes": n_leerlingen - n_jongens}

    return pd.DataFrame.from_dict(rows, orient="index").reset_index(names="Groepen")


class PreferenceExcelGenerator:
    """Class to generate the excel for the preferences

    Parameters
    ----------
    groups_to : list
        Group names to which the students can be sent
    n_groups_from : int, optional
        Number of groups the students are coming from, by default 4
    """

    df_header = pd.DataFrame(
        [
            (
                "Leerling",
                "MinimaleTevredenheid",
                "Jongen/meisje",
                "Stamgroep",
                "Graag met",
                "Graag met",
                "Graag met",
                "Graag met",
                "Graag met",
                "Graag met",
                "Graag met",
                "Graag met",
                "Graag met",
                "Graag met",
                "Liever niet met",
                "Liever niet met",
                "Niet in",
                "Niet in",
            ),
            (
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                1,
                1,
                2,
                2,
                3,
                3,
                4,
                4,
                5,
                5,
                1,
                1,
                1,
                2,
            ),
            (
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                "Waarde",
                "Gewicht",
                "Waarde",
                "Gewicht",
                "Waarde",
                "Gewicht",
                "Waarde",
                "Gewicht",
                "Waarde",
                "Gewicht",
                "Waarde",
                "Gewicht",
                "Waarde",
                "Waarde",
            ),
        ]
    )
    possible_students = [
        ("Anna", "Meisje"),
        ("Bram", "Jongen"),
        ("Claire", "Meisje"),
        ("Daan", "Jongen"),
        ("Eva", "Meisje"),
        ("Finn", "Jongen"),
        ("Gina", "Meisje"),
        ("Hugo", "Jongen"),
        ("Iris", "Meisje"),
        ("Jesse", "Jongen"),
        ("Kiki", "Meisje"),
        ("Lars", "Jongen"),
        ("Mila", "Meisje"),
        ("Noah", "Jongen"),
        ("Olivia", "Meisje"),
        ("Pim", "Jongen"),
        ("Quinn", "Jongen"),
        ("Rosa", "Meisje"),
        ("Sam", "Jongen"),
        ("Tess", "Meisje"),
        ("Umut", "Jongen"),
        ("Vera", "Meisje"),
        ("Wout", "Jongen"),
        ("Xena", "Meisje"),
        ("Yara", "Meisje"),
        ("Zane", "Jongen"),
        ("Lieke", "Meisje"),
        ("Nina", "Meisje"),
        ("Oscar", "Jongen"),
        ("Paul", "Jongen"),
        ("Rik", "Jongen"),
        ("Sofie", "Meisje"),
        ("Tom", "Jongen"),
        ("Una", "Meisje"),
        ("Valerie", "Meisje"),
        ("Wes", "Jongen"),
        ("Xavi", "Jongen"),
        ("Yentl", "Meisje"),
        ("Zion", "Jongen"),
    ]

    def __init__(self, groups_to: list, n_groups_from=4):
        self.groups_to = groups_to
        self.groups_from = list(string.ascii_uppercase)[:n_groups_from]

    @staticmethod
    def generate_wishes(student: str, options: list, max_num_wishes=5) -> list[tuple]:
        """Generate wishes for one student with variable number of students"""
        my_options = options[:]
        my_options.remove(student)
        n_wishes_student = random.randint(0, max_num_wishes)
        selected = random.sample(my_options, k=n_wishes_student)
        wishes = [(wish, random.randint(1, 3)) for wish in selected]
        return wishes

    @staticmethod
    def generate_minimale_tevredenheid() -> float:
        """Generate either np.nan or a minimal satisfaction [0.2, 0.8]"""
        p_has_minimale_tevredenheid = 0.2
        minimale_tevredenheid_min = 0.2
        minimale_tevredenheid_max = 0.8
        if random.random() >= p_has_minimale_tevredenheid:
            return np.nan
        return random.sample(
            list(np.arange(minimale_tevredenheid_min, minimale_tevredenheid_max, 0.1)),
            k=1,
        )[0]

    @staticmethod
    def generate_not_with(
        student: str, total: list, wishes: list[tuple]
    ) -> tuple[str, int]:
        """Generate a single wish that is not already in wishes for not with"""
        if random.random() < 0.5:
            return ("", "")
        already_wished = [w[0] for w in wishes]
        my_options = [x for x in total if x not in already_wished and x != student]
        person = random.choice(my_options)
        return (person, random.randint(1, 3))

    @staticmethod
    def generate_not_in(
        group_names: list[str], wishes: list[tuple], not_with: tuple
    ) -> list[str, str]:
        """Generate 0, 1 or 2 groups the student can not be in"""
        already_wished = [w[0] for w in wishes + [not_with]]
        possible_groups = [gr for gr in group_names if gr not in already_wished]
        # -1 because there should always be at least one group to place the student in
        max_n_not_in_possible = max(min(2, len(possible_groups) - 1), 0)
        n_not_in = random.randint(0, max_n_not_in_possible)
        not_in = random.sample(possible_groups, n_not_in)

        return not_in + [""] * (2 - n_not_in)

    def generate(
        self, num_students=35, fname=os.path.join(FOLDER, "voorkeuren_generated.xlsx")
    ):
        """Generate and optionally write preference excel for num_students"""
        assert 1 <= num_students <= 40, "Number of students must be between 1 and 40"

        selected_students = self.possible_students[:num_students]

        total = [name for name, _ in selected_students] + self.groups_to

        rows = []
        for student in selected_students:
            name, gender = student
            minimale_tevredenheid = self.generate_minimale_tevredenheid()
            stamgroep = random.choice(self.groups_from)
            wishes = self.generate_wishes(name, total)
            not_with = self.generate_not_with(name, total, wishes)
            not_in = self.generate_not_in(self.groups_to, wishes, not_with)

            row = [name, minimale_tevredenheid, gender, stamgroep]
            for i in range(5):
                try:
                    row.extend(wishes[i])
                except IndexError:
                    row.extend(["", ""])
            row.extend(not_with)
            row.extend(not_in)
            rows.append(row)

        df = pd.concat([self.df_header, pd.DataFrame(rows)])
        if fname:
            df.to_excel(fname, index=False, header=False)
        return df


def generate_niet_samen(leerlingen: list, n_groups=4, n_rules=5) -> pd.DataFrame:
    """Generate the not_together excel file"""
    rules = []
    for _ in range(n_rules):
        n_children = random.randint(2, min(12, len(leerlingen)))
        group = random.sample(leerlingen, k=n_children)
        rule = [math.ceil(n_children / n_groups), *group] + [np.nan] * (12 - n_children)
        rules.append(rule)
    cols = ["Max aantal samen"] + [f"Leerling {i}" for i in range(1, 13)]
    df = pd.DataFrame(rules, columns=cols)
    return df


def main(n_groups=4, n_students=35, n_rules=5):
    """Generate the three input files and write to testdata"""
    groups = generate_groups(n_groups)
    groups.to_excel(os.path.join(FOLDER, "groepen_generated.xlsx"), index=False)
    df_students_excel = PreferenceExcelGenerator(groups["Groepen"].tolist()).generate(
        n_students
    )
    leerlingen = df_students_excel[0].iloc[3:].tolist()
    df_niet_samen = generate_niet_samen(leerlingen, n_groups=n_groups, n_rules=n_rules)
    df_niet_samen.to_excel(
        os.path.join(FOLDER, "niet_samen_generated.xlsx"), index=False
    )


if __name__ == "__main__":
    main(2, 5, 1)
