"""Write pre-filled input templates"""

import logging
from io import BytesIO

import numpy as np
import openpyxl
import pandas as pd
from openpyxl.worksheet.datavalidation import DataValidation

from .datareader import VOORKEUREN_SCHEMA

logger = logging.getLogger(__name__)


def add_data_validations(wb):
    """Add data validations to workbook

    For jongen/meisje, for niet in (groups) and for preferences (students + group)
    """
    ws1 = wb["Sheet1"]
    val_specs = [
        (
            "Sheet2!$A:$A",
            "C",
            "Verkeerd ingevuld geslacht",
            "Het geslacht moet of 'Jongen' of 'Meisje' zijn",
        ),
        (
            "Sheet2!$B:$B",
            "QR",
            "Onbekende groep",
            "Spel de groepsnaam exact zoals deze in de lijst staat",
        ),
        (
            "Sheet2!$C:$C",
            "EGIKMO",
            "Onbekende groep of leerling",
            "Spel de naam exact zoalsdeze in de lijst staat",
        ),
    ]

    for rng, cols_to_be_validated, errortitle, error in val_specs:
        dv = DataValidation(
            type="list",
            formula1=f"={rng}",
            allow_blank=True,
            showErrorMessage=True,
            errorTitle=errortitle,
            error=error,
        )
        for col in cols_to_be_validated:
            dv.add(f"{col}4:{col}1048576")
        ws1.add_data_validation(dv)


def fill_in_known_values(groups_to, groep_die_doorgaat, wb):
    """Fill the students in from the workbook, and the data to be used for validation"""
    ws1 = wb["Sheet1"]
    for i, (_, row) in enumerate(groep_die_doorgaat.iterrows(), start=4):
        ws1[f"A{i}"].value = row["uniekenaam"]
        ws1[f"C{i}"].value = row["geslacht"]
        ws1[f"D{i}"].value = row["groepsnaam"]

    logger.debug("Data ingevuld")

    all_leerlingen = groep_die_doorgaat["uniekenaam"].tolist()
    ws2 = wb["Sheet2"]
    for i, gr in enumerate(groups_to, start=1):
        ws2[f"B{i}"].value = gr
    for i, sub in enumerate(groups_to + all_leerlingen, start=1):
        ws2[f"C{i}"].value = sub


def create_prefilled_excel(groups_to: list, df_total: pd.DataFrame) -> BytesIO:
    """Fill Excel template for preferences and return as file"""

    wb = openpyxl.load_workbook("input_templates/voorkeuren.xlsx")

    fill_in_known_values(groups_to, df_total, wb)
    add_data_validations(wb)

    output = BytesIO()
    wb.save(output)
    output.seek(0)

    return output


def write_preferences_to_excel(df, fname, **kwargs):
    """Write a voorkeuren DataFrame to Excel, prepending the three-row MultiIndex header.

    Pandas cannot write a MultiIndex-with-nan column header directly, so the header is
    materialised as plain rows derived from VOORKEUREN_SCHEMA.  kwargs are forwarded to
    .to_excel().
    """
    keys = list(VOORKEUREN_SCHEMA.columns.keys())
    df_header = pd.DataFrame(
        [
            ["Leerling"] + [k[0] for k in keys],
            [np.nan] + [k[1] for k in keys],
            [np.nan] + [k[2] for k in keys],
        ]
    )
    assert df_header.shape[1] == df.shape[1]
    concatted = pd.concat(
        [
            df_header.set_axis(range(df_header.shape[1]), axis="columns"),
            df.set_axis(range(df.shape[1]), axis="columns"),
        ],
        ignore_index=True,
    )
    return concatted.to_excel(fname, index=False, header=False, **kwargs)
