"""Write pre-filled input templates"""

import logging
from io import BytesIO

import openpyxl
import pandas as pd
from openpyxl.worksheet.datavalidation import DataValidation

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
