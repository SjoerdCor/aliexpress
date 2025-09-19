"""Write pre-filled input templates"""

import logging
import zipfile
from io import BytesIO

import openpyxl
from openpyxl.worksheet.datavalidation import DataValidation

logger = logging.getLogger(__name__)


def add_data_validations_not_together(wb, df):
    """Add data validations for students to not_together"""
    ws2 = wb["Sheet2"]
    for i, ll in enumerate(df["uniekenaam"].unique().tolist(), start=1):
        ws2[f"A{i}"].value = ll

    ws1 = wb["Sheet1"]

    dv = DataValidation(
        type="list",
        formula1="=Sheet2!$A:$A",
        allow_blank=True,
        showErrorMessage=True,
    )
    for col in "BCDEFGHIJKLM":
        dv.add(f"{col}2:{col}1048576")
    ws1.add_data_validation(dv)


def add_data_validations(wb):
    """Add data validations to workbook

    For jongen/meisje, for niet in (groups) and for preferences (students + group)
    """
    ws1 = wb["Sheet1"]
    val_specs = [
        ("Sheet2!$A:$A", "C"),  # jongen/meisje
        ("Sheet2!$B:$B", "QR"),  # niet in: only groups
        ("Sheet2!$C:$C", "EGIKMO"),  # (negative) preferences: groups + students
    ]

    for rng, cols_to_be_validated in val_specs:
        dv = DataValidation(
            type="list", formula1=f"={rng}", allow_blank=True, showErrorMessage=True
        )
        for col in cols_to_be_validated:
            dv.add(f"{col}4:{col}1048576")
        ws1.add_data_validation(dv)


def fill_in_groups_to(groups_to, wb):
    """Fill the students in from the workbook, and the data to be used for validation"""
    ws1 = wb["Sheet1"]
    for i, (gr, values) in enumerate(groups_to.items(), start=2):
        ws1[f"A{i}"] = gr
        ws1[f"B{i}"] = values["Jongens"]
        ws1[f"C{i}"] = values["Meisjes"]

        ws1[f"A{i}"].protection = openpyxl.styles.Protection(locked=False)
        ws1[f"B{i}"].protection = openpyxl.styles.Protection(locked=False)
        ws1[f"C{i}"].protection = openpyxl.styles.Protection(locked=False)

    dv_int = DataValidation(
        type="whole",
        operator="greaterThanOrEqual",
        formula1="0",
        allow_blank=True,
        showErrorMessage=True,
        errorTitle="Alleen niet-negatieve gehele getallen zijn toegestaan.",
    )
    dv_int.add("B2:B1048576")
    dv_int.add("C2:C1048576")
    ws1.add_data_validation(dv_int)
    ws1.protection.sheet = True


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


def create_zip_with_templates(groups_to, df_total):
    """Fill in Excel templates and package them into a ZIP file"""
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:

        wb_groups = openpyxl.load_workbook("input_templates/groepen.xlsx")
        fill_in_groups_to(groups_to, wb_groups)
        _add_to_zip(zip_file, wb_groups, "groepen.xlsx")

        wb_prefs = openpyxl.load_workbook("input_templates/voorkeuren.xlsx")
        fill_in_known_values(list(groups_to.keys()), df_total, wb_prefs)
        add_data_validations(wb_prefs)
        _add_to_zip(zip_file, wb_prefs, "voorkeuren.xlsx")

        wb_not_together = openpyxl.load_workbook("input_templates/niet_samen.xlsx")
        add_data_validations_not_together(wb_not_together, df_total)
        _add_to_zip(zip_file, wb_not_together, "niet_samen.xlsx")

    zip_buffer.seek(0)
    return zip_buffer


def _add_to_zip(zip_file, workbook, filename):
    """Helper to save workbook to buffer and add to ZIP"""
    buf = BytesIO()
    workbook.save(buf)
    buf.seek(0)
    zip_file.writestr(filename, buf.read())
