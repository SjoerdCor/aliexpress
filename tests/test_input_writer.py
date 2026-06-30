# pylint: disable=protected-access
"""Tests for the input_writer module."""

import warnings

import pandas as pd

from aliexpress.data import input_writer
from aliexpress.data.datareader import VOORKEUREN_SCHEMA


def test_dropdown_columns_geslacht():
    """Jongen/meisje zit in kolom C (positie 1 in schema; A=Leerling, B=MinimaleTevredenheid)."""
    assert input_writer._dropdown_columns("geslacht") == "C"


def test_dropdown_columns_groepen():
    """De twee Niet-in Waarde-kolommen landen op Q en R."""
    assert input_writer._dropdown_columns("groepen") == "QR"


def test_dropdown_columns_leerlingen_en_groepen():
    """De vijf Graag-met- en één Liever-niet-met-Waarde-kolommen landen op E G I K M O."""
    assert input_writer._dropdown_columns("leerlingen_en_groepen") == "EGIKMO"


def test_voorkeuren_template_columns_match_schema():
    """Kolomstructuur in input_templates/voorkeuren.xlsx moet overeenkomen met VOORKEUREN_SCHEMA.

    Als deze test faalt: het template en het schema zijn uit sync. Pas het template aan
    zodat het overeenkomt met het schema (of andersom), maar bewerk nooit alleen de ene kant.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        raw = pd.read_excel(
            "input_templates/voorkeuren.xlsx", header=None, nrows=3, index_col=0
        )

    with pd.option_context("future.no_silent_downcasting", True):
        raw.iloc[0] = raw.iloc[0].ffill().infer_objects(copy=False)
        raw.iloc[1] = raw.iloc[1].ffill().infer_objects(copy=False)
    raw.iloc[2] = raw.iloc[2].replace(
        {"Naam (leerling of stamgroep)": "Waarde", "Stamgroep": "Waarde"}
    )

    template_columns = pd.MultiIndex.from_arrays(
        [raw.iloc[0], raw.iloc[1], raw.iloc[2]], names=["TypeWens", "Nr", "TypeWaarde"]
    )
    schema_columns = pd.MultiIndex.from_tuples(
        VOORKEUREN_SCHEMA.columns.keys(), names=["TypeWens", "Nr", "TypeWaarde"]
    )
    pd.testing.assert_index_equal(template_columns, schema_columns)
