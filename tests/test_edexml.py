# pylint: disable=redefined-outer-name  # for fixtures
"""Tests for datareader.EdexReader"""

from io import BytesIO

import pytest

from aliexpress.data import datareader


@pytest.fixture
def reader_mini():
    """Read small xml"""
    xml_mini = b"""<?xml version="1.0" encoding="utf-8"?>
<EDEX>
  <groepen>
    <groep key="G1">
      <naam>Groep 3A</naam>
      <jaargroep>3</jaargroep>
    </groep>
  </groepen>
  <leerlingen>
    <leerling key="L1">
      <achternaam>Jansen</achternaam>
      <voornamen>Peter Jan</voornamen>
      <roepnaam>Peter</roepnaam>
      <voorletters>PJ</voorletters>
      <geboortedatum>2015-04-12</geboortedatum>
      <geslacht>1</geslacht>
      <jaargroep>3</jaargroep>
      <instroomdatum>2021-08-16</instroomdatum>
      <groep key="G1"/>
    </leerling>
    <leerling key="L2">
      <achternaam>De Vries</achternaam>
      <voornamen>Anna</voornamen>
      <roepnaam>Anna</roepnaam>
      <voorletters>A</voorletters>
      <geboortedatum>2015-11-03</geboortedatum>
      <geslacht>2</geslacht>
      <jaargroep>3</jaargroep>
      <instroomdatum>2021-08-16</instroomdatum>
      <groep key="G1"/>
    </leerling>
  </leerlingen>
</EDEX>
"""

    return datareader.EdexReader(BytesIO(xml_mini))


@pytest.fixture
def reader_combi():
    """Read file with leerlingen and groepen"""
    xml_combi = b"""<?xml version="1.0" encoding="utf-8"?>
<EDEX>
  <groepen>
    <groep key="G2">
      <naam>Groep 4B</naam>
      <jaargroep>4</jaargroep>
    </groep>
    <groep key="G3">
      <naam>Groep 5C</naam>
      <jaargroep>5</jaargroep>
    </groep>
    <groep key="G4_4">
      <naam>Groep 4/5 Combi</naam>
      <jaargroep>4</jaargroep>
    </groep>
    <groep key="G4_5">
      <naam>Groep 4/5 Combi</naam>
      <jaargroep>5</jaargroep>
    </groep>
  </groepen>
  <leerlingen>
    <leerling key="L3">
      <achternaam>Mulder</achternaam>
      <voornamen>Kees</voornamen>
      <roepnaam>Kees</roepnaam>
      <voorletters>K</voorletters>
      <geboortedatum>2014-07-20</geboortedatum>
      <geslacht>1</geslacht>
      <jaargroep>4</jaargroep>
      <instroomdatum>2020-08-17</instroomdatum>
      <groep key="G2"/>
    </leerling>
    <leerling key="L4">
      <achternaam>Bakker</achternaam>
      <voornamen>Sophie</voornamen>
      <roepnaam>Sophie</roepnaam>
      <voorletters>S</voorletters>
      <geboortedatum>2013-12-05</geboortedatum>
      <geslacht>2</geslacht>
      <jaargroep>5</jaargroep>
      <instroomdatum>2019-08-19</instroomdatum>
      <groep key="G4_5"/>
    </leerling>
  </leerlingen>
</EDEX>
"""

    return datareader.EdexReader(BytesIO(xml_combi))


@pytest.fixture
def reader_edge():
    """Read xml with ongespecificeerd or unknown geslacht"""
    xml_edge = b"""<?xml version="1.0" encoding="utf-8"?>
<EDEX>
  <groepen>
    <groep key="G5">
      <naam>Groep 6A</naam>
      <jaargroep>6</jaargroep>
    </groep>
  </groepen>
  <leerlingen>
    <leerling key="L5">
      <achternaam>van Dijk</achternaam>
      <voornamen>Sam</voornamen>
      <roepnaam>Sam</roepnaam>
      <voorletters>S</voorletters>
      <geboortedatum>2012-09-01</geboortedatum>
      <geslacht>0</geslacht>
      <jaargroep>6</jaargroep>
      <instroomdatum>2018-08-20</instroomdatum>
      <groep key="G5"/>
    </leerling>
    <leerling key="L6">
      <achternaam>Unknown</achternaam>
      <voornamen>Case</voornamen>
      <roepnaam>Case</roepnaam>
      <voorletters>C</voorletters>
      <geboortedatum>2012-01-15</geboortedatum>
      <geslacht>9</geslacht>
      <jaargroep>6</jaargroep>
      <instroomdatum>2018-08-20</instroomdatum>
      <groep key="G5"/>
    </leerling>
  </leerlingen>
</EDEX>
"""
    return datareader.EdexReader(BytesIO(xml_edge))


def test_parse_leerlingen_mini(reader_mini):
    """Test small reads leerlingen fine"""
    df = reader_mini.df_leerlingen
    assert df.shape[0] == 2
    assert "roepnaam" in df.columns
    assert df.loc["L1", "geslacht"] == "Jongen"
    assert df.loc["L2", "geslacht"] == "Meisje"
    assert df.loc["L1", "jaargroep"] == 3


def test_parse_groepen_mini(reader_mini):
    "Test groepen are parsed fine"
    df = reader_mini.df_groepen
    assert df.shape[0] == 1
    assert df.loc["G1", "jaargroep"] == 3
    assert df.loc["G1", "naam"] == "Groep 3A"


def test_get_full_df_mini(reader_mini):
    "Test combination of small file is fine"
    df = reader_mini.get_full_df()
    assert "groepsnaam" in df.columns
    assert df.loc["L1", "groepsnaam"] == "Groep 3A"
    assert "jaargroep_groep" not in df.columns


def test_combi_group_names_and_years(reader_combi):
    """Test combigroepen works fine"""
    df = reader_combi.df_groepen
    # er bestaan twee groep-keys met dezelfde 'naam' (combi)
    combi_rows = df[df["naam"] == "Groep 4/5 Combi"]
    assert set(combi_rows["jaargroep"].astype(int).tolist()) == {4, 5}


def test_leerlingen_in_combigroep(reader_combi):
    """Test leerlingen in combigroep"""
    df = reader_combi.get_full_df()
    # Sophie zit in combi-groep
    assert df.loc["L4", "groepsnaam"].startswith("Groep 4/5")


def test_edge_cases(reader_edge):
    """Test edge case for geslachten"""
    df = reader_edge.df_leerlingen
    assert df.loc["L5", "geslacht"] == "Onbekend"
    assert df.loc["L6", "geslacht"] == "Niet gespecificeerd"
