"""Generates a dummy EDEX XML file with groups and students for testing purposes."""

import random
import xml.etree.ElementTree as ET
from xml.dom import minidom

import numpy as np
import pandas as pd
from faker import Faker

fake = Faker("nl_NL")
Faker.seed(42)
random.seed(42)

SAMPLE_GROUP_NAMES = [
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
    "Koala's",
    "Kangoeroes",
    "Lama's",
    "Dolfijnen",
    "Walvissen",
    "Haaien",
    "Octopussen",
    "Krabben",
    "Schildpadden",
    "Eekhoorns",
    "Mieren",
    "Vlinders",
    "Bijen",
    "Miereneters",
    "Luiaards",
    "Wasberen",
    "Elanden",
    "Herten",
    "Vleermuizen",
    "Mollen",
    "Egels",
    "Muskusratten",
    "Bevers",
    "Zwanen",
]


def split_achternaam(volledige_achternaam):
    """
    Splitst een Nederlandse achternaam in (tussenvoegsel, achternaam).
    Regels:
    - Alle woorden vóór het eerste woord dat met een hoofdletter begint → tussenvoegsel
    - Het eerste hoofdletter-woord en alles daarna → achternaam
    """
    if not volledige_achternaam:
        return (None, None)

    woorden = volledige_achternaam.split()
    tussenvoegsel = []
    achternaam = []
    gevonden_hoofdletter = False

    for w in woorden:
        if not gevonden_hoofdletter and w[0].isupper():
            gevonden_hoofdletter = True

        if gevonden_hoofdletter:
            achternaam.append(w)
        else:
            tussenvoegsel.append(w)

    return (
        " ".join(tussenvoegsel) if tussenvoegsel else None,
        " ".join(achternaam) if achternaam else None,
    )


def generate_dummy_groups(n_groups_per_jaarlaag=3):
    """Genereert een DataFrame met dummy groepen voor jaargroepen 1-8."""
    jaargroepen = [[1, 2], [3, 4, 5], [6, 7, 8]]
    rows = []
    for jaargroep in jaargroepen:
        for _ in range(n_groups_per_jaarlaag):
            code = str(random.randint(1e11, 1e12 - 1))
            naam = random.sample(SAMPLE_GROUP_NAMES, 1)[0]
            full_name = f"{'-'.join(str(i) for i in jaargroep)} {naam} (Leerkracht1 en Leerkracht2)"
            for jaar in jaargroep:
                rows.append(
                    {"key": f"{code}_lj_{jaar}", "naam": full_name, "jaargroep": jaar}
                )
    df = pd.DataFrame(rows)
    return df


def generate_dummy_leerlingen(df_groepen_dummy, n_leerlingen=250):
    """Genereert een DataFrame met dummy leerlingen."""
    groepen_keuze = np.random.choice(df_groepen_dummy["key"], size=n_leerlingen)

    data = []
    for groep in groepen_keuze:
        jaargroep = (
            df_groepen_dummy.loc[df_groepen_dummy["key"] == groep, "jaargroep"]
            .sample()
            .values[0]
        )
        achternaam = fake.last_name()
        voorvoegsel, achternaam = split_achternaam(achternaam)

        aantal_voornamen = random.randint(1, 3)
        voornamen = " ".join(fake.first_name() for _ in range(aantal_voornamen))
        roepnaam = voornamen.split()[0]
        voorletters = "".join(name[0] for name in voornamen.split())
        geslacht = np.random.choice([1, 2])
        instroomdatum = fake.date_between(start_date="-6y", end_date="-1y").strftime(
            "%Y-%m-%d"
        )
        geboortedatum = fake.date_between(start_date="-13y", end_date="-4y").strftime(
            "%Y-%m-%d"
        )
        leerlingnummer = fake.unique.random_number(digits=4, fix_len=True)
        data.append(
            {
                "key": leerlingnummer,
                "achternaam": achternaam,
                "voornamen": voornamen,
                "voorletters": voorletters,
                "roepnaam": roepnaam,
                "geboortedatum": geboortedatum,
                "geslacht": geslacht,
                "jaargroep": jaargroep,
                "groep": groep,
                "instroomdatum": instroomdatum,
                "voorvoegsel": voorvoegsel,
            }
        )

    return pd.DataFrame(data)


def add_groepen_to_xml(df_groepen_dummy, root):
    """Voegt groepen toe aan de XML-boom."""
    groepen_el = ET.SubElement(root, "groepen")
    for _, row in df_groepen_dummy.iterrows():
        groep = ET.SubElement(groepen_el, "groep", key=row["key"])
        naam = ET.SubElement(groep, "naam")
        naam.text = str(row["naam"])
        jaargroep = ET.SubElement(groep, "jaargroep")
        jaargroep.text = str(row["jaargroep"])


def add_leerlingen_to_xml(df_leerlingen_dummy, root):
    """Voegt leerlingen toe aan de XML-boom."""
    leerlingen_el = ET.SubElement(root, "leerlingen")
    for _, row in df_leerlingen_dummy.iterrows():
        leerling_el = ET.SubElement(leerlingen_el, "leerling")

        for col in df_leerlingen_dummy.columns:
            if col == "groep":
                ET.SubElement(leerling_el, "groep", key=str(row["groep"]))
            elif pd.notna(row.get(col)):  # groep should not be written twice
                ET.SubElement(leerling_el, col).text = str(row[col])


def write_xml(df_groepen_dummy, df_leerlingen_dummy, file_loc="edex_test.xml"):
    """Schrijft de XML-boom naar een bestand."""
    root = ET.Element("EDEX")

    add_groepen_to_xml(df_groepen_dummy, root)
    add_leerlingen_to_xml(df_leerlingen_dummy, root)

    xml_str = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ")
    with open(file_loc, "w", encoding="utf-8") as f:
        f.write(pretty_xml)


def main():
    """Genereert een dummy EDEX XML-bestand."""
    df_groepen_dummy = generate_dummy_groups()
    df_leerlingen_dummy = generate_dummy_leerlingen(df_groepen_dummy, n_leerlingen=250)
    write_xml(df_groepen_dummy, df_leerlingen_dummy, file_loc="edex_test.xml")


if __name__ == "__main__":
    main()
