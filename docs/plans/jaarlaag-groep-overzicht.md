# Plan: jaarlaag×groep-overzicht in het rapport

**Status:** concept, ter goedkeuring. Geschreven 2026-07-05 als hand-off naar een verse
implementatiesessie. Bouwt voort op branch `feature/year-balance` (herindelen + CP-SAT);
start een nieuwe branch dáárvandaan (bijv. `feature/jaarlaag-rapport`), niet vanaf master.

## Waarom

ADR-0012 (herindelen) noemt als consequentie: *"Rapport krijgt een jaarlaag×groep-overzicht
zodat de leerkracht de bereikte spreiding kan zien."* Dat ontbreekt nog. Het Klassenoverzicht
heeft per groep twee rijen — "Totaal" en "Jaarlaag" — waarbij "Jaarlaag" álle bewegende
leerlingen optelt ([solutions.py:269](../../src/aliexpress/solver/solutions.py#L269)). Bij
herindelen met drie jaarlagen is die rij gelijk aan "Totaal" (bezetting 0) en is de spreiding
per jaarlaag — juist de zorg die tot ADR-0012 leidde — onzichtbaar.

## Ontwerp

**Gekozen richting (aanbeveling): de bestaande Klassenoverzicht-tabel krijgt één rij per
jaarlaag** in plaats van de ene verzamelrij "Jaarlaag". Geen nieuwe sheet: de webpagina
(result.html rendert de tabellen generiek via `to_html`) en de Excel-export
(`group_report.to_excel`) liften automatisch mee, en de leerkracht ziet totaal + spreiding in
één tabel. Het alternatief — een aparte sheet "Jaarlaagverdeling" — vergt conditionele
plumbing (weglaten bij doorzetten) en is afgewezen als extra complexiteit.

Datastroom, van solver naar rapport:

1. **`GroupComposition`** ([results.py:16](../../src/aliexpress/solver/results.py#L16))
   vervangt `boys_year`/`girls_year` door een per-cohort telling:
   `per_year: dict[int | None, SexCounts]` met `SexCounts` als klein frozen dataclass
   `(boys: int, girls: int)`. Geen dubbele velden: het oude jaar-totaal is de som over de
   cohorten. `boys_total`/`girls_total` blijven (bezetting + nieuw). Plain Python values,
   dus serialiseerbaar zoals de docstring belooft.
2. **`results._group_composition`** groepeert de toegewezen leerlingen per
   `students[s].get("Jaarlaag")` (zelfde sleutel als `_balance_families._cohorts`).
3. **`solutions._calculate_group_report`** maakt per groep de rij `("Totaal")` plus per
   cohort een rij `f"Jaarlaag {jaar}"`; het `None`-cohort houdt het kale label `"Jaarlaag"`
   (het Excel-invoerpad zonder jaarlaag verandert dus niet van uiterlijk). Rijvolgorde
   expliciet reindexen — Totaal eerst, dan jaarlagen numeriek oplopend — en niet op de
   alfabetische `unstack`-sortering leunen (die zet "Jaarlaag 10" vóór "Jaarlaag 6").
   `display_groepsindeling` leest alleen `(group, "Totaal")` en blijft ongemoeid.

### Besloten (owner, 2026-07-05)

- **Zichtbare labelwijziging bij doorzetten via EDEXML: geaccepteerd.** `jaargroep` zit in
  `CANDIDATE_FIELDS` en reist in beide modi mee naar `StudentEntry.year_group`; bij
  doorzetten-EDEXML heeft `students_info` dus één concrete jaarlaag en wordt de rij
  "Jaarlaag" → bijv. "Jaarlaag 5". Dat is duidelijker voor de gebruiker; verhuis de
  structurele test-invarianten mee. Geen aparte conditie voor het één-cohort-geval.
- **Veldvervanging in `GroupComposition`** (`boys_year`/`girls_year` → `per_year`): alle
  gebruikers zijn intern (solutions.py, tests), dus dit gaat zonder deprecatiepad.

## Stappen (TDD, één commit per stap)

**Stap 0 — empirische verificatie (geen productiecode).** Stel per invoerpad vast wat er
werkelijk in `students_info["..."]["Jaarlaag"]` zit: (a) doorzetten-EDEXML, (b) herindelen,
(c) Excel-upload (`upload_preferences`), (d) een op de roster-pagina handmatig toegevoegde
leerling (krijgt die een `jaargroep`?). Let op: de docstring van
`preferences_form.StudentEntry` beweert "year_group is None in doorzetten mode" — dat lijkt
achterhaald sinds `jaargroep` in `CANDIDATE_FIELDS` zit; corrigeer die docstring in deze stap
als de meting dat bevestigt.
*Succescriterium:* de vier antwoorden staan in de commit-/PR-tekst; een gemengd geval
(None-cohort naast genummerde cohorten) is als testgeval in stap 2 opgenomen als het kan
voorkomen.

**Stap 1 — `SexCounts` + `per_year` in `GroupComposition`.** Nieuwe unit test (bijv.
`tests/test_results.py`): een handgemaakte `engine.Solution` met leerlingen in twee
jaarlagen → `to_solution_result` levert per groep de juiste `per_year`-telling en
totalen; plus het `None`-geval. Pas `tests/test_solutions.py` en de reconciliatie in
`tests/integration/test_pinned_optimum.py` in dezelfde commit aan op de nieuwe velden.
*Succescriterium:* nieuwe test groen; quick suite groen.

**Stap 2 — per-jaarlaag rijen in het Klassenoverzicht.** Rode test eerst: `group_report`
met een multi-jaar `GroupComposition` heeft rijen `("A", "Jaarlaag 6")` … met kloppende
`VerschilJongensMeisjes`/`Groepsgrootte`, jaarlagen numeriek geordend ná "Totaal", en het
None-cohort als kaal "Jaarlaag". *Succescriterium:* nieuwe tests groen; quick suite groen;
`display_groepsindeling`-tests ongewijzigd groen.

**Stap 3 — integratiebewijs.** In `tests/integration/test_integration_herindelen.py`: bouw
een `SolutionAnalyzer` over het bestaande multi-jaar resultaat en assert dat elke groep
rijen "Jaarlaag 6/7/8" heeft die per groep optellen tot "Totaal". Draai de volledige
integratiesuite. *Succescriterium:* `uv run pytest tests/integration` volledig groen —
inclusief de exacte doorzetten-pins (die toetsen tevredenheid, niet rijlabels; als een
structurele assert over "Jaarlaag"-rijen breekt (`test_integration_main.py:261`), hoort de
aanpassing dáár bij deze stap, met de owner-bevestiging hierboven als grond).

**Stap 4 — documentatie + smoke.** Werk de docstrings van `results.py`/`solutions.py` bij
en de README waar die de uitvoer-sheets beschrijft. Draai de app één keer echt (run-
aliexpress-skill) met een herindel-proces en bekijk de resultaatpagina.
*Succescriterium:* screenshot/observatie van de nieuwe rijen in de PR-tekst.
NB: ADR-0012 wordt op dit moment in een parallelle sessie bewerkt — de consequentie-regel
daar níét in deze branch aanpassen; alleen melden.

## Wat verandert NIET

- De solver: model, balansfamilies, strategieën, engine, tevredenheidsmetriek — niets.
  De gepinde per-leerling tevredenheidswaarden blijven exact gelijk.
- De sheets Groepsindeling, Overgangsmatrix, Leerlingtevredenheid, VervuldeVoorkeuren.
- Templates en routes: result.html rendert de tabellen generiek; er is geen webwijziging.
- Het Excel-invoerpad blijft eruitzien zoals nu (None-cohort → kaal "Jaarlaag"-label).

## Testcommando's

```bash
uv run pytest tests/ --ignore=tests/integration --ignore=tests/browser -q --no-cov  # per stap
uv run pytest tests/integration                                                     # stap 1 en 3
uv run pylint src/aliexpress app.py                                                 # vóór elke commit
```
