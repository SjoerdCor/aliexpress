# Plan: nieuwe Groepsindeling-resultaatweergave (kaarten + chips + klassenoverzicht)

**Status:** goedgekeurd; in uitvoering op branch `feature/groepsindeling-weergave` (vanaf `master`).
Stap 0 is afgerond (uitkomsten hieronder verwerkt). Geschreven als zelfstandige hand-off naar een verse
implementatiesessie.

**Leesvolgorde voor de verse sessie:** dit plan + `CLAUDE.md` + [ADR-0016](../adr/0016-groepsindeling-gestructureerde-viewmodel.md).
Het uiterlijke ontwerp staat als open-baar-in-de-browser referentie naast dit plan:
[`groepsindeling-resultaatweergave-poc.html`](./groepsindeling-resultaatweergave-poc.html) — open
dat bestand in een browser; dat ís het doelontwerp (op drie kleine deltas na, zie §Ontwerp).

## Waarom

De resultatenpagina rendert vandaag vijf pandas-DataFrames als kale HTML-tabellen
([`tasks.py:55`](../../src/aliexpress/web/tasks.py#L55) → `df.to_html()` →
[`result.html`](../../templates/result.html) `{{ html | safe }}`). De Groepsindeling en het
Klassenoverzicht verdienen een leesbaarder weergave: **groepskaarten met naamchips**, een
**klik-popover per leerling**, en een **compact klassenoverzicht met balans**. Dat kan niet uit
een `to_html`-string; de template heeft de onderliggende waarden nodig (ADR-0016).

## Ontwerp (samenvatting van de goedgekeurde POC)

Open de POC-referentie voor het beeld. In woorden:

- **Brede witte container** (de bestaande `.container` uit `base.html`), met de **wizard-stepper**
  (`current_step = 7`, "Resultaat bekijken" actief), de knop **Bekijk sociogram**, en onderaan de
  **Download**-knop en de navigatie **← Nog niet helemaal… / Ja, ik ben tevreden! →** — precies de
  elementen die de huidige `result.html` al heeft.
- **Volgorde:** eerst de **groepskaarten** (de verdeling is de hoofdmoot), dááronder het
  **klassenoverzicht**, dááronder een **inklapbare legenda** (`<details>`, open bij openen). De drie
  analyse-tabellen (Overgangsmatrix, Leerlingtevredenheid, VervuldeVoorkeuren) blijven als
  pandas-HTML in een secundair tabblok bestaan.
- **Groepskaart:**
  - **Kop:** groepsnaam + totalen ("24 leerlingen · 12 jongens · 12 meisjes", inclusief zittende
    bezetting). *(Delta 1 t.o.v. de POC-referentie, die de totalen in de kop weglaat: zet ze terug.)*
  - **Body:** één of meer **jaarlaag-secties**. Elke sectie heeft een kop
    "Jaarlaag N — M leerlingen" en daaronder **twee kolommen: Jongens | Meisjes**. Bij Doorzetten is
    er één jaarlaag-sectie; **toon die kop óók** ("Jaarlaag 5 — 16 leerlingen"). *(Delta 2 t.o.v. de
    POC-referentie, die bij Doorzetten geen jaarlaag-kop toont.)*
  - **Domeintaal:** de popover-meta gebruikt **"jaarlaag N"**, niet "jaargroep N" (de POC schrijft
    per abuis "jaargroep"; CONTEXT.md verbiedt dat woord). *(Delta 3 t.o.v. de POC-referentie.)*
  - **Kolomkop per sekse:**
    - Zonder bezetting (Herindelen): `Jongens (5)`.
    - Met bezetting (Doorzetten): `Jongens: 8 nieuw`. "nieuw" = de
      chips (bewegende leerlingen),
  - **Chip:** toon de **unieke korte naam** (roepnaam + minimaal aantal achternaam-letters om te
    differentiëren) + de oude groep als **3-letterafkorting** (bijv. `Kik`). Hergebruik de bestaande
    `candidatedetermination.unique_display_names(participants)`; die korte naam bestaat alleen in het
    webpad — bij **Excel/CLI valt de chip terug op de volledige naam** (er is dan geen losse
    roepnaam/achternaam). Geen kleur op de chip in ruststand. De **popover toont altijd de volledige
    naam**.
- **Klik-popover per chip** (zie §Interactie), met: volledige naam, volledige oude klasnaam +
  jaarlaag, een **tevredenheids-bolletje** (rood→groen schaal, geclampt op [−100%, +100%], `—`
  als de leerling geen voorkeuren heeft), de **wensen** (graag-met/liever-niet als groen ✓ /
  rood ✗), **Niet-in** (grijs, "gerespecteerd") en de **Extra zekerheid**-badge (twee niveaus, zie
  §Inhoudelijke regels).
- **Klassenoverzicht** (compacte tabel, zelfstandig leesbaar want mét de aantallen):
  - Rijen: **Totaal** + per **jaarlaag** (Totaal-rij eerst, dan jaarlagen numeriek; de `None`-jaarlaag
    krijgt het kale label "Jaarlaag" en staat direct na Totaal — hergebruik `_year_label` /
    `_year_sort_key` uit [`solutions.py`](../../src/aliexpress/solver/solutions.py)).
  - Kolommen: één per groep (elke cel: aantal + `♂n ♀m`), plus **Grootteverschil** en
    **Onbalans ♂/♀**.
  - Caption exact: *"Klassenoverzicht — aantallen per groep. Rechts de balans: grootteverschil =
    verschil tussen de grootste en de kleinste groep; onbalans ♂/♀ = het grootste verschil tussen
    jongens en meisjes binnen één groep."*

### Interactie (popover)

- Opent op **klik** (toggle), niet op hover — hover blokkeert de chips onder een open popover en
  werkt niet op touch.
- Positioneer **naast** de chip (rechts; links als er rechts geen ruimte is), niet eronder.
- Sluiten: nogmaals klikken, buiten klikken, of **Escape**. Ook bruikbaar met toetsenbord
  (chip `tabindex="0"`, Enter/spatie togglet). Zie de `<script>` in de POC-referentie voor een
  werkende, over te nemen implementatie.

### Inhoudelijke regels (domeintrouw — zie `CONTEXT.md`)

- **Tevredenheid**: `student_satisfaction` (0..1 voor puur-positief; kan negatief bij geschonden
  vermij-voorkeuren). Toon als percentage, **geclampt op [−100%, +100%]**. Een leerling **zonder
  voorkeuren** toont `—` (niet "100%"), ook al is die intern 100% tevreden.
- **Extra zekerheid** (Stap 0 vastgesteld): koppel de badge aan de bestaande vormtaal
  `.badge-zeker--partial` (lichte outline) / `.badge-zeker--full` (goud gevuld) en gebruik de
  **letterlijke bestaande formulierteksten**, **niet** de POC-placeholder "minstens zo tevreden als
  X%". Bron: `MinimaleTevredenheid` in `students_info`. Mapping (geverifieerd in
  `templates/preferences_form.html:131-136` + `form_parsers.py:129-132`):
  - `NaN` → **geen badge**.
  - `0.5` (formulierwaarde `50`) → `--partial`, tekst **"Minstens tevreden"** (badge-titel:
    *"Extra zekerheid: minstens tevreden"*).
  - `1.0` (formulierwaarde `100`) → `--full`, tekst **"Alle voorkeuren gehonoreerd"** (badge-titel:
    *"Extra zekerheid: alle voorkeuren gehonoreerd"*).
- **Niet-in** is een harde eis en dus altijd gerespecteerd; toon grijs met "gerespecteerd".

## Datastroom en waar het view-model wordt gebouwd

Alles wat nodig is, is al aanwezig in `SolutionAnalyzer` (display-space, namen zoals ingevoerd):
`self.result` (`SolutionResult`), `self.students_info`, `self.preferences`, `self.input_sheet`,
`self.satisfied_constraints`, `self.student_performance`.

- **Toewijzing → kaarten:** `result.assignment` (leerling → groep).
- **Samenstelling/balans:** `result.group_composition[group]` =
  `GroupComposition(boys_total, girls_total, per_year)` (zie
  [`results.py`](../../src/aliexpress/solver/results.py)). **Bezetting per sekse** is afleidbaar:
  `occupancy_boys = boys_total − Σ per_year.boys` (idem meisjes). **Movers per jaarlaag** =
  `per_year[year]`.
- **Grootteverschil** per rij = `max − min` van de groepsaantallen voor die rij. **Onbalans ♂/♀**
  per rij = `max over groepen van |boys − girls|` voor die rij (Totaal-rij incl. bezetting;
  jaarlaag-rijen alleen movers).
- **Wensen (graag met):** `result.satisfied[(student, Nr)]` (bool) + het doel via
  `preferences.loc[(student, "Graag met", Nr), "Waarde"]`.
- **Herkomst:** `students_info[student]["Stamgroep"]` (vol) en `[:3]` (afkorting).
- **Jaarlaag/sekse/zekerheid:** `students_info[student]` (`Jaarlaag`, `Jongen/meisje`,
  `MinimaleTevredenheid`).
- **Unieke korte naam (chip-label):** een nieuwe display-map `unique_name` (matching-key → korte naam),
  **parallel aan `student_display`** op `PreferenceData`. Gevuld in het webpad met
  `candidatedetermination.unique_display_names(participants)`; **leeg bij Excel/CLI** → dan valt de chip
  terug op de volledige naam. `_export` relabelt 'm mee (net als de andere display-maps) en geeft 'm door
  aan `SolutionAnalyzer`; de builder leest `unique_name.get(full_name, full_name)` voor het chip-label.

**Plaats van de builder (besloten):** de dataclasses komen in een nieuwe, Flask-vrije module
`src/aliexpress/solver/groepsindeling_view.py`; de **builder is een methode op `SolutionAnalyzer`**,
`groepsindeling_view() -> GroepsindelingView`, náást de bestaande display-views. Reden: `SolutionAnalyzer`
leeft al in de rapportagelaag (display-space), bezit al alle benodigde velden (`self.result`,
`self.students_info`, `self.preferences`, `self.input_sheet`) en produceert al élke andere view
(`display_groepsindeling`, `_calculate_group_report`, `display_transition_matrix`); dit is domweg de
zoveelste view op dezelfde oplossing. Zo hoeft de `students_info → DataFrame`-merge niet gedupliceerd te
worden. Het view-model zelf blijft platte dataclasses (los van Flask, `dataclasses.asdict`-baar), dus de
naad voor de Tussenstand blijft schoon.

**Naamgeving (besloten):** het view-substantief houdt de Nederlandse domeinnaam aan, consistent met de
al ingeburgerde identifiers (`self.groepsindeling`, `display_groepsindeling`, `_write_groepsindeling`,
`Stamgroep`, `Jaarlaag`): `GroepsindelingView` / `groepsindeling_view()`. De sub-dataclasses zijn Engels
en sluiten aan op de bestaande `per_year` / `_year_label` / `_year_sort_key`: dus **`YearSection`** (niet
`CohortSection` — CONTEXT.md zet "cohort" op de _Avoid_-lijst voor Jaarlaag), naast `SexColumn`,
`StudentChip`, `Preference`, `GroupCard`, `BalanceRow`.

Voorgesteld dataclass-skelet (platte waarden, `dataclasses.asdict`-baar):

```python
@dataclass(frozen=True)
class Preference:              kind: str; target: str; fulfilled: bool      # kind: "graag_met"|"liever_niet_met"
@dataclass(frozen=True)
class StudentChip:
    chip_name: str; full_name: str; origin_abbrev: str; origin_full: str   # chip_name = unique_name of val terug op full_name
    year_group: int | None; satisfaction: float | None               # None -> toon "—"
    preferences: list[Preference]; not_in: list[str]; min_satisfaction: str | None      # None|"partial"|"full"
@dataclass(frozen=True)
class SexColumn:        sex: str; new_count: int; occupancy_count: int; total_count: int; students: list[StudentChip]
@dataclass(frozen=True)
class YearSection:      year: int | None; label: str; size: int; boys: SexColumn; girls: SexColumn
@dataclass(frozen=True)
class GroupCard:        name: str; total: int; boys_total: int; girls_total: int; year_sections: list[YearSection]
@dataclass(frozen=True)
class BalanceRow:       label: str; is_total: bool; per_group: dict[str, tuple[int,int,int]]; size_diff: int; sex_imbalance: int
@dataclass(frozen=True)
class GroepsindelingView: group_order: list[str]; groups: list[GroupCard]; balance_rows: list[BalanceRow]
```

**Pijplijn:**
1. [`main.py:_export`](../../src/aliexpress/main.py#L89) bouwt `view = sa.groepsindeling_view()` en
   geeft het terug in het resultdict: `{"download":…, "dataframes":…, "groepsindeling_view": view}`.
   Uit `dataframes` verdwijnen "Groepsindeling" en "Klassenoverzicht" (die zitten nu in `view`);
   over blijven de drie analyse-tabellen.
2. [`tasks.py`](../../src/aliexpress/web/tasks.py) schrijft naast `result_tables.json` een
   `groepsindeling_view.json` (`json.dump(dataclasses.asdict(view), …)`).
3. [`results.py:result_page`](../../src/aliexpress/web/routes/results.py#L90) laadt beide bestanden
   en geeft ze aan de template.
4. `result.html` rendert het view-model via een macro
   `{% import "partials/groepsindeling.html" as gi %}{{ gi.render(view) }}`, en daaronder de drie
   analyse-tabellen (zoals nu). Nieuwe CSS achteraan `static/style.css`; de popover-JS inline in de
   partial of in `static/` (zoals de POC).

## Stappen (TDD, één commit per stap)

**Stap 0 — empirische verificatie (AFGEROND; geen productiecode).** Uitkomsten, geverifieerd tegen de
bron:
1. **Chip-naam:** in display-space bestaat géén los voornaam-veld; de volledige naam is
   `roepnaam achternaam` (webformulier, `form_parsers.py:121`) of een vrij "Leerling"-veld (Excel). De
   chip toont daarom de **unieke korte naam** uit `candidatedetermination.unique_display_names`
   (roepnaam + minimaal aantal achternaam-letters), via de nieuwe `unique_name`-map op `PreferenceData`;
   **Excel/CLI vallen terug op de volledige naam**. De popover toont altijd de volledige naam.
2. **"Geen voorkeuren" → `—`:** `satisfaction=None` precies wanneer de leerling niet voorkomt in
   `get_graag_met(preferences)` (geen positieve én geen negatieve tevredenheid-wens). Een leerling met
   alléén "Liever niet met" (of alléén "Niet in") staat daar wél/niet in: liever-niet vouwt onder
   "Graag met" met negatief gewicht → toont `student_satisfaction` (100% als niets geschonden); een
   pure "Niet in" telt als géén voorkeur → `—`.
3. **Liever-niet & Niet-in vervuld-status:** afleiden uit `assignment` + `preferences.loc[(s,"Graag
   met",Nr),"Waarde"]` (doel = groepsleutel in `group_composition` óf klasgenoot in `assignment`).
   Graag-met vervuld = doel in dezelfde groep; liever-niet vervuld = doel NIET in dezelfde groep;
   Niet-in (bron: `input_sheet` kolom `("Niet in", k, "Waarde")`) is een harde eis → altijd
   "gerespecteerd".
4. **Extra-zekerheid-niveaus:** `NaN`=geen badge, `0.5`→`--partial` ("Minstens tevreden"),
   `1.0`→`--full` ("Alle voorkeuren gehonoreerd") — zie §Inhoudelijke regels voor de letterlijke teksten.
5. **Bezetting alleen bij Doorzetten:** bevestigd — beide Herindelen-modi forceren occupancy 0
   (`candidatedetermination.py:111,136`; `wizard.py:349-358`); alleen Doorzetten heeft bezetting > 0 met
   precies één `per_year`-jaarlaag (movers). `occupancy_boys = boys_total − Σ per_year.boys`.

**Stap 1 — view-model + builder (unit).** Nieuwe module + rode test (`tests/test_groepsindeling_view.py`):
bouw uit een handgemaakte `SolutionResult` + `students_info` + `preferences` (+ optionele `unique_name`)
een `GroepsindelingView` en assert: groepskaarten met totalen; jaarlaag-secties met de juiste `size`;
`SexColumn` met `new_count`/`occupancy_count`/`total_count`; chips met korte naam (unique_name én de
volledige-naam-fallback bij ontbrekende map)/volledige naam/herkomst; `satisfaction`
geclampt en `None` bij geen voorkeuren; `wishes` met correcte `fulfilled` (incl. liever-niet/niet-in);
`zekerheid` partial/full/None; en `balance_rows` met kloppende `size_diff`/`sex_imbalance` en rijvolgorde
(Totaal, dan jaarlagen numeriek, `None`-jaarlaag als "Jaarlaag" na Totaal). Dek het Doorzetten-geval
(bezetting > 0, één jaarlaag) én het Herindelen-geval (meerdere jaarlagen, bezetting 0).
*Succescriterium:* nieuwe test groen; quick suite groen; `uv run pylint` schoon.

**Stap 2a — `unique_name`-datacontract (AF, commit 929a9f3).** Veld `unique_name: dict` op
`PreferenceData` (matching-key → korte naam, parallel aan `student_display`), round-trip in
`to_json`/`from_json` (`.get`-fallback voor oude JSON); gevuld in het webpad via
`candidatedetermination.unique_display_names(participants)` (her-gesleuteld met `matching_key`), leeg in
het Excel-pad. Pure datacontract-uitbreiding, geen consumptie.

**Stap 2b — pijplijn, puur additief (breekt niets).** Bewuste resequencing t.o.v. het oorspronkelijke
plan: het *inkorten* van `dataframes` en de integratietest-migratie verhuizen naar Stap 3, waar ze
atomair samenvallen met de nieuwe rendering; 2b voegt alleen toe.
- `main.py:_export` bouwt `view = sa.groepsindeling_view(unique_display)` waarbij
  `unique_display = {student_display[k]: short for k, short in preference_data.unique_name.items()}`
  (matching-key → volledige naam → korte naam, dus display-space), en geeft het terug:
  `{"download":…, "dataframes":… (ongewijzigd, alle vijf), "groepsindeling_view": view}`.
- `tasks.py:_write_result_files` schrijft naast `result_tables.json` een `groepsindeling_view.json`
  (`json.dump(dataclasses.asdict(view), …, ensure_ascii=False)`).
- `results.py:result_page` laadt `groepsindeling_view.json` **als het bestaat** (optioneel; de macro komt
  pas in Stap 3) en geeft het aan de template.
- **Tests:** `tests/test_wizard.py` — de `_write_result_files`-mocks een `groepsindeling_view` meegeven en
  asserten dat het JSON-bestand wordt geschreven; `result_page` moet een ontbrekend view-JSON tolereren.
  `tests/browser/test_distribution_browser.py` — assert dat `groepsindeling_view.json` bestaat.
*Succescriterium:* quick + `uv run pytest tests/integration` + `tests/browser` groen; het view-JSON
bestaat na een run. **De integratietests blijven ongewijzigd** (dataframes nog vijf).

**Stap 3 — template-macro + CSS + JS + het inkorten.** Nieuwe partial
`templates/partials/groepsindeling.html` met de macro; `result.html` rendert het view-model (kaarten +
klassenoverzicht + legenda) en houdt de drie analyse-tabellen. **Nu pas** verdwijnen "Groepsindeling" en
"Klassenoverzicht" uit `dataframes`/`result_tables.json` (ze zitten in het view-model). CSS achter in
`static/style.css` (hergebruik de bestaande tokens/kleuren; geen nieuwe donkere variant). Popover-JS zoals
in de POC (klik-toggle, plaatsing rechts/links, sluiten via klik-buiten/Escape).
- **Test-migratie (nu de dataframes inkorten):** in `tests/integration/test_integration_main.py`
  `_EXPECTED_KEYS` terug naar de drie analyse-tabellen; de balans/structuur-invarianten uit
  `_assert_consistency` + de per-test balans-asserts herformuleren op `result["groepsindeling_view"]`:
  `totaal`-metriek = de `is_total`-`BalanceRow` (`sex_imbalance` ⇒ `VerschilJongensMeisjes.max()`,
  `size_diff` ⇒ `Groepsgrootte.max()−min()`), jaarlaag-metriek = max over de jaarlaag-`BalanceRow`s; groep-
  en jaarlaagstructuur uit `view.groups`; `jaar Groepsgrootte.sum() == n_students` = som van de
  jaarlaag-rijen. **Behoud exact dezelfde drempels en de gepinde tevredenheid** (die blijft in
  `Leerlingtevredenheid`). `test_distribute_students_from_json_matches_xlsx` ook de view vergelijken.
  `tests/test_wizard.py` + `tests/browser/test_distribution_browser.py` bijwerken (geen Groepsindeling-tab
  meer; kaarten i.p.v.).
- **Nieuwe browsertest** (`tests/browser/`): kaarten renderen, popover opent op klik en sluit op
  Escape/buiten-klik, legenda is inklapbaar, klassenoverzicht aanwezig met de balans-kolommen.
*Succescriterium:* `uv run pytest tests/browser` + `tests/integration` groen (incl. de nieuwe test);
handmatige controle tegen de POC-referentie.

**Stap 4 — documentatie + smoke.** Werk de README (uitvoer-weergave) en de docstrings van
`main.py`/`tasks.py`/`results.py` bij. Draai de app één keer echt (run-aliexpress-skill) met een
**Doorzetten**- én een **Herindelen**-proces en bekijk de resultatenpagina.
*Succescriterium:* screenshot/observatie van kaarten, popover en klassenoverzicht in de PR-tekst.

## Wat verandert NIET

- **De solver, de balans-families, de strategieën, de engine, de tevredenheidsmetriek** — niets. De
  gepinde per-leerling tevredenheidswaarden blijven exact gelijk.
- **De Excel-export (`to_excel`)**: blijft alle sheets schrijven (Groepsindeling, Klassenoverzicht,
  Overgangsmatrix, Leerlingtevredenheid, VervuldeVoorkeuren).
- **De drie analyse-tabellen** op de webpagina blijven pandas-HTML.
- **Wizard/routes** behalve `result_page` (dat één extra JSON-bestand laadt).

## Toekomstig hergebruik (buiten scope, naad schoon houden)

De glossary kent **Tussenstand**: de beste kandidaat-verdeling tijdens het rekenen. Een
tussenoplossing is óók een `SolutionResult`, dus dezelfde `groepsindeling_view`-builder + macro kunnen
die later op de processing-pagina tonen. Houd de builder daarom vrij van "definitief"-aannames en de
macro zelfstandig aanroepbaar met alleen een `GroepsindelingView`. De processing-integratie (welke
solver-callback, hoe vaak verversen, welke velden voorlopig zijn) is expliciet **geen** onderdeel van
dit plan.

## Testcommando's

```bash
uv run pytest tests/ --ignore=tests/integration --ignore=tests/browser -q --no-cov  # per stap
uv run pytest tests/integration                                                     # stap 2
uv run pytest tests/browser                                                         # stap 3
uv run pylint src/aliexpress app.py                                                 # vóór elke commit
```
