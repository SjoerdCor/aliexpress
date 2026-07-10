# Plan: nieuwe Groepsindeling-resultaatweergave (kaarten + chips + klassenoverzicht)

**Status:** concept, ter goedkeuring. Geschreven als zelfstandige hand-off naar een verse
implementatiesessie. Start een nieuwe branch vanaf `master` (bijv.
`feature/groepsindeling-weergave`), niet direct op master.

**Leesvolgorde voor de verse sessie:** dit plan + `CLAUDE.md` + [ADR-0016](../adr/0016-groepsindeling-gestructureerde-viewmodel.md).
Het uiterlijke ontwerp staat als open-baar-in-de-browser referentie naast dit plan:
[`groepsindeling-resultaatweergave-poc.html`](./groepsindeling-resultaatweergave-poc.html) — open
dat bestand in een browser; dat ís het doelontwerp (op twee kleine deltas na, zie §Ontwerp).

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
  - **Chip:** gebruik de display_name + de oude groep als **3-letterafkorting** (bijv. `Kik`). Geen
    kleur op de chip in ruststand.
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
- **Extra zekerheid**: koppel de badge aan de bestaande vormtaal `.badge-zeker--partial` (lichte
  outline) / `.badge-zeker--full` (goud gevuld) en gebruik de **bestaande UI-teksten uit het
  voorkeuren-formulier** (niveaus *minstens één voorkeur* / *belangrijkste voorkeur*), **niet** de
  POC-placeholder "minstens zo tevreden als X%". Bron: `MinimaleTevredenheid` in `students_info`
  (`NaN` = geen badge). Stap 0 stelt de exacte mapping naar de twee niveaus vast.
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
    first_name: str; full_name: str; origin_abbrev: str; origin_full: str
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

**Stap 0 — empirische verificatie (geen productiecode).** Stel vast en noteer in de PR-tekst:
1. **Voornaam vs. volledige naam:** hoe is de leerlingnaam ingevoerd/opgeslagen, en hoe leid je de
   voornaam voor de chip af (eerste token?) versus de volledige naam voor de popover? Controleer
   `preferences_form.StudentEntry` / de roster-invoer en de display-namen.
2. **"Geen voorkeuren" → `—`:** hoe detecteer je dat een leerling geen (positieve) voorkeuren heeft
   (bijv. geen `Graag met`-rijen / alle `weights ≤ 0`), zodat je `satisfaction=None` zet i.p.v. 100%.
   Let op: geen voorkeuren = `—`. Bij geen negatieve voorkeuren moet wel 100% worden getoond.
3. **Liever-niet & Niet-in vervuld-status:** `result.satisfied` dekt alleen `Graag met`. Leid de
   vervuld-status van `Liever niet met` en `Niet in` af uit `assignment` (doel in dezelfde groep?
   voor Niet-in: leerling niet in de uitgesloten groep — altijd waar want harde eis). Bevestig de
   bron van die rijen (`input_sheet` / `preferences`). Let op: negatieve wensen zijn Liever niet met
4. **Extra-zekerheid-niveaus:** welke `MinimaleTevredenheid`-waarden horen bij `.badge-zeker--partial`
   vs `--full`, en wat zijn de exacte bestaande UI-teksten? Zoek in het voorkeuren-formulier
   (`templates/preferences_form.html` / `preferences_form.py`) en hergebruik die.
5. **Bezetting alleen bij Doorzetten:** bevestig dat `occupancy > 0` samenvalt met één jaarlaag
   (Doorzetten) en dat Herindelen bezetting 0 heeft — de kolomkop-regel "nieuw + zittend" geldt dan
   alleen bij één jaarlaag.
*Succescriterium:* de vijf antwoorden staan in de commit-/PR-tekst; onduidelijkheden zijn opgelost
vóór stap 1.

**Stap 1 — view-model + builder (unit).** Nieuwe module + rode test (`tests/test_groepsindeling_view.py`):
bouw uit een handgemaakte `SolutionResult` + `students_info` + `preferences` een `GroepsindelingView`
en assert: groepskaarten met totalen; jaarlaag-secties met de juiste `size`; `SexColumn` met
`new_count`/`occupancy_count`/`total_count`; chips met voornaam/volledige naam/herkomst; `satisfaction`
geclampt en `None` bij geen voorkeuren; `wishes` met correcte `fulfilled` (incl. liever-niet/niet-in);
`zekerheid` partial/full/None; en `balance_rows` met kloppende `size_diff`/`sex_imbalance` en rijvolgorde
(Totaal, dan jaarlagen numeriek, `None`-jaarlaag als "Jaarlaag" na Totaal). Dek het Doorzetten-geval
(bezetting > 0, één jaarlaag) én het Herindelen-geval (meerdere jaarlagen, bezetting 0).
*Succescriterium:* nieuwe test groen; quick suite groen; `uv run pylint` schoon.

**Stap 2 — serialisatie + pijplijn.** `main.py:_export` geeft `view` terug; `tasks.py` schrijft
`groepsindeling_view.json`; `dataframes` bevat nog slechts de drie analyse-tabellen. Pas de tests aan
die de `dataframes`-sleutels toetsen (`tests/integration/test_integration_main.py` rond regel 219/364,
`tests/test_wizard.py` rond de `result_tables.json`-assert) én `tests/browser/test_distribution_browser.py`
(dat `result_tables.json` verwacht — voeg de `groepsindeling_view.json`-assert toe).
*Succescriterium:* quick + `uv run pytest tests/integration` groen; het view-JSON bestaat na een run.

**Stap 3 — template-macro + CSS + JS.** Nieuwe partial `templates/partials/groepsindeling.html` met de
macro; `result.html` gebruikt die en houdt de drie analyse-tabellen. CSS achter in `static/style.css`
(hergebruik de bestaande tokens/kleuren; geen nieuwe donkere variant). Popover-JS zoals in de POC
(klik-toggle, plaatsing rechts/links, sluiten via klik-buiten/Escape). Nieuwe browsertest
(`tests/browser/`): kaarten renderen, popover opent op klik en sluit op Escape/buiten-klik, legenda is
inklapbaar, klassenoverzicht is aanwezig met de balans-kolommen.
*Succescriterium:* `uv run pytest tests/browser` groen (incl. de nieuwe test); handmatige controle
tegen de POC-referentie.

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
