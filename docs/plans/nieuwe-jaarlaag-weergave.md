# Plan: nieuwe jaarlaag tonen bij forward-modi

## Doel

Bij de forward-Verdeelmodi tonen de resultaten (groepskaarten, klassenoverzicht, de drie
tabellen) én de Excel-export de **Nieuwe jaarlaag** — de jaarlaag waarin de leerling ná de
Overgang zit, dus één hoger dan de opgeslagen **Huidige jaarlaag**. Nu tonen ze de huidige
jaarlaag, wat een jaar te laag is (een doorgezette leerling uit jaarlaag 5 hoort in het
resultaat onder "Jaarlaag 6").

Forward-modi = **Doorzetten** (`forward`) en **Herindelen met doorzetten**
(`redistribute_and_forward`). **Herindelen met dezelfde groepen** (`redistribute`) kent geen
Overgang: huidige en nieuwe jaarlaag vallen samen, dus daar verandert niets.

Zie `CONTEXT.md` voor de termen *Huidige jaarlaag*, *Nieuwe jaarlaag*, *Overgang* en
*Verdeelmodus*.

## Vastgestelde beslissingen (grilling-uitkomst)

1. **Puur weergave, geen data-migratie.** De opgeslagen `jaargroep`/`Jaarlaag` per leerling
   blijft de huidige jaarlaag (die voedt kandidaatselectie, matching, `get_groups_to =
   jaargroep + 1`). De +1 wordt alleen aan de weergave-rand toegepast.
2. **Één `year_offset: int`** (0 of 1), afgeleid uit de al opgeslagen `mode`. Er wordt niets
   nieuws opgeslagen.
3. **Offset alleen op ints; `None` blijft `None`.** De `None`-cohort komt in forward-modi niet
   voor (bewezen: EDEXML vult jaargroep altijd, en handmatig toegevoegde leerlingen erven
   `default_jaargroep` in `form_parsers.build_participants`). `None` bestaat alleen in het
   kale-Excel/CLI-pad, waar `year_offset` per definitie 0 is.
4. **Terminologie (route B).** Alle *zichtbare* "jaargroep"-labels worden "Jaarlaag" — de
   glossary markeert "jaargroep" al als af te raden. Interne code, JS-variabelen en de
   EDEXML-kolomnaam blijven `jaargroep`.
5. **Geen ADR** (kleine weergaveconventie, geen architectuurbeslissing). Docstrings + README +
   glossary volstaan; de glossary is al bijgewerkt in de grilling-sessie.

## Wat NIET verandert (Surgical Changes)

- De solver, het model, de balans-engine, `results.py`/`GroupComposition.per_year` (keys
  blijven de huidige jaarlaag).
- `year_sort_key` blijft op de rauwe `per_year`-keys werken → sorteervolgorde onveranderd.
- De CLI/Excel-invoerpad (default `year_offset=0`).
- Interne veldnamen: `new_jaargroep[]`, `CANDIDATE_FIELDS`, de EDEXML-kolom `jaargroep`, JS-vars.
- `groups_to.html:44` (`({{ s.jaargroep }})` — een kaal getal, geen term).
- Geen nieuwe browser- of integration-test.

## Uitgangssituatie (feiten uit de code)

- `year_label(year)` in `src/aliexpress/solver/groepsindeling_view.py:26` produceert
  `"Jaarlaag"` (None) / `"Jaarlaag N"`. Aangeroepen vanuit:
  - `groepsindeling_view.build()` → `YearSection.label`, `BalanceRow.label`.
  - `solutions.py:_calculate_group_report` (`:285`, `:295`) → het Groepsindeling-blad in de
    Excel (`to_excel` schrijft alle bladen).
- De template rendert **drie** jaarlaag-plekken
  (`templates/partials/groepsindeling.html`): `section.label` (`:91`), `row.label` (`:117`)
  én het **rauwe int** `chip.year_group` in de popover (`:49`, "jaarlaag {{ chip.year_group }}").
  Alle drie moeten meeschuiven, anders wordt de popover inconsistent met de sectiekop.
- Bouwketen: `main._export()` (`main.py:89`) construeert `SolutionAnalyzer(...)`, roept
  `sa.to_excel()`, de drie `sa.display_*()` en `sa.groepsindeling_view()` aan.
  `_export` wordt aangeroepen door `distribute_students_from_data()` (`main.py:168`), die op
  zijn beurt door `tasks.run_solve_thread()` (`web/tasks.py:119`) wordt aangeroepen. `tasks`
  kent het proces en kan de `mode` lezen.
- `mode` wordt gelezen met `get_process_mode(path)` (`web/routes/processes.py:63`);
  `is_redistribute_mode` (`:77`) is een **andere** as (`redistribute` én
  `redistribute_and_forward` zijn beide "redistribute") en kan hier niet worden hergebruikt.

## Vertical slices (TDD, één test → één implementatie per slice)

### Slice 1 — mode → offset (tracer bullet)

- **RED** — `tests/test_processes.py`: `test_year_offset_for_mode` — `forward → 1`,
  `redistribute_and_forward → 1`, `redistribute → 0`.
- **GREEN** — `web/routes/processes.py`: `year_offset_for_mode(mode: str) -> int` naast
  `is_redistribute_mode`, met docstring die uitlegt dat dit de "verschuift-van-jaar"-as is,
  los van `is_redistribute_mode`.
- **Runs:** `uv run pytest tests/test_processes.py -q --no-cov`.

### Slice 2 — `shift_year` + `year_label(offset)`

- **RED** — `tests/test_groepsindeling_view.py`: `test_year_label_offset` —
  `year_label(5, offset=1) == "Jaarlaag 6"`; `year_label(5) == "Jaarlaag 5"` (default 0
  ongewijzigd); `year_label(None, offset=1) == "Jaarlaag"` (None-veilig).
- **GREEN** — `groepsindeling_view.py`:
  - `shift_year(year: int | None, offset: int) -> int | None` — `None`-veilig
    (`None` → `None`, anders `year + offset`).
  - `year_label(year, offset=0)` gebruikt `shift_year`.
  - Docstrings: leg uit dat `offset` de Nieuwe-jaarlaag-verschuiving is (0 bij modi zonder
    Overgang).
- **Runs:** quick suite.

### Slice 3 — `build(year_offset=…)` schuift kop, balansrij én chip-int

- **RED** — `tests/test_groepsindeling_view.py`: `test_build_shifts_year_for_forward` —
  gebruik de bestaande `_doorzetten_analyzer()`-fixture (jaarlaag-5 movers), roep
  `sa.groepsindeling_view()` aan met offset 1 (via de fixture aangepast in slice 4, of roep
  `build(..., year_offset=1)` direct aan). Assert:
  - de `YearSection.label` == `"Jaarlaag 6"` en `YearSection.year == 6`;
  - de bijbehorende `BalanceRow.label` == `"Jaarlaag 6"`;
  - een `StudentChip.year_group == 6`.
  Plus een `year_offset=0`-variant die 5 houdt (regressieborg).
- **GREEN** — `groepsindeling_view.py`:
  - `build(..., year_offset: int = 0)` en `_ViewBuilder` krijgt het veld.
  - `_build_group_card`: `YearSection.year = shift_year(year, offset)`,
    `label = year_label(year, offset)`.
  - `_build_chip`: `year_group = shift_year(info.get("Jaarlaag"), offset)`.
  - `_year_balance_row`: `label = year_label(year, offset)`.
  - Groeperings-/sorteersleutels blijven op de **rauwe** `year` (huidige jaarlaag).
- **Runs:** quick suite.

### Slice 4 — `SolutionAnalyzer(year_offset=…)` schuift het Groepsindeling-blad

- **RED** — `tests/test_solutions.py`: `test_group_report_shifts_year_for_forward` — bouw een
  `SolutionAnalyzer(..., year_offset=1)` met jaarlaag-5 movers; assert dat de rijlabels van
  `_calculate_group_report()` / het Groepsindeling-blad `"Jaarlaag 6"` bevatten (en de
  `year_offset=0`-variant `"Jaarlaag 5"`).
- **GREEN** — `solutions.py`:
  - `SolutionAnalyzer.__init__(..., year_offset: int = 0)` → `self.year_offset`.
  - `_calculate_group_report`: `year_label(year, self.year_offset)` op `:285` en in de
    `row_order` op `:295`.
  - `groepsindeling_view()`: geef `year_offset=self.year_offset` door aan `build(...)`.
- **Runs:** quick suite (`tests/test_solutions.py` + `tests/test_groepsindeling_view.py`).

### Slice 5 — bedrading mode → offset → weergave (end-to-end plumbing)

- **GREEN (pass-through, gedekt door slice 1 + 4)** — geen nieuwe test; succescriterium is dat
  de bestaande integration-tests groen blijven met default `0`:
  - `main._export(result, preference_data, target_groups, year_offset=0)` geeft
    `year_offset` door aan `SolutionAnalyzer(...)`.
  - `main.distribute_students_from_data(..., year_offset: int = 0)` reikt het door aan
    `_export`.
  - `web/tasks.run_solve_thread`: leid `year_offset = year_offset_for_mode(get_process_mode(
    get_process_path(ctx.school_id, ctx.process_name)))` af en geef het mee aan
    `distribute_students_from_data(...)`.
- **Rationale (Goal-Driven):** de weergave-honorering is bewezen in slice 3–4, de mapping in
  slice 1. Dit is risicoloze parameter-doorgave; de default `0` houdt CLI/Excel en alle
  bestaande callers ongewijzigd.
- **Runs:** `uv run pytest tests/integration` (moet groen blijven; asserteren exacte
  tevredenheidswaarden, niet labels) + quick suite.
- **Handmatige verificatie:** via de `run-aliexpress`-skill een doorzet-proces draaien en op de
  resultatenpagina bevestigen dat jaarlaag 5 → "Jaarlaag 6" toont.

### Slice 6 — terminologie-harmonisatie (route B, zichtbare labels)

- **RED/UPDATE** — `tests/browser/test_herindelen_browser.py:219` asserteert nu
  `["— jaargroep —", "Jaargroep 6", "Jaargroep 7"]`; werk die bij naar de nieuwe
  "Jaarlaag"-labels.
- **GREEN** — zichtbare label-strings → "Jaarlaag" (kleine letter midden in de zin):
  - `templates/upload_edexml.html`: "jaargroep die moet worden ingedeeld" →
    **"Jaarlaag (huidig) die moet worden ingedeeld"**; "Welke jaargroepen deel je opnieuw in?"
    → **"Welke jaarlagen (huidig) deel je opnieuw in?"**.
  - `templates/roster.html`: dropdown-placeholder "— jaargroep —" → "— jaarlaag —";
    optie "Jaargroep ${j}" → "Jaarlaag ${j}"; pill "jaargroep ${jaargroep}" → "jaarlaag …";
    foutmelding "Geef ook de jaargroep aan." → "Geef ook de jaarlaag aan."
  - `templates/select_groups.html`: "jaargroepen" → "jaarlagen" (`:5`, `:8`).
  - `templates/processes.html`: de twee info-popover-teksten (`:41`, `:51`) →
    "jaarlaag"/"jaarlagen".
- **Runs:** `uv run pytest tests/browser -q` (of minimaal het aangepaste bestand).
- **Let op:** interne `name="new_jaargroep[]"`, JS-vars en `title`-attributen blijven ongemoeid.

### Slice 7 — documentatie

- `README.MD` (rond `:99`, de group-cards-alinea): één zin dat bij Doorzetten en Herindelen
  met doorzetten de getoonde jaarlaag de nieuwe (post-Overgang) jaarlaag is, terwijl de invoer
  de huidige jaarlaag vraagt.
- `CONTEXT.md`: **al gedaan** in de grilling-sessie (`Huidige jaarlaag`, `Nieuwe jaarlaag`,
  gescherpte `Jaarlaag`).

## Commit-strategie

Eén commit per groene slice (test + implementatie samen), conform CLAUDE.md. Werk op een
verse feature branch vanaf `master` (bijv. `feature/new-year-layer-display`).

## Testselectie per slice

- Slice 1: `tests/test_processes.py` (raakt `web/routes/processes.py` → volgens CLAUDE.md ook
  browser-suite; maar de wijziging is een pure helper — draai quick + `test_processes`).
- Slice 2–4: quick suite.
- Slice 5: `tests/integration` + quick suite (raakt `main.py`, `web/tasks.py`).
- Slice 6: `tests/browser`.
- Vóór merge: volledige suite `uv run pytest tests/` **en** `uv run pytest tests/ -m slow`.
