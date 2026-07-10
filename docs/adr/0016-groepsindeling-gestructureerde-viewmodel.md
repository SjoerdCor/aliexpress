---
status: accepted
---

# Groepsindeling-resultaat als gestructureerd view-model, niet als voorgerenderde pandas-HTML

De resultatenpagina rendert vandaag vijf pandas-DataFrames die in
[`tasks.py`](../../src/aliexpress/web/tasks.py) via `df.to_html()` tot HTML-strings worden
platgeslagen en in [`result.html`](../../templates/result.html) rauw worden ingeplakt
(`{{ html | safe }}`). De "Groepsindeling"- en "Klassenoverzicht"-tabellen zijn daardoor kale
tabellen: geen kaarten, geen naamchips, geen detail per leerling, geen balans-in-één-oogopslag.

## Beslissing

De **Groepsindeling** (met het **Klassenoverzicht** erin geïntegreerd) wordt niet langer als
`to_html`-string geleverd, maar als een **gestructureerd, JSON-serialiseerbaar view-model**
(dataclasses met platte Python-waarden) dat in de rapportagelaag wordt gebouwd en door een
herbruikbare Jinja-macro + eigen CSS/JS wordt gerenderd tot groepskaarten met naamchips, een
klik-popover per leerling en een compact klassenoverzicht met balans.

- **Alleen deze twee views** krijgen het view-model. De drie analyse-tabellen
  (Overgangsmatrix, Leerlingtevredenheid, VervuldeVoorkeuren) blijven pandas-HTML in
  `result_tables.json` — daar werkt een tabel prima.
- **De Excel-export (`to_excel`) verandert niet**: die blijft alle sheets schrijven.
- **De component is bewust herbruikbaar ontworpen** (view-model + macro + CSS), zodat dezelfde
  weergave later een **Tussenstand** op de processing-pagina kan tonen: een tussenoplossing van de
  solver is óók een `SolutionResult`, dus dezelfde builder en macro renderen die voorlopige
  verdeling. Die integratie valt buiten deze beslissing; de naad wordt alleen schoon gehouden
  (de builder mag niet aannemen dat een verdeling definitief is).

## Waarom

Chips, een detail-popover per leerling en een balansoverzicht dat "waar komt dat verschil vandaan"
laat zien, kunnen niet uit een `to_html()`-string komen — de template heeft de onderliggende
waarden nodig. Een gestructureerd view-model in de rapportagelaag houdt de template dom, is
triviaal te serialiseren (platte dataclasses → `asdict` → JSON, zoals `SolutionResult` al belooft),
en levert precies de naad die de Tussenstand-weergave (reeds in de glossary voorzien) nodig heeft.

## Overwogen alternatieven

- **Rijkere HTML in de pandas-`Styler` proppen.** Houdt de pijplijn ongewijzigd, maar
  styler-HTML kan geen klik-popovers/kaart-layout dragen en maakt de opmaak onleesbaar en niet
  herbruikbaar. Afgewezen.
- **Alle vijf de views naar view-models.** Groter oppervlak zonder winst: voor de drie
  analyse-tabellen is een tabel de juiste vorm. Buiten scope gehouden.

## Consequenties

- `result_tables.json` bevat voortaan alleen nog de drie analyse-tabellen; de Groepsindeling komt
  uit een nieuw `groepsindeling_view.json`. Tests die de sleutels van `dataframes` toetsen
  (`test_integration_main`, `test_wizard`) verschuiven mee.
- De gepinde integratie-optima (exacte tevredenheid) blijven ongewijzigd: de solver en de metriek
  worden niet aangeraakt.
- Het uiterlijke ontwerp is vastgelegd in een POC-referentie naast het implementatieplan
  ([`docs/plans/groepsindeling-resultaatweergave-poc.html`](../plans/groepsindeling-resultaatweergave-poc.html)).
