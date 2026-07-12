# Metingen: tijdschatting (ETA) voor de processing-pagina

**Status:** slice 7 afgerond, 2026-07-12. Voedt **slice 7 (Kalibratie)** van
[plan-processing-page-ux.md](plan-processing-page-ux.md). De eerste cijfers kwamen uit het
tunen van een testdata-scenario (`main_redistribute_and_forward`); slice 7 heeft de
**mechanische conclusies** daarna bevestigd op het automatische webpad op deze machine (zie
"Bevestiging op deze machine"). De absolute getallen blijven machine- en run-afhankelijk
(5× spreiding), maar welke voorspeller signaal draagt is stabiel.

## Wat is er gemeten, en hoe

- Pijplijn: `main.distribute_students_from_data` → `engine.solve_within_minimal_relaxation`
  (het automatische webpad: minimal-relaxation + lexmaxmin). Dit is exact wat de webapp
  draait; `solve_with_fixed_balance` is CLI/tests en is *niet* gemeten.
- Per-stage duren via de `ProgressListener.stage_finished(stage, seconds)` uit slice 1
  (`stage` ∈ `floor`, `balance`, `satisfaction`). Per-level duren uit de bestaande
  INFO-logregels van `strategies._lexmaxmin`.
- Eén machine (Windows 11, CP-SAT `num_workers=8`). Scenario: `redistribute_and_forward`,
  72 leerlingen, 4 bestemmingsgroepen, 3 cross-stamgroep niet-samen-regels.

## De centrale bevinding

**Solve-tijd wordt bepaald door de *structuur* van de niet-samen-regels, niet door de
probleemgrootte, en is bovendien sterk niet-deterministisch tussen runs.** Twee gevolgen
die de ETA-strategie sturen:

1. **Een klif.** Eén integer strakker in één regel bepaalt of het scenario seconden of
   minuten kost — er is geen gladde helling van grootte naar tijd.
2. **Grote run-tot-run spreiding.** Dezelfde instantie (zelfde config/seed) mat vandaag
   **88,7s** en eerder **432s** — ~5×. Oorzaak: CP-SAT racet 8 zoekstrategieën parallel;
   welke thread als eerste een bound propageert hangt af van OS-scheduling. De
   herindelen-acceptatietest documenteert ditzelfde (347-869s op één instantie).

Daaruit volgt: een **betrouwbare ETA vóór de solve bestaat niet**, maar een **adaptieve
ETA ná de eerste level(s) draagt wél signaal** — zie de voorspeller-toetsing.

## Ruwe metingen

### Totaaltijd vs. configuratie (redistribute_and_forward, 4 groepen)

| Config | Totaaltijd |
|---|---|
| 70 lln, random, 5 losse regels | ~6 s |
| 75 lln, random, 30 regels | ~23 s |
| 150 lln, random, 5 regels | ~27 s |
| 120 lln, random | 6,7-19 s (varieert met #regels) |
| 72 lln, 3-12 gematigde Max=2-regels | 31 → 15 s (méér regels = sneller) |
| 60 lln, tight regels (2,2,**1**) | 18,7 s |
| 72 lln, los (2,2,**2**) | 31,0 / 31,4 s |
| **72 lln, tight (2,2,1)** | **88,7 s / 432 s** (twee runs, zelfde instantie) |
| 88 lln, realistische 3 regels | 339,9 s |

Merk op: méér gematigde regels maakt het *sneller* (constraints snoeien de zoekruimte),
dus "meer constraints" is geen tijd-voorspeller.

### Per-stage en per-level (72 lln, twee configuraties)

**Snel — Max=2, totaal 31,4 s**

| Stage | Duur |
|---|---|
| floor (stap 1) | 0,1 s |
| balance (stap 2) | 1,8 s |
| satisfaction (stap 3) | 29,1 s |

Levelduren stap 3: 3,51 · 4,11 · 4,17 · 2,48 · 4,10 · 5,48 s

**Hard — Max=1, totaal 88,7 s**

| Stage | Duur |
|---|---|
| floor (stap 1) | 0,2 s |
| balance (stap 2) | 8,5 s |
| satisfaction (stap 3) | 79,6 s |

Levelduren stap 3: 9,87 · 9,38 · 18,76 · 12,08 · 7,46 · 8,76 · 8,51 s

Twee patronen die de ETA dragen:
- **Stap 3 domineert** (29/31 resp. 80/89 van de tijd). Stap 1 is verwaarloosbaar; stap 2
  is klein.
- **Per-level duren clusteren binnen een run** (snel 2,5-5,5 s; hard 7-19 s) en het
  *aantal* levels is vergelijkbaar (6-7). Het is dus de *levelduur* die snel vs. hard
  onderscheidt, niet het aantal.

### Bevestiging op deze machine (slice 7, automatisch webpad)

Twee verse runs op deze machine via `engine.solve_within_minimal_relaxation` (exact het
webpad), met een opnemende `ProgressListener`. Ze bevestigen de bovenstaande conclusies:

| Instance | Totaal | floor | balance | satisfaction | levels | interim-cadans |
|---|---|---|---|---|---|---|
| herindelen 18 lln, 3 groepen | 0,06 s | 0,008 s | 0,016 s | 0,033 s | 0 | n.v.t. (2 interim: floor+balance) |
| herindelen 88 lln, 4 groepen | 231,9 s | 0,27 s | 22,0 s | **209,6 s (90%)** | 7 | 15–32 s tussen updates |

- **Stap 3 domineert** opnieuw (90% bij de harde run); floor verwaarloosbaar; balance klein.
- De 232 s ligt onder de gedocumenteerde 340–870 s-band voor ditzelfde 88-lln-scenario — een
  verse bevestiging van de grote run-tot-run-spreiding, geen tegenspraak.
- **Interim-cadans 15–32 s**: de Tussenstand ververst tijdens een lange solve een handvol
  keer (hier 7 levels), grofmazig maar merkbaar — hij staat niet minutenlang stil, en
  verdient zijn plek dus juist in de lange runs.
- **Callback-overhead (plan-risico): sluitend.** Interim vuurt alléén op stage-grenzen
  (floor, balance, per level ≈ 9× bij de harde run), nooit tijdens de CP-SAT-zoektocht (er is
  bewust geen `CpSolverSolutionCallback`). De solver-side `read_solution` mat ≈ 0 ms. De enige
  niet-triviale kost — de pandas-`GroepsindelingView`-bouw in de main-adapter — draait ≤ 9×
  per solve en zit achter 2 s-demping in de weblaag. Dat is < 1% van een solve en verdwijnt in
  de 5× intrinsieke spreiding; een A/B t.o.v. master op totale wall-time is daarom zinloos
  (de ruis is groter dan het effect). Overhead is bij ontwerp begrensd, niet in wall-time meetbaar.

## Toetsing van de vier voorspellers uit het plan

| Voorspeller | Oordeel | Bewijs |
|---|---|---|
| (i) probleemgrootte vooraf (lln × groepen) | **Zwak** | Structuur domineert; kan Max=1 vs Max=2 niet zien; totaaltijd niet-monotoon in aantal (120 lln 6,7-19 s); + ~5× spreiding tussen runs. |
| (ii) duur van stage 1 (floor) | **Nutteloos** | 0,1-0,2 s in beide gevallen — draagt geen informatie. |
| (iii) duur van stage 2 (balance) | **Zwak** | 1,8 vs 8,5 s: enig signaal, maar klein absoluut en klein t.o.v. stap 3. |
| (iv) gemiddelde levelduur tot nu toe | **Sterk (binnen de run)** | Eerste level 3,5 s (snel) vs 9,9 s (hard) onderscheidt al; volgende levels blijven in dezelfde band; stap 3 domineert. |

## Aanbevolen ETA-vorm (voorstel voor slice 8, ter beslissing owner)

- **Vóór de eerste level** (stap 1/2 lopen nog): alleen een **statische verwachting** op
  probleemgrootte — de eerlijke degradatie uit het plan ("dit duurt meestal minder dan
  een minuut" / "dit kan enkele minuten duren"). Beloof geen minutenprecisie.
- **Vanaf de eerste 1-2 afgeronde levels**: **adaptief** —
  `resterende ≈ (nog te verwachten levels) × (gemiddelde levelduur tot nu toe)`,
  **licht overschattend**, tekstueel als "naar verwachting nog ~X minuten", en bijgesteld
  na elk afgerond level. De levelduur uit stap 3 (predictor iv) is de motor; stap 1/2
  negeren voor de schatting.
- **Onzekerheden expliciet houden** ("ongeveer"): het aantal *resterende* levels is vooraf
  onbekend (de plateau-meldingen tonen de voortgang: "N leerlingen kunnen nog omhoog"
  krimpt naar 0), en de spreiding blijft groot. Daarom bewust ruw en aan de hoge kant —
  precies de ontwerpkeuze die het plan al vastlegt.

**Antwoord op de kernvraag "zegt het veel of de eerste solve 10 s of 1 min duurde?":**
ja — het is verreweg het sterkste signaal, mits gemeten *binnen de lopende run* op de
eerste stap-3-level(s). Het discrimineert snel vs. hard met een factor die de rest van de
run vasthoudt. Wat het niet kan: vooraf voorspellen, en de ~5× spreiding tussen runs
wegnemen.

## Disclosure-gating: wanneer rijke voortgang tonen? (owner-vraag, ter beslissing)

Los van de ETA speelt een tweede ontwerpvraag (owner, 2026-07-12): plateau-meldingen en de
Tussenstand zijn waardevol bij een lange solve, maar bij een korte run (10–30 s, vaak)
verschijnen ze allemaal in de laatste seconden vlak vóór de redirect — onrustig. Voorstel:
**toon plateaus + Tussenstand (+ ETA) pas als de verstreken tijd een drempel (~30 s) passeert;
de kern (stepper, invoeroverzicht, sociogram) altijd.**

De meetdata onderbouwt dit:
- **Een klif, geen helling.** Solves zijn óf snel (seconden) óf minuten; de tijd is niet
  vooraf te voorspellen. Een drempel op *verstreken* tijd (niet op een voorspelling) is
  daarom de enige robuuste knop.
- **De rijke componenten verschijnen sowieso pas laat bij een lange run.** Bij de 232 s-run
  was balance alleen al 22 s; het eerste plateau volgde daarna. Een drempel rond ~30 s
  onderdrukt precies de flits-bij-korte-runs zonder een echt lange run iets te ontnemen — die
  is na 30 s nog volop bezig.
- **Gratis bovenop het gebouwde.** De backend blijft alles emitten; de gate is een dunne
  JS-drempel vóór `updatePlateaus`/`updateInterimResult` in `processing.html`. Slices 4/5
  worden niet weggegooid, alleen later onthuld.

**Owner-beslissing (2026-07-12):** gate op **~30 s** verstreken tijd. Achter de gate:
plateau-meldingen + Tussenstand. Altijd zichtbaar: de kern (stepper, invoeroverzicht,
sociogram) én de verstreken-tijd/ETA-regel — die laatste dient als geruststelling vanaf de
start ("dit duurt meestal < 1 min"), niet als onrust. Dit is een eigen kleine slice
(JS-drempel in `processing.html`), los van slice 8.

## Instrumentatie-gat: per-level duur bereikt de weblaag niet

De sterkste voorspeller (predictor iv, de levelduur) zit **nu alleen in de logs**, niet in
het gestructureerde voortgangskanaal. `strategies._lexmaxmin` berekent de levelduur wel
(`time.perf_counter() - t_start`, [strategies.py:169](../src/aliexpress/solver/strategies.py))
maar `_report_level` geeft aan de listener alleen `plateau_finished(min_satisfaction,
count)` door — **zonder seconden**. Er is geen `solve_finished(label, seconds)`-event (het
plan ontwierp dat, maar slices 1-5 hebben het niet gebouwd — logisch, ETA is slice 8).

Gevolg voor de bouw:
- **Slice 7 (meten)** kan de levelduren gewoon uit de logs halen (zie meetopstelling
  hieronder) — geen codewijziging nodig.
- **Slice 8 (schatter in `progress_writer`, weblaag)** heeft de levelduur in `progress.json`
  nodig om predictor (iv) te voeden. Dat vraagt één kleine uitbreiding: geef de seconden
  mee aan `plateau_finished` (de waarde ligt al klaar op regel 169) of voeg het geplande
  `solve_finished(label, seconds)` toe, en zet het per level in `progress.json`. Zonder die
  uitbreiding ziet de weblaag alleen de **per-stage totalen** (`stage_seconds`), en van stap
  3 pas het totaal aan het eind — te laat voor een lopende schatting.

## Herbruikbare meetopstelling (voor slice 7)

De benchmark-scripts waarmee bovenstaande tabellen zijn gemaakt waren wegwerp-`python -c`
one-liners; de herbruikbare kern is klein. Hang een opnemende listener aan de solve en lees
de levelduren uit de strategies-logs:

```python
import logging
from aliexpress.solver.progress import ProgressListener

class RecordingListener(ProgressListener):
    """Captures per-stage totals + plateaus for ETA calibration."""
    def __init__(self):
        self.stage_seconds = []   # [(stage, seconds)] — floor / balance / satisfaction
        self.plateaus = []        # [(min_satisfaction, n_can_improve)]
    def stage_finished(self, stage, seconds):
        self.stage_seconds.append((stage, seconds))
    def plateau_finished(self, min_satisfaction, n_can_improve):
        self.plateaus.append((min_satisfaction, n_can_improve))

class LevelDurations(logging.Handler):
    """Per-level seconds live only in the logs (see the gap above)."""
    def __init__(self):
        super().__init__(); self.msgs = []
    def emit(self, record):
        if "lexmaxmin level" in record.getMessage():
            self.msgs.append(record.getMessage())  # "...plateau=0.62, 34 above, 8.51s"

# rec = RecordingListener()
# logging.getLogger("aliexpress.solver.strategies").addHandler(LevelDurations())
# distribute_students_from_data(pref, groups, not_together, year_offset=1, listener=rec)
```

Een controleerbare **harde fixture zonder de 30-min slow-suite**:
`main_redistribute_and_forward` (72 lln) met de derde niet-samen-regel op
`Max_aantal_samen=1` geeft een meerminuten-solve met bekende stagestructuur; standaard
(Max=2) is het een stabiele ~30 s-run. Zo kun je de meetopstelling snel valideren voordat
je de dure suites draait.

## Status na slice 7

Afgerond in slice 7:
- **Bevestigd op deze machine** op het automatische webpad (zie "Bevestiging op deze
  machine"). De mechanische conclusies — structuur-gedomineerd, klif, ~5× spreiding, stap 3
  domineert, predictor (iv) is de motor — houden stand.
- **Callback-overhead** sluitend afgehandeld: bij ontwerp begrensd (interim alleen op
  stage-grenzen, ≈ 9× per solve), solver-side ≈ 0 ms, niet in wall-time meetbaar tegen de 5×
  ruis. Een A/B t.o.v. master op totale wall-time is daarom niet zinvol.
- **Disclosure-gating-vraag** geanalyseerd (zie die sectie), ter beslissing owner.

Bewust niet meer gedaan: een aparte `tests/integration`/slow-suite-meetronde. De 88-lln
automatische run dekt het slow-suite-regime; `tests/integration` bestaat uit triviale
sub-seconde-instances (vgl. de 18-lln-run) die de ETA-analyse niets toevoegen. Desgewenst
alsnog te draaien, maar naar verwachting zonder nieuw signaal.

Beslissingen die naar **slice 8** doorschuiven:
- **Instrumentatie-gat dichten** (eerste bouwstap van slice 8, geen slice-7-werk): geef de
  levelduur mee aan `plateau_finished` (de waarde ligt klaar op
  [strategies.py:169](../src/aliexpress/solver/strategies.py)) of voeg
  `solve_finished(label, seconds)` toe, en zet het per level in `progress.json`. Zonder dit
  voedt niets predictor (iv) live.
- Definitieve **ETA-vorm** + **gating-drempel**, na akkoord owner.
