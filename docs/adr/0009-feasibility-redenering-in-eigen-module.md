---
status: accepted
---

> **Update (ADR-0013):** de PuLP/HiGHS-backend en het `ProblemSolver`-substraat die dit
> ADR beschrijft, zijn vervangen door CP-SAT. Dit document blijft staan als historisch
> verslag van die architectuur; zie ADR-0013 voor de huidige implementatie.

# Feasibility-redenering in eigen module (`feasibility.py`)

`ProblemSolver` bevat de veranderlijke PuLP-toestand: de gedeelde beslissingsvariabelen
(`in_group`, `studentsatisfaction`) en de hoofd-LP (`self.prob`). Die toestand is het
*substraat* — het bouwmateriaal voor elke analyse. De analyses zelf — "is dit probleem
oplosbaar?", "wat is het minimale relaxatiebudget?", "welke voorkeursfamilie veroorzaakt
onoplosbaarheid?" — zijn *vragen aan het model*, geen deel van het model.

Vroeger zaten al die vragen als methoden op `ProblemSolver`, door elkaar met de
constraint-bouwers en de optimalisatielogica. Het gevolg: de klasse kende zijn eigen
infeasibility-diagnose, zijn eigen balans-check én de hoofd-solve. Dat maakt de klasse
moeilijker te lezen en de analyses moeilijker te vinden.

## Beslissing: substraat-object + analyses-als-functies

Alle feasibility-redenering staat in `feasibility.py` als vrije functies. Elke functie
ontvangt het `ProblemSolver`-object als parameter, bouwt een *wegwerp-LP* (een
`pulp.LpProblem` die niet `self.prob` is), stelt één vraag, en geeft een gewone Python-waarde
terug. De `ProblemSolver` fungeert als substraat — het levert variabelen en constraint-bouwers;
het onthoudt geen analyse-resultaten.

De functies in `feasibility.py`:

| Functie | Vraag |
|---|---|
| `check_balance_feasibility(solver)` | Zijn de vaste balanslimieten überhaupt haalbaar? (handmatig pad) |
| `minimal_relaxation_budget(solver, groupbalance)` | Wat is het kleinste gewogen balansrelaxatiebudget `R*` waarbij elke leerling nog een wens kan halen? (automatisch pad) |
| `feasible_when_relaxed(solver, …)` | Wordt het haalbaar als één voorkeursfamilie wordt verzacht? (voor diagnose) |
| `diagnose(solver)` | Welke voorkeursfamilie veroorzaakt onoplosbaarheid? |

`_balance.py` is het gedeelde hulpbestand voor `solver/`: het bevat `GroupBalance`,
`STRICTEST_BALANCE` en `get_solver()`. Het staat buiten `problemsolver.py` en
`feasibility.py` om een circulaire import te vermijden (beide importeren van hier;
dit bestand importeert geen van beide). De CBC/HiGHS\_CMD-fallback is verwijderd nadat
empirisch bevestigd is dat `pulp.HiGHS().available()` betrouwbaar `True` teruggeeft.

## Overwogen alternatieven

**Alle analyses als methoden op `ProblemSolver` laten** — eenvoudig, maar de klasse groeit
mee met elke nieuwe analyse. De boundary tussen "het model bouwen" en "redeneren over het
model" vervaagt.

**Aparte klasse `FeasibilityAnalyzer(solver)`** — maakt de scheiding expliciet maar voegt
een initialisatiestap toe zonder voordeel: de functies zijn stateless, een klasse zou alleen
`self.solver` bevatten. Vrije functies zijn directer.

## Consequenties

- `feasibility.py` is de coherente woning voor alle feasibility-redenering; ADR-0008
  (diagnose) past naadloos in dit patroon.
- `ProblemSolver` bevat geen analyse-logica meer; alleen constraint-bouwers, de hoofd-solve
  en het substraat.
- `main._check_feasibility` roept `feasibility.check_balance_feasibility(ps)` rechtstreeks
  aan — geen methode-doorstuurlaag.
- Nieuwe analyses (per-element diagnose, zie ADR-0008) kunnen in `feasibility.py` worden
  toegevoegd zonder `ProblemSolver` aan te raken.
- `GroupBalance` staat nog in `problemsolver.py`; een module-niveau default voor
  `minimal_relaxation_budget` vereist verplaatsing naar een neutraal bestand. Geparkeerd:
  zodra dat een meerwaarde heeft (bijv. configureerbare balanslimieten), is dat de
  volgende stap.
