---
status: accepted
---

# CP-SAT (OR-Tools) als enige solver-backend

ADR-0009 en ADR-0010 beschrijven de architectuur rond `ProblemSolver`: een PuLP/HiGHS
LP-substraat met vrije-functie-analyses ernaast (`feasibility.py`, `satisfaction.py`,
`optimizationstrategies.py`). Op realistische herindelen-schaal (meerdere jaarlagen,
meerdere honderden leerlingen) bleek HiGHS het probleem niet meer binnen een bruikbare
tijd te bewijzen (DNF na het volledige tijdsbudget). Een CP-SAT-model (OR-Tools) van
hetzelfde probleem bewees een equivalente instantie in seconden.

## Beslissing: CP-SAT vervangt PuLP/HiGHS volledig

CP-SAT is de enige solver-backend. De CP-SAT-implementatie is niet langer een aparte laag
naast PuLP (voorheen `solver/cpsat/`) — na de migratie is `solver/` zelf de CP-SAT-solver:
`modelbuilder.py` (bouwt het model), `engine.py` (orkestreert een solve), `strategies.py`
(aggregeert per-leerling tevredenheid tot één doelfunctie), `feasibility.py` (diagnose),
`_balance_families.py` (de zes balansfamilies) en `scaling.py` (integer-schaling van
gewichten). PuLP, `highspy` en alle PuLP-gekoppelde modules (`problemsolver.py`,
`optimizationstrategies.py`, `pulp_logical.py`, `pulp_thresholds.py`, het oude
`feasibility.py`) zijn verwijderd; de dependency staat niet meer in `pyproject.toml`.

### Determinisme-afspraak

CP-SAT's `num_workers` racet meerdere zoekstrategieën parallel; met een vaste
`num_workers` (8) en vaste `random_seed` (1) is het *bewijs* (de gevonden optimale
waarde) deterministisch, maar de doorlooptijd om dat bewijs te vinden niet. Een solve
termineert altijd met een bewezen optimum, ongeacht hoelang dat duurt — er is geen
tijdslimiet die een sub-optimale uitkomst zou kunnen opleveren.

### `ProblemSolver`-substraat vervalt

Het substraat-patroon uit ADR-0009 (`ProblemSolver` als gedeelde LP-toestand,
analyses als vrije functies die het substraat ontvangen) is vervangen: CP-SAT bouwt bij
elke analyse een verse `cp_model.CpModel` (`Problem`/`SoftProblem`-dataclasses in
`modelbuilder.py`), omdat CP-SAT-variabelen aan precies één model horen en dus niet
gedeeld kunnen worden zoals PuLP-variabelen dat wel konden. De vrije-functie-stijl van
ADR-0009/0010 (analyses als functies, geen methoden op een muterende klasse) blijft het
patroon — alleen het gedeelde substraat is vervangen door "bouw een vers model".

## Overwogen alternatieven

**HiGHS blijven gebruiken met een langere tijdslimiet of een grovere relaxatie** —
verandert de uitkomst (minder strikte klassenbalans) om binnen tijd te blijven; lost het
onderliggende schaalprobleem niet op en tast de kwaliteit van de indeling aan.

**CP-SAT naast PuLP houden (twee backends)** — was de tussenstap tijdens de migratie
(`solver/cpsat/` naast de PuLP-modules), gebruikt om de twee backends op pinned
tevredenheidswaarden te verifiëren (zie `tests/integration/test_solver_equivalence.py`).
Na die verificatie voegt een tweede, ongebruikte backend alleen onderhoudslast toe.

## Consequenties

- `pulp`/`highspy` zijn geen dependencies meer; `ortools` is gepind (een versiebump kan
  een ander, even optimaal ontaard optimum kiezen, wat de gepinde integratiewaarden zou
  verschuiven).
- Geen aparte `cpsat`-naamlaag meer: `solver/` is de CP-SAT-solver. `CpSatProblem` /
  `CpSatSoftProblem` / `CpSatSolution` heten nu `Problem` / `SoftProblem` / `Solution`.
  `SolutionResult`/`GroupComposition` (het gerapporteerde resultaat) wonen in
  `solver/results.py`.
- ADR-0009 en ADR-0010 blijven staan als historisch verslag van de PuLP-architectuur;
  het substraat-patroon dat ze beschrijven is niet meer de huidige implementatie (zie
  hierboven).
