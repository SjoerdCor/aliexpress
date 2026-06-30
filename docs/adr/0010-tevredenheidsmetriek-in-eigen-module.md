---
status: accepted
---

# Tevredenheidsmaat in eigen module (`satisfaction.py`)

`ProblemSolver` vermengt de **probleemdefinitie** (welke voorkeuren worden vervuld gegeven
een indeling) met de **tevredenheidsmaat** (gegeven vervulde voorkeuren → tevredenheid per
leerling). Dat zijn twee aparte modelleerkeuzes, maar ze zaten als methoden ingebakken in
dezelfde klasse naast de constraint-bouwers.

Tegelijk heette het bestand dat de meetkundige kern al bevatte `preferences_utils.py` — een
"utils"-rommellade zonder coherent thema. De tevredenheidsfunctie (`get_satisfaction_integral`)
woonde er al, maar de toepassing ervan op het LP (`_calculate_weighted_preferences`,
`calculate_student_satisfaction`) stond nog op `ProblemSolver`.

## Beslissing: maat als vrije functies in `satisfaction.py`

De tevredenheidsmaat staat als vrije functies in `satisfaction.py` (hernoemd uit
`preferences_utils.py`). De functies ontvangen het `ProblemSolver`-object als `solver`,
conform het patroon van `feasibility.py` (ADR-0009).

### De drieledige grens

| Module | Vraag |
|---|---|
| `problemsolver` | Welke voorkeuren worden vervuld gegeven een indeling? |
| `satisfaction` | Hoe meten we tevredenheid per leerling uit vervulde voorkeuren? |
| `optimizationstrategies` | Hoe aggregeren we per-leerling tevredenheid tot één doelwaarde? |

### De maat is een modelleerkeuze

De huidige maat is de **integraal van 0.5^x**: de marginale waarde van elke extra voorkeur
daalt exponentieel. Het lexmaxmin-algoritme (in `optimizationstrategies`) drijft het gevolg
dat iedereen eerst voorkeur 1 krijgt vóór iemand voorkeur 2 — dat is geen eigenschap van
de maat zelf, maar van de aggregatiestrategie. Het module bevat ook `get_satisfaction_percentage`
(lineaire maat: gewogen fractie gehonoreerde voorkeuren) als alternatief, om de
ontwerpruimte expliciet te maken.

### `get_graag_met` → `preferences_data.py`

De accessor `get_graag_met` (positieve-voorkeur-rijen uit de DataFrame) is een
voorkeuren-accessor, geen tevredenheidsconcept. Hij is verplaatst naar `preferences_data.py`,
zodat zowel `satisfaction.py` als `feasibility.py` hem kunnen importeren zonder circulaire
afhankelijkheden.

### `pulp_thresholds.py` als gedeelde leaf

De big-M-indicatormechaniek (`apply_threshold_constraint(s)`) is geëxtraheerd naar
`pulp_thresholds.py`. Zowel `satisfaction.py` als `optimizationstrategies.py` gebruiken
deze constraints; een gedeelde leaf vermijdt duplicatie en houdt de importgraaf cykelvrij.

### Write-back eliminatie

`self.weighted_satisfied` en `self.weights` waren redundant: ze werden berekend in de
metriekfuncties en teruggeschreven op het object alleen om door `extract_solution` te worden
uitgelezen. Beide zijn afleidbaar uit `self.satisfied` en `self.preferences`:

- `weights` = `dict(get_graag_met(self.preferences)["Gewicht"])` — pure invoer.
- `weighted_satisfied[k]` = `s * w` als `w > 0`, anders `(1 - s) * w`, met
  `s = bool(round(self.satisfied[k].value()))`.

De integratietests (exacte per-leerling-tevredenheid) bevestigen dat de eliminatie
gedrags-neutraal is.

## Overwogen alternatieven

**Maat als methoden op `ProblemSolver` laten** — eenvoudig, maar de klasse zou een
modelleerkeuze (de maat) blijven mengen met de probleemdefinitie (de constraints). Wisselen
van maat zou `ProblemSolver` aanraken.

**Aparte klasse `SatisfactionMetric`** — maakt de keuze explicit als object, maar de
functies zijn stateless. Vrije functies met `solver` als parameter zijn directer, conform
het patroon in `feasibility.py`.

## Consequenties

- `satisfaction.py` is de coherente woning voor de tevredenheidsmaat; een alternatieve maat
  wisselen vergt alleen een aanpassing in dit bestand.
- `ProblemSolver` bevat geen metrieklogica meer; alleen constraint-bouwers, de hoofd-solve
  en de publieke levenscyclus.
- `optimizationstrategies.py` bevat `set_optimization_target` als vrije functie — de
  keuzeschakelaar die `solver.optimize` vertaalt naar een concrete aggregatiestrategie.
- De refactor is terugdraaibaar: commits 4–9 zijn atomair en gedrags-neutraal.
