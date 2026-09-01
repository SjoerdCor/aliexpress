status: implemented

# Plan Balansgrenzen

Dit plan werkt de instelbare bovengrenzen op de automatische balansrelaxatie
uit ADR-0017 uit. De vaste `GroupBalance`-solver en de betekenis van
`BalanceMaxima` vallen buiten scope.

## Plakken

### Plak 1 — maxima in het automatische solverpad

`BalanceMaxima` begrenst per balansfamilie de soft slack van het automatische
pad. `None` betekent onbeperkt; de bestaande defaults blijven data-driven.

### Plak 2 — invoer en normale verwerking

De processing-pagina bewaart de zes maxima en gebruikt ze bij een automatische
solve. De bestaande balans-leximin en de voortgangsfases blijven inhoudelijk
ongewijzigd.

### Plak 3 — diagnose bij te krappe grenzen

Een `StageInfeasible` in het automatische pad wordt als volgt onderscheiden:

1. Als minstens één familie begrensd is, wordt een nieuw soft model gebouwd met
   `UNCAPPED`. De harde voorkeuren blijven gelijk.
2. Is dit model infeasible, dan wint de bestaande
   `feasibility.diagnose`: de gebruikersmelding noemt geen Balansgrenzen.
3. Is het model haalbaar, dan worden uitsluitend de begrensde families voorzien
   van een overflow:

   `overflow = max(0, slack - (current - STRICTEST_LIMIT))`.

   De gewogen overflows (`SLACK_WEIGHTS[family] * overflow`) worden in aflopende
   volgorde leximin-geminimaliseerd. Elke bewezen positie wordt vastgezet; zodra
   een positie nul is, zijn de resterende posities ook nul.
4. Alleen positieve overflows worden teruggegeven als één gezamenlijke suggestie
   met `current` en `suggested = current + overflow`. De gevonden oplossing is
   daarmee tevens een haalbaarheidsbewijs voor de voorgestelde set.

De diagnose is geen UI-solvefase en verstuurt geen voortgangsevents. De normale
sorteernetwerk- en leximinlogica is gedeeld; de overflowvariabelen en hun
resultaat blijven onderdeel van de aparte diagnoseverantwoordelijkheid.

De gebruiker krijgt een Nederlandse melding met de zes UI-namen, de huidige en
voorgestelde grens en de verhoging. Bij meerdere wijzigingen vermeldt de tekst
dat ze bij elkaar horen en dat een andere combinatie ook mogelijk kan zijn.

## Prestatiemeting plak 3

De meting is uitgevoerd op synthetische data die in de tests is vastgelegd of
direct in het tijdelijke benchmarkscript is opgebouwd. Er is geen data uit
`instance/storage` gebruikt. De CP-SAT-workerinstelling was 8; tijden zijn
wall-clock seconden per procesrun.

| scenario | variant | modelbouw | vloer | overflow-stages | totaal | status |
|---|---|---:|---:|---:|---:|---|
| 5 leerlingen / 3 groepen / clique-cap 1 | leximin | 0,0035 | 0,0121 | 0,0170 (M0) | 0,0328 | OPTIMAL |
| idem | gewogen som | — | — | 0,0050 (één objective) | 0,0207 | OPTIMAL |
| 5 leerlingen / 2 groepen / clique + sekse-cap | leximin | 0,0025 | 0,0040 | 0,0081 + 0,0067 | 0,0215 | OPTIMAL |
| idem | gewogen som | — | — | 0,0074 (één objective) | 0,0161 | OPTIMAL |
| 24 leerlingen / 4 groepen / meerdere krappe caps | leximin, run 1 | 0,0042 | 0,0121 | 0,0186 + 0,0254 + 0,0174 | 0,0780 | OPTIMAL |
| idem | leximin, run 2 | 0,0040 | 0,0170 | 0,0265 + 0,0178 + 0,0168 | 0,0825 | OPTIMAL |
| idem | leximin, run 3 | 0,0042 | 0,0116 | 0,0182 + 0,0171 + 0,0152 | 0,0668 | OPTIMAL |
| idem | gewogen som, run 1 | — | — | 0,0164 (één objective) | 0,0323 | OPTIMAL |
| idem | gewogen som, run 2 | — | — | 0,0173 (één objective) | 0,0321 | OPTIMAL |
| idem | gewogen som, run 3 | — | — | 0,0159 (één objective) | 0,0342 | OPTIMAL |
| 2 leerlingen, uitgesloten van alle 3 groepen | leximin | 0,0029 | 0,0002 | — | 0,0031 | INFEASIBLE in floor |
| idem | gewogen sombaseline | — | — | — | 0,0041 | INFEASIBLE in floor |

De leximinvariant bewijst op deze fixtures steeds zijn eigen lexicografische
optimum in 2, 3 of 4 solverstages inclusief de vloer; de onbegrensd-infeasible
variant stopt in de eerste stage en levert `None`. De tijdelijke gewogen-som-
baseline was op alle gemeten synthetische gevallen sneller. Beide varianten
eindigden hier binnen praktisch bruikbare tijd, maar de fixtures zijn te klein
om daaruit een schaal- of SLA-conclusie te trekken. Het moeilijke geval liet
wel run-to-run-variatie zien (0,0668–0,0825 s), daarom is het driemaal gemeten.

De twee varianten gaven in deze meting dezelfde concrete suggesties. Dat is geen
productie-eis: leximin bepaalt de gewogen overflowvector, niet welke concrete
toewijzing of familie-identificatie bij een gelijkwaardige optimumrepresentatie
wordt gekozen. Tests controleren daarom de gezamenlijke noodzakelijke set en
haalbaarheid, niet een toevallige solvertoewijzing.

## Buiten scope

- wijzigingen aan `solve_with_fixed_balance` of `GroupBalance`;
- minima voor Balansgrenzen;
- nieuwe progressiefases, ETA-logica, timeouts of afbreekknoppen;
- tooltip-, styling- en overige polish voor een latere plak.
