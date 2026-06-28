---
status: accepted
---

# Onoplosbare voorkeuren: familie-niveau leave-one-out diagnose

Bij het automatische pad (`groupbalance=None`) berekent `solve_within_minimal_relaxation` eerst
het minimale-relaxatiebudget `R*`. Die LP heeft de balans al zacht, dus onoplosbaarheid komt puur
voort uit de botsing van de harde voorkeur-constraints onderling: Niet-samen-regels en Extra
zekerheid (minimale tevredenheid), en indirect Niet-in-groep-uitsluitingen. Voorheen gooide dat een
rauwe `ValueError: Could not determine the minimal class-balance relaxation`. We vervangen dat door
een diagnose op **familie-niveau**: via leave-one-out (relax één familie, houd de andere hard) wordt
bepaald welke familie *noodzakelijk/voldoende* is om de verdeling weer mogelijk te maken. Het
resultaat is een `FeasibilityError` met een vriendelijke Nederlandse melding die de leerkracht
vertelt wélke soort keuze te versoepelen — zonder een individuele leerling of regel aan te wijzen.

## Considered Options

**Eén geblende slack-solve** — alle harde constraints tegelijk zacht maken, de totale slack
minimaliseren en aflezen welke positief is. Verworpen: de onoplosbaarheid is een eigenschap van de
*combinatie*; welke constraint de "schuld" krijgt hangt af van de slack-gewichten — schijnzekerheid,
precies de degeneratie-val die `solve_within_minimal_relaxation` elders vermijdt.

**Familie-niveau leave-one-out (gekozen)** — twee feasibility-solves: "lost het versoepelen van
alléén de Extra zekerheid het op?" en idem voor Niet-samen (de andere familie blijft hard, balans
blijft zacht). De uitkomst is *robuust en niet-arbitrair*: ze gaat over de familie, niet over een
willekeurig lid. Dat geeft vijf gevallen — `min_satisfaction` / `not_together` (die familie alleen
volstaat), `either` (elk afzonderlijk volstaat), `both` (alleen samen), `fundamental` (ook samen niet
→ ligt aan "Niet in"). Een derde solve is alleen nodig in het `both`/`fundamental`-onderscheid.

## Consequences

- De melding spreekt op familie-niveau, **zonder leerling-/regelnamen**: bv. "De gevraagde extra
  zekerheid is te streng: verlaag de extra zekerheid een stap bij de leerlingen waar je die hebt
  ingesteld." Geen percentages, geen exacte max-getallen. Niet-in-groep-uitsluitingen worden níét als
  afstembare knop voorgesteld (het zijn feiten, geen keuzes); ze komen alleen voor in het
  `fundamental`-geval.
- De diagnose-machinerie houdt dezelfde regie als `R*` — balans zacht — anders diagnosticeert ze
  tegen een striktere balans dan de echte solve vereist.
- De orkestratie (de vijf gevallen) en `feasible_when_relaxed` leven in `feasibility.py`,
  naast de overige feasibility-redenering (relaxatiebudget, balans-check). Dat is het
  coherente huis voor analyses-op-het-model; zie ADR-0009 voor de architectuurkeuze.
  De twee harde constraint-methoden kregen een `make_soft`-optie (hard pad ongewijzigd)
  zodat de feasibility-check ze hergebruikt zonder duplicatie. `main.py` vertaalt de case
  naar een `FeasibilityError("infeasible_preferences")`; de Nederlandse teksten staan in
  `validation_messages.py` — net als het bestaande `calculate_feasibility` /
  `_check_feasibility`-paar voor de balans.
- Alleen solver-status *Infeasible* leidt tot deze diagnose; andere statussen (Unbounded/Undefined/
  solverfout) houden de bestaande `ValueError`, zodat een technische fout niet als
  voorkeuren-conflict wordt gepresenteerd.

## Toekomstige uitbreiding (geparkeerd, niet verworpen)

Een rijkere variant — *per-element diagnose met namen* — is bewust uitgesteld, niet afgewezen. Die
zou per familie zacht-per-element rekenen en de betrokken leerlingen/regels bij naam noemen met een
concreet streefniveau (bv. "zet Anna terug naar 'minstens één voorkeur'" of "verhoog deze regel naar
max. 3 samen"). Dat is specifieker en daarmee handelbaarder voor de leerkracht.

Twee redenen om het nu níét te doen: (1) het voegt fors meer code en bug-gevoelige Nederlandse tekst
toe (discrete-niveau-vertaling, getallen, naam-vertaling); (2) **degeneratie** — het minimum is
meestal niet uniek, dus een specifieke leerling noemen suggereert een unieke boosdoener die er niet
is. Een toekomstige implementatie moet die degeneratie eerlijk afhandelen: óf alleen *noodzakelijke*
elementen noemen (die in élke oplossing moeten wijken), óf expliciet als "één manier — er zijn
mogelijk andere oplossingen" formuleren. De huidige familie-niveau-diagnose is hiervoor de robuuste
basis waarop dit later kan voortbouwen.
