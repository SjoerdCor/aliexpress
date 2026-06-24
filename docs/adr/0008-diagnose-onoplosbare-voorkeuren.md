---
status: accepted
---

# Onoplosbare voorkeuren: per-familie isolatie-diagnose i.p.v. één geblende slack-solve

Bij het automatische pad (`groupbalance=None`) berekent `solve_within_minimal_relaxation` eerst
het minimale-relaxatiebudget `R*`. Die LP heeft de balans al zacht, dus onoplosbaarheid komt puur
voort uit de botsing van de harde constraints onderling: Niet-samen-regels en Extra zekerheid
(en, indirect, Niet-in-groep-uitsluitingen). Voorheen gooide dat een rauwe
`ValueError: Could not determine the minimal class-balance relaxation`. We vervangen dat door een
diagnose die per constraint-familie *in isolatie* test of het versoepelen ervan alléén de
verdeling weer mogelijk maakt, en dat vertaalt naar een vriendelijke Nederlandse `FeasibilityError`
met concreet advies in de vocabulaire van de leerkracht.

## Considered Options

**Eén geblende slack-solve** — alle harde constraints tegelijk zacht maken, de totale slack
minimaliseren en aflezen welke slack positief is. Verworpen: de onoplosbaarheid is inherent een
eigenschap van de *combinatie* (een Extra-zekerheid-eis kan onhaalbaar zijn juist omdat een
Niet-samen-regel de benodigde plaatsing blokkeert). Welke familie de "schuld" krijgt, is dan een
artefact van de gekozen slack-gewichten — schijnzekerheid, en precies de degeneratie-val die
`solve_within_minimal_relaxation` elders zorgvuldig vermijdt.

**Leave-one-out isolatie (gekozen)** — per familie één solve waarin die familie zacht-per-element
is en de andere familie hard blijft. De oplosbaarheid van zo'n solve is *discriminerend* (de andere
familie blijft immers hard), wat vier eerlijke gevallen geeft: alleen Extra zekerheid lost het op /
alleen Niet-samen / beide afzonderlijk (óf/óf) / geen van beide alleen (dan een gecombineerde solve,
of de terugval-melding voor het fundamentele/Niet-in-geval). De slack-waarden leveren meteen
concreet advies. Kost ~2-3 solves, uitsluitend in het foutpad.

## Consequences

- Het advies spreekt in discrete keuzes, niet in rauwe percentages: een onhaalbare Extra zekerheid
  op *belangrijkste voorkeur* wordt geadviseerd terug te zetten naar *minstens één voorkeur* of
  *geen eis* (zie [ADR-0003](0003-betekenisvolle-niveaus-ipv-vrije-getallen.md)); een te krappe
  Niet-samen-regel naar een hoger max-aantal-samen of een leerling uit de regel. De toon is
  beschrijvend ("deze keuzes botsen — versoepel er één"), niet voorschrijvend, omdat deze keuzes
  bewust pedagogisch zijn; Niet-in-groep-uitsluitingen worden níét als afstembare knop voorgesteld
  (het zijn feiten, geen keuzes).
- De diagnose-machinerie moet dezelfde regie als `R*` houden — balans zacht — anders diagnosticeert
  ze tegen een striktere balans dan de echte solve vereist.
- Een nieuwe `FeasibilityError`-code bakt zich in het contract tussen de diagnose-methode in
  `problemsolver.py` (bouwt en lost de isolatie-LP's op), de orkestratie in `main.py` en de
  Nederlandse teksten in `validation_messages.py` — net als het bestaande `calculate_feasibility` /
  `_check_feasibility`-paar voor de balans.
- Alleen solver-status *Infeasible* leidt tot deze diagnose; andere statussen (Unbounded/Undefined/
  solverfout) vallen terug op de generieke `internal_error`-melding, zodat een technische fout niet
  als voorkeuren-conflict wordt gepresenteerd.
