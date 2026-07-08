---
status: accepted
---

# Relaxatievloer: strikt positieve tevredenheid i.p.v. minstens één wens

De adaptieve relaxatie breekt de klassenbalans alleen zover als nodig om een
minimale vloer van tevredenheid te halen, en houdt de balans daarna zo strak
mogelijk. De eerste lexicografische stap van `solve_within_minimal_relaxation`
bepaalt die vloer: hij minimaliseert het aantal leerlingen dat de vloer niet
haalt en fixeert dat aantal, waarna stap 2 de balansrelaxatie minimaliseert.

Tot nu toe was die vloer *"minstens één positieve wens gehonoreerd"* (de
`unmet`-literal, alleen voor leerlingen mét een positieve wens). Sinds
ADR-0014 kan tevredenheid negatief zijn, en toen brak die formulering: ze meet
de verkeerde grootheid.

## Beslissing: de vloer is strikt positieve tevredenheid

De vloer wordt *"iedere leerling heeft strikt positieve tevredenheid"* —
uniform over álle leerlingen, uitgedrukt op de `satisfaction`-variabele:
`satisfaction[leerling] <= 0` telt als onder de vloer, en stap 1 minimaliseert
dat aantal.

Dat verenigt de drie leerlingtypes met één criterium:

- **Puur-positief**: `> 0` ⟺ minstens één wens gehonoreerd. Identiek aan het
  oude gedrag — backward-compatible.
- **Gemengd** (positieve + vermij-wensen): `> 0` ⟺ de netto gewogen som is
  positief. Strenger dan vroeger: een gehonoreerde wens die door een geschonden
  vermij-wens wordt uitgewist telt niet meer als "geregeld".
- **Alleen-vermij**: tevredenheid is 1.0 (niets geschonden) of negatief, dus
  `> 0` ⟺ geen vermij-wens geschonden. Nieuw: deze leerlingen hadden geen
  `unmet`-literal en werden door stap 1 volledig genegeerd; nu kan de balans
  ook gebroken worden om hén positief te houden.

## Waarom strikt positief (`> 0`), niet non-negatief (`>= 0`)

Een puur-positieve leerling zonder gehonoreerde wens heeft tevredenheid exact
0. Onder `>= 0` zou die als "geregeld" tellen, waarmee de kern-eigenschap
"iedereen krijgt minstens één wens" zou vervallen — een regressie. Strikt
positief behoudt die eigenschap precies (0 is niet `> 0`).

## Overwogen alternatieven

- **Non-negatieve vloer (`>= 0`)** — afgewezen om de regressie hierboven.
- **Severity-gewogen stap 1** (diep-negatieve leerlingen zwaarder dan
  net-nul) — afgewezen. Stap 1 beantwoordt bewust één scherpe vraag ("hoeveel
  leerlingen kunnen de vloer niet halen") en fixeert dat aantal; de zwaarte
  (de laagste tevredenheid optillen) is juist de taak van de latere
  lexmaxmin-fase. Zwaarte in stap 1 zou die verantwoordelijkheid dupliceren.
- **Criterium op de rúwe gewogen som i.p.v. de satisfaction-variabele** —
  afgewezen: bij gewogen som 0 is een alleen-vermij-leerling positief (1.0)
  maar een positieve-wens-leerling nul. Alleen de satisfaction-variabele, die
  de sprong naar 1.0 codeert, verenigt de drie types.

## Consequenties

- Meer relaxatie mogelijk dan vroeger: de klassenbalans wordt nu ook gebroken
  om geschonden vermij-wensen te voorkomen, niet alleen om positieve wensen te
  vervullen. Stap 2 houdt die relaxatie minimaal.
- De feasibility-diagnose is onaangetast: stap 1 wordt alleen INFEASIBLE als
  het onderliggende harde model dat is, wat niet van de objective afhangt.
- Integratietests die exacte per-leerling-waarden vastpinnen kunnen verschuiven
  waar gemengde of alleen-vermij-leerlingen voorkomen. Net als bij ADR-0014 is
  dat beoogd gedrag, geen regressie; opnieuw pinnen ná begrip van het waarom.
- Het afrondingsrandgeval (een minuscuul positieve netto die naar de
  geschaalde 0 afrondt) wordt met realistische gewichten niet bereikt en
  empirisch geverifieerd, niet omheen gearchitecteerd.
