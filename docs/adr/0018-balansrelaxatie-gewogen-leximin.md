---
status: accepted
---

# Balansrelaxatie via gewogen leximin over de slack-families

Het automatische pad minimaliseerde de balansrelaxatie als één gewogen som
(`SLACK_WEIGHTS`, per-jaar ×100 en hele-groep ×49, plus een max-slack-term die
de relaxatie moest spreiden). Sinds een Balansgrens (ADR-0017) hard kan binden,
faalt die vorm precies waar de grenzen voor bedoeld zijn: op het adversariële
herindeel-scenario vindt CP-SAT binnen seconden goede oplossingen, maar kan het
minimum van géén enkele gesommeerde doelstelling meer *bewijzen* — gewogen
lineair niet (gap ~21% na 900s), kwadratisch niet (gap ~34% na 900s; de beste
oplossing verbeterde na de eerste minuten niets meer). De ondergrens blijft
structureel hangen omdat de relaxatie slack fractioneel tussen families kan
blijven herschuiven; tie-brekende of kwadratische gewichten veranderen daar
niets aan. Stages die één variabele minimaliseren bewijzen wél.

## Beslissing: leximin over de gewogen slacks, op beide paden

De balansfase minimaliseert het *gesorteerde profiel* van de zes gewogen
slacks, van groot naar klein: eerst de grootste `w_f · slack_f` zo laag
mogelijk (bewezen) en vastgepind, dan de op-één-na-grootste ("hooguit één
familie mag erboven uitsteken"), enzovoort — tot het profiel op nul eindigt.
Elke stage minimaliseert één variabele, de vorm die onder bindende grenzen
bewijsbaar blijkt.

- **De gewichten blijven, de max-slack-term verdwijnt.** `SLACK_WEIGHTS`
  (100/49) is een voorkeursratio — één punt scheefheid per jaarlaag telt als
  ~twee punten over de hele groep — en bepaalt in de leximin wat "de grootste
  piek" is. De spreiding die de max-slack-term moest afdwingen is in leximin
  het dragende principe zelf: pieken eerst omlaag, families kunnen niet vrij
  tegen elkaar wegstrepen zoals in een som. Het is bovendien hetzelfde
  normatieve principe dat de tevredenheidsfase al op leerlingen toepast.
- **Eén semantiek op beide paden.** Ook zonder bindende (of überhaupt
  gezette) Balansgrens geldt dezelfde leximin — geen apart gedrag voor het
  begrensde pad. Op het gezonde doorzetten-scenario is de uitkomst gemeten
  identiek aan de oude gewogen som (zelfde slackvector, zelfde tevredenheid
  per leerling, +0,2s).
- **Gezonde invoer is niet gratis — de kost zit in de pin, niet in de
  balanskeuze.** Op de FULL-integratie-instantie vindt de balansfase exact
  hetzelfde profiel als de oude som (gewogen gesorteerd 200, 200, 100, 100,
  98, 0; piek 200 in beide). Het verschil ontstaat pas in wat er gepind
  wordt: het oude sombudget (`som(gewicht·slack) + 100·max_slack ≤ 898`) laat
  de tevredenheidsfase daarbinnen vrij herverdelen, en die eindigt gestapeld
  op 200, 200, 200, 49, 49, 0 — een derde familie naar de piek, som nog steeds
  precies 898. De profiel-pin houdt elk niveau `M_k` vast, dus de eindbalans
  blijft 200, 200, 100, 100, 98, 0: leximin-strikt beter (positie 3: 100 tegen
  200). Prijs: totale tevredenheid 33,614 → 32,571 (−3,1%), 15 van de 43
  leerlingen verschuiven; de laagst-bedeelde leerling blijft exact gelijk
  (0,516129), de op-één-na-laagste zakt van 0,5333 naar 0,5231.
- **Bewezen optimaal blijft de norm.** Op het adversariële scenario bewijzen
  alle leximin-stages (628s totaal), gelijk aan strikte per-familie-lex maar
  met een gespreid profiel: exact de oplossing die de kwadratische som vond
  maar nooit kon certificeren.

## Overwogen alternatieven

- **Gewogen som behouden en meer rekentijd geven** — afgewezen: bij 900s
  (ruim boven het leximin-totaal) nog steeds geen bewijs én geen betere
  oplossing; het probleem is structureel (koppeling), niet een kwestie van
  geduld.
- **Kwadratische som** (spreiding via oplopende marginale kosten) — afgewezen:
  zelfde koppelingsprobleem, geen bewijs; leximin bereikt hier dezelfde
  oplossing mét bewijs.
- **Strikte per-familie-lex** (zes stages in vaste familievolgorde) — bewijst
  even goed en even snel, maar absolute prioriteit dumpt de relaxatie op de
  laagste families (clique tegen zijn grens om één punt hoger te winnen);
  het gespreide leximin-profiel is de gewenste semantiek.
- **Leximin voor het profiel, maar pinnen via een sombudget in plaats van het
  profiel** (variant B) — afgewezen ten gunste van de profiel-pin (variant A).
  A is het zuivere, enkelvoudige criterium ("minimale balansafwijking")
  zonder max-hack en zonder tweede optimalisatiefase over de balans zelf;
  conceptuele zuiverheid en robuustheid wegen zwaarder dan de winst die B in
  de middenband zou geven. Die winst valt namelijk niet op de
  slechtst-bedeelde leerling — die is in A en B identiek beschermd via
  lexmaxmin — maar in de middenband: B zou circa 6 leerlingen naar ≥⅔
  tevredenheid tillen die A daaronder laat. De ~3% die A daarvoor kost, is dus
  een bewust geaccepteerde prijs. Het oude gedrag pinde net als B een
  sombudget (zij het op de som + max-hack, zonder ooit een profiel te
  berekenen); de −3,1% die op de FULL-instantie gemeten is bij het overstappen
  naar A, maakt precies deze afweging zichtbaar.
- **Handmatige solver-hulp** (value-precedence symmetry breaking, solution
  hints van stage naar stage) — afgewezen: gemeten contraproductief (+40%
  resp. +57%, gecombineerd zelfs verlies van het bewijs); CP-SAT's eigen
  presolve/symmetriedetectie wint van handmatige sturing.
- **Tijds- of gap-begrenzing met gedegradeerd resultaat** — buiten deze
  wijziging gehouden: een zwaar geval met bindende grenzen rekent door (met
  voortgangspagina en Tussenstand); alleen bewezen INFEASIBLE geeft een
  fout. Een stopknop op de bewezen checkpoints van de toren is een mogelijk
  vervolg, geen onderdeel van deze beslissing.

## Consequenties

- De balansfase bestaat uit maximaal zes korte stages (stopt zodra de rest
  van het profiel nul is); de voortgang toont dus meer, kleinere stappen.
- `MAX_SLACK_WEIGHT` en de max-slack-variabele verdwijnen; `SLACK_WEIGHTS`
  verandert van objective-coëfficiënten in de piek-maatstaf van de leximin.
- Bij bindende grenzen op pathologische invoer blijft de volledige solve duur
  (orde ~10 minuten balansfase op het stress-scenario, plus de
  tevredenheidsfase); dat is de geaccepteerde prijs van harde grenzen op de
  haalbaarheidsrand, bewust zonder tijdslimiet gelaten.
- Integratietests die exacte uitkomsten pinnen worden her-pind: ook het
  gezonde referentiescenario verschuift (FULL-instantie: totale tevredenheid
  33.614 → 32.571, 15 van de 43 leerlingen), niet doordat de balansstage een
  ander profiel vindt maar doordat de profiel-pin de tevredenheidsfase minder
  ruimte laat dan het oude sombudget.
- Bij bindende Balansgrenzen heeft de rekentijd van de balansfase een hoge
  run-tot-run-variantie, terwijl de uitkomst deterministisch blijft: een
  herhaalde meting op hetzelfde adversariële scenario bewees in beide runs
  dezelfde profielwaarden (M₀=400, M₁=300, M₂=200), maar een stage die in de
  ene run na 192s bewees, had in een andere run na 600s zelfs nog geen
  ondergrens. Dat is inherent aan "doorrekenen tot bewijs" zonder tijdslimiet
  (zie Overwogen alternatieven): elke stage levert altijd hetzelfde bewezen
  optimum, alleen de tijd om daar te komen wisselt.
