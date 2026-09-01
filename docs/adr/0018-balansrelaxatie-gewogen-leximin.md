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

De balansfase minimaliseert de zes gewogen slacks in aflopende volgorde,
van groot naar klein: eerst de grootste `w_f · slack_f` zo laag
mogelijk (bewezen) en vastgepind, dan de op-één-na-grootste ("hooguit één
familie mag erboven uitsteken"), enzovoort — tot de resterende slacks nul zijn.
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
  dezelfde gesorteerde gewogen slacks als de oude som (200, 200, 100, 100,
  98, 0; piek 200 in beide). Het verschil ontstaat pas in wat er gepind
  wordt: het oude sombudget (`som(gewicht·slack) + 100·max_slack ≤ 898`) laat
  de tevredenheidsfase daarbinnen vrij herverdelen, en die eindigt gestapeld
  op 200, 200, 200, 49, 49, 0 — een derde familie naar de piek, som nog steeds
  precies 898. De pin op de gesorteerde gewogen slacks houdt elk niveau `M_k`
  vast, dus de eindbalans
  blijft 200, 200, 100, 100, 98, 0: leximin-strikt beter (positie 3: 100 tegen
  200). Prijs: totale tevredenheid 33,614 → 32,571 (−3,1%), 15 van de 43
  leerlingen verschuiven; de laagst-bedeelde leerling blijft exact gelijk
  (0,516129), de op-één-na-laagste zakt van 0,5333 naar 0,5231.
- **Bewezen optimaal blijft de norm.** Op het adversariële scenario bewijzen
  alle leximin-stages (628s totaal), gelijk aan strikte per-familie-lex maar
  met gespreide gewogen slacks: exact de oplossing die de kwadratische som vond
  maar nooit kon certificeren.

## Exacte formulering en overdracht tussen de fasen

De normatieve keuze hierboven verandert niet, maar de eerste implementatie
bleek onnodig veel balance-optimization-only variabelen en constraints mee te
slepen. De productieformulering gebruikt
daarom drie exact equivalente versterkingen:

- **Eén groepsindex per leerling.** De one-hot `in_group`-booleans blijven
  bestaan voor alle balanstellingen. Daarnaast krijgt iedere leerling één
  gehele groepsindex, aan dezelfde one-hot keuze gekoppeld. "Leerling A zit bij
  leerling B" is daardoor één gereïficeerde gelijkheid van twee groepsindices,
  hergebruikt voor wederzijdse wensen. Voorheen bouwde iedere wens per
  doelgroep een `BoolAnd`, een `BoolOr` en daarna een `max`; op het 72-leerlingen-
  scenario daalt het model van 1.741 naar 905 variabelen en verdwijnen 908
  `BoolAnd`- en 908 `BoolOr`-constraints. De twee representaties zijn equivalent:
  `AddExactlyOne` kiest precies één one-hot-boolean en die ware boolean fixeert
  de groepsindex op dezelfde groep.
- **Een echt sorteernetwerk voor zes gewogen slacks.** Vijftien vaste
  compare-swaps (`max`/`min`) materialiseren de volledige aflopende reeks
  gesorteerde gewogen slacks
  één keer. Iedere leximin-stage minimaliseert daarna rechtstreeks één uitgang
  van dat netwerk. De oude `exceed`-constructie maakte per rang zes booleans die
  alleen uitdrukten dat hoogstens `k` uitzonderingen boven een variabele grens
  mochten liggen; die cardinaliteitsrelaxatie was correct, maar propageerde zwakker
  tussen opeenvolgende rangen.
- **Een schoon tevredenheidsmodel met een exacte tabel voor gesorteerde gewogen
  slacks.** Na het bewijs
  van de balans wordt het model opnieuw opgebouwd. Alleen de bewezen
  relaxatievloer en de volledige gesorteerde gewogen slacks gaan mee; alle
  balance-optimization-only sorteervariabelen, doelstellingen en constraints
  verdwijnen. De slacks worden niet
  vastgezet op de toevallige familie→slack-toewijzing van de laatste
  balansoplossing. In plaats daarvan bevat `AddAllowedAssignments` alle geldige
  permutaties van deze slacks die, gegeven de familiegewichten en
  Balansgrenzen, echte slacktuples vormen. Zo blijven alle oplossingen met
  exact dezelfde leximin-slacks beschikbaar voor tevredenheidsoptimalisatie,
  terwijl de pin sterk en compact propageert.

## Dezelfde norm voor de cap-infeasibilitydiagnose

De beslissing geldt ook wanneer ingestelde Balansgrenzen de eerste
relaxatievloer infeasible maken. De diagnose bouwt dan een nieuw model met
`UNCAPPED` en dezelfde harde voorkeuren. Een infeasible eerste stage betekent
dat de bestaande voorkeurendiagnose wint; de melding noemt de grenzen dan niet.

Is het onbeperkte model wel haalbaar, dan worden alleen begrensde families
voorzien van `max(0, slack - cap_slack)`. De gewogen overflows worden met
hetzelfde sorteernetwerk en dezelfde gewogen leximin-norm geminimaliseerd als de
normale balansslacks. Dit levert een gezamenlijke, bewezen haalbare verruiming
op; een toevallige familie-toewijzing uit één solveroplossing wordt niet als
extra eis gepind.

### Meting van de diagnose

Op synthetische fixtures duurde de leximin-diagnose 0,0328 s voor de kleine
clique-case, 0,0215 s voor twee positieve overflows en 0,0668–0,0825 s voor
het moeilijke 24-leerlingen-geval (drie runs). De eerste stage van het
synthetische onbeperkt-infeasible geval stopte in 0,0031 s met `INFEASIBLE`.
De leximinvarianten hadden respectievelijk 2, 3 en 4 stages inclusief de
vloer; het onbeperkt-infeasible geval had alleen de vloerstatus.

De tijdelijke gewogen-sombaseline gaf op dezelfde fixtures dezelfde concrete
suggesties en was sneller: 0,0207 s, 0,0161 s en 0,0321–0,0342 s. Dat is geen
reden om de productienorm terug te draaien: de som optimaliseert een ander
criterium en de leximinvariant levert de gevraagde bewezen gewogen
overflowvector. Beide varianten waren op deze kleine synthetische gevallen
praktisch bruikbaar; de meting ondersteunt geen schaalgarantie. Gelijkwaardige
leximinoptima kunnen bovendien meer dan één familie- of toewijzingsuitkomst
toestaan, zodat de diagnose één bewezen mogelijkheid toont.

## Prestatiemeting na deze formulering

Op het opgeslagen adversariële `testschool/herdoor`-scenario (72 leerlingen,
vier doelgroepen, grenzen 3/4/3/4/6/4) zijn twee volledige productieruns gedaan
met 8 workers en zonder tijdslimiet. Beide bewezen exact dezelfde gesorteerde
gewogen slacks
`(400, 300, 200, 200, 49, 0)` en leverden exact dezelfde verdeling van
leerlingtevredenheid op.

| meting | vloer | balans | tevredenheid | totaal |
|---|---:|---:|---:|---:|
| oude geregistreerde formulering | 28,0s | 3.505,2s | 2.971,4s | 6.504,6s (108,4 min) |
| nieuwe formulering, run 1 | 3,5s | 554,6s | 480,0s | 1.038,1s (17,3 min) |
| nieuwe formulering, run 2 | 16,7s | 751,9s | 569,4s | 1.338,1s (22,3 min) |

Dezelfde benchmark op de opgeslagen gewone processen met 35 leerlingen laat
zien dat de lange staart niet bij ieder pad hoort:

| verdeelmodus | runs | vloer | balans | tevredenheid | totaal |
|---|---:|---:|---:|---:|---:|
| Doorzetten | 3 | 0,12–0,20s | 0,90–0,93s | 2,37–2,75s | 3,45–3,93s |
| Herindelen met dezelfde groepen | 3 | 0,05–0,07s | 0,31–0,35s | 0,67s | 1,07–1,10s |

Dit zijn representatieve opgeslagen scenario's, geen schaalgarantie. Ze
rechtvaardigen wel een asymmetrische UX: geen algemene waarschuwing bij het
veelgebruikte Doorzetten of bij gewoon Herindelen, maar hooguit een rustige
waarschuwing bij grotere of sterk begrensde gevallen van Herindelen met
doorzetten.

De wijziging is daarmee ongeveer vijf- tot zesmaal sneller en maakt het geval
weer praktisch hanteerbaar, maar haalt de gewenste 10–15 minuten end-to-end
niet betrouwbaar. Die grens kan dus geen SLA zijn zolang iedere CP-SAT-stage
tot een bewijs moet doorrekenen. De bestaande Tussenstand blijft daarom
belangrijk tijdens de bewijsstaart; hoe een gebruiker daarmee verder kan gaan
blijft een afzonderlijke productbeslissing.

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
  de gespreide leximin-slacks zijn de gewenste semantiek.
- **Leximin voor de gesorteerde gewogen slacks, maar pinnen via een sombudget
  in plaats daarvan** (variant B) — afgewezen ten gunste van de pin op de
  gesorteerde gewogen slacks (variant A).
  A is het zuivere, enkelvoudige criterium ("minimale balansafwijking")
  zonder max-hack en zonder tweede optimalisatiefase over de balans zelf;
  conceptuele zuiverheid en robuustheid wegen zwaarder dan de winst die B in
  de middenband zou geven. Die winst valt namelijk niet op de
  slechtst-bedeelde leerling — die is in A en B identiek beschermd via
  lexmaxmin — maar in de middenband: B zou circa 6 leerlingen naar ≥⅔
  tevredenheid tillen die A daaronder laat. De ~3% die A daarvoor kost, is dus
  een bewust geaccepteerde prijs. Het oude gedrag pinde net als B een
  sombudget (zij het op de som + max-hack, zonder ooit de gesorteerde gewogen
  slacks te berekenen); de −3,1% die op de FULL-instantie gemeten is bij het
  overstappen naar A, maakt precies deze afweging zichtbaar.
- **Handmatige solver-hulp** (value-precedence symmetry breaking, solution
  hints van stage naar stage) — afgewezen: gemeten contraproductief (+40%
  resp. +57%, gecombineerd zelfs verlies van het bewijs); CP-SAT's eigen
  presolve/symmetriedetectie wint van handmatige sturing.
- **16 in plaats van 8 workers** — afgewezen na meting op beide hoofdfasen.
  Meer workers betekent in CP-SAT een ander portfolio van zoekstrategieën, niet
  simpelweg tweemaal zoveel van dezelfde zoekactie. In de balansprefix werden
  `M₀` en `M₁` samen 196s in plaats van 99s; in de tevredenheidsfase werd het
  eerste plateau 117s in plaats van 52s. De extra cores maakten het bewijs dus
  consequent trager.
- **Een minimaal 12-comparatornetwerk in plaats van het insertion-netwerk met
  15 comparators** — afgewezen. Het kleinere netwerk was exact equivalent en
  gebruikte zes hulpvariabelen minder, maar alle eerste vier stages voor de
  gesorteerde gewogen slacks
  werden trager (`16/104/74/196s` tegen `15/85/64/182s`). Minder variabelen
  bleek hier niet hetzelfde als betere propagatie; de topologie van het
  insertion-netwerk sloot beter aan op CP-SAT's zoekportfolio.
- **Discrete domeinen voor de sorteernetwerkuitgangen** — afgewezen. Omdat een
  gewogen slack alleen een veelvoud van 49 of 100 kan zijn, leek een exact
  domein zonder tussenwaarden sterker dan het interval `0..upper`. In de
  praktijk maakte het de lineaire relaxatie en domeinverwerking duurder:
  alleen `M₀` liep op van circa 15s naar 72s.
- **Bewezen waarden van de gesorteerde gewogen slacks als gelijkheid pinnen** —
  afgewezen na een volledige balansmeting. `Mₖ == optimum` is logisch equivalent
  aan de gebruikte
  `Mₖ <= optimum`, omdat de ontbrekende ondergrens zojuist bewezen is. De
  expliciete gelijkheden veranderden presolve en branching echter ongunstig:
  de balansfase werd 618s in plaats van 555s; `M₄` alleen 242s in plaats van
  197s.
- **Na ieder tevredenheidsplateau opnieuw een schoon model bouwen** —
  afgewezen. Dezelfde techniek helpt sterk op de natuurlijke grens tussen
  balans en tevredenheid, maar binnen lexmaxmin moeten de bewezen
  boven-drempel-aantallen toch opnieuw worden gemodelleerd. De zwaarste telling
  werd 194s in plaats van 163s (het hele niveau 222s in plaats van 187s).
  Ook alleen de balance-optimization-only minimumvariabele vastpinnen had geen
  betekenisvolle winst: 483s tegen 480s voor de volledige tevredenheidsfase.
- **Tijds- of gap-begrenzing met gedegradeerd resultaat** — buiten deze
  wijziging gehouden: een zwaar geval met bindende grenzen rekent door (met
  voortgangspagina en Tussenstand); alleen bewezen INFEASIBLE geeft een
  fout. Een stopknop op de bewezen checkpoints van de toren is een mogelijk
  vervolg, geen onderdeel van deze beslissing.

## Consequenties

- De balansfase bestaat uit maximaal zes korte stages (stopt zodra de rest
  van de gesorteerde gewogen slacks nul is); de voortgang toont dus meer,
  kleinere stappen.
- `MAX_SLACK_WEIGHT` en de max-slack-variabele verdwijnen; `SLACK_WEIGHTS`
  verandert van objective-coëfficiënten in de piek-maatstaf van de leximin.
- Bij bindende grenzen op pathologische invoer blijft de volledige solve duur:
  op het stress-scenario 17,3–22,3 minuten totaal, waarvan 9,2–12,5 minuten
  balans. Dat is de geaccepteerde prijs van harde grenzen op de
  haalbaarheidsrand, bewust zonder tijdslimiet gelaten.
- Integratietests die exacte uitkomsten pinnen worden her-pind: ook het
  gezonde referentiescenario verschuift (FULL-instantie: totale tevredenheid
  33.614 → 32.571, 15 van de 43 leerlingen), niet doordat de balansstage een
  andere gesorteerde gewogen slacks vindt maar doordat de pin daarop de
  tevredenheidsfase minder
  ruimte laat dan het oude sombudget.
- Bij bindende Balansgrenzen heeft de rekentijd een hoge run-tot-run-variantie,
  terwijl de uitkomst deterministisch blijft: de twee volledige metingen
  bewezen dezelfde gesorteerde gewogen slacks en dezelfde satisfactieverdeling,
  maar verschilden
  vijf minuten in totaaltijd (17,3 tegen 22,3 minuten). Dat is inherent aan
  "doorrekenen tot bewijs" zonder tijdslimiet (zie Overwogen alternatieven):
  elke stage levert hetzelfde bewezen optimum, alleen de tijd om daar te komen
  wisselt.
