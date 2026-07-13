---
status: accepted
---

# Instelbare balansgrenzen als harde bovengrens op de automatische relaxatie

Het automatische pad (`solve_within_minimal_relaxation`) relaxeert de
klassenbalans onbegrensd: het zoekt de minimale relaxatie waarbij iedere
leerling positieve Tevredenheid haalt (de Relaxatievloer, ADR-0015) en levert
daarbij een relaxatie-*budget* op; de lexmaxmin-fase mag dat hele budget daarna
gebruiken om leerlingen te clusteren als dat de tevredenheid verhoogt. Is het
minimale budget groot, dan kan dat leiden tot een geldige maar onbruikbare
indeling — bijv. iedereen in 2 van de 4 groepen — die de gebruiker nu niet kan
bijsturen.

## Beslissing: per balans-familie een instelbaar maximum (Balansgrens)

De school kan per proces, op een geavanceerd paneel op de processing-pagina,
een **maximum** zetten op elk van de zes balans-families (groepsgrootte per
jaarlaag/totaal, jongens-meisjes per jaarlaag/totaal, stamgroep-clique
totaal/per sekse). Technisch begrenst het maximum de bijbehorende
slack-variabele: `slack <= grens - STRICTEST_LIMIT`.

Kernkeuzes:

- **Hard, wint van de Relaxatievloer.** Kan de vloer alleen worden gehaald met
  een relaxatie die de grens overschrijdt, dan geldt de grens — niet de vloer.
  Sommige leerlingen kunnen daardoor onder de vloer (≤0) uitkomen; dat is een
  geldige, expliciet gecommuniceerde uitkomst, geen fout. Wie dat niet wil, zet
  de grens op **Onbeperkt** (het oude gedrag).
- **Standaard aan, met genereuze data-driven defaults.** Niet onbeperkt (dat
  liet juist de onbruikbare indeling ontstaan), maar ook niet krap: de defaults
  zijn zo ruim dat infeasibility zeldzaam blijft. `diff_total`/`gender_total` =
  `max(4, huidige spreiding)` (nooit strakker dan wat de bezetting al afdwingt),
  `diff_year`/`gender_year` = 3, clique/clique_sex = 2× de even-verdeel-bodem
  (⌈grootste stamgroep(-deel) / #groepen⌉). Berekend in de web-laag; de solver
  krijgt altijd concrete getallen.
- **Los van de vaste `GroupBalance`.** Dit is níet het bestaande
  vaste-balans-pad (`solve_with_fixed_balance`), dat élke familie op een vast
  getal pint. Een aparte `BalanceMaxima`-dataclass (zes `int | None`, `None` =
  Onbeperkt) begrenst enkel de bovenkant van het automatische pad; de minimale
  relaxatie en de weging eronder blijven intact.

## Infeasibility: één overflow-solve met exacte, haalbare tip

Een harde grens kan het probleem onoplosbaar maken. Bij infeasibility draait één
extra solve die de balans-slacks weer vrijlaat en per familie een *overflow*
boven de grens minimaliseert, gewogen met de bestaande `SLACK_WEIGHTS`:

- **Oplossing met overflow > 0** → de grenzen zijn de oorzaak; de gebruiker
  krijgt de gezamenlijke minimale loosening met exacte getallen te zien
  ("kan wél met: groepsgrootteverschil ≥ 6 én jongens/meisjes-verschil ≥ 5") en
  keert terug naar het paneel om aan te passen.
- **INFEASIBLE** → zelfs onbeperkt bestaat geen indeling; het ligt aan de harde
  voorkeuren, en de bestaande `feasibility.diagnose` (Niet-samen / Extra
  zekerheid / Niet-in) neemt het over. De grenzen worden dan niet genoemd.

## Overwogen alternatieven

- **Default = Onbeperkt (huidig gedrag), paneel puur opt-in** — afgewezen: dan
  blijft de onbruikbare indeling bestaan voor wie het paneel niet vindt, en kost
  bijsturen een extra, onnodige run.
- **Zachte defaults die bij infeasibility zichzelf minimaal oprekken** — mooi,
  maar vergt een extra solve en meer machinerie; afgewezen wegens complexiteit.
  In plaats daarvan: genereuze harde defaults (zeldzaam infeasible) + concrete
  hulp achteraf.
- **Losse per-familie minima in de tip** ("elke grens afzonderlijk minimaal X")
  — vergt een solve per familie; afgewezen ten gunste van één gezamenlijke
  minimale loosening.
- **Waarschuwen bij een grens krapper dan de default** — afgewezen: een krappe
  keuze kan juist de beste zijn, en past het niet, dan hoort de gebruiker het
  alsnog concreet via de tip.
- **Minima nu meenemen** — geparkeerd als kleine symmetrische vervolgstap
  (basislimiet per familie op `min` i.p.v. `STRICTEST_LIMIT`); de maxima zijn
  het concrete pijnpunt.

## Consequenties

- Het automatische pad krijgt standaard begrensde slacks; onaangeraakte
  processen kunnen dus een andere (strakkere) indeling geven dan vóór deze
  wijziging, en in zeldzame gevallen een tier-2-uitkomst (enkele leerlingen ≤0)
  of infeasibility waar het voorheen "gewoon werkte". Dat is beoogd gedrag.
- De processing-pagina krijgt een idle-toestand (samenvatting + paneel + Start);
  de solve start pas bij Start, niet meer bij binnenkomst. De live-voortgangs-UX
  (polling, ETA, tussenstand) blijft ongewijzigd en leeft enkel in de
  "draait"-tak.
- `feasibility.diagnose` gaat niet langer uit van "balans is altijd volledig
  soft": een grens is een nieuwe infeasibility-bron, afzonderlijk gediagnosticeerd.
