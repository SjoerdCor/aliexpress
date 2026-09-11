# Implementatieplan — informatiehiërarchie homepage

**Status:** implementatiegereed; inhoud en scope goedgekeurd op 2026-09-11. Dit is een
vervolgverbetering op de bestaande homepage, geen nieuw totaalontwerp.

## Doel

De homepage brengt bezoekers sneller van het herkenbare probleem naar de startactie en
presenteert daarna in deze volgorde de opbrengst, toepassingssituaties, zichtbare voorbeelden
en inhoudelijke onderbouwing. De pagina blijft enthousiast en begrijpelijk; technische
solver- en wizarddetails horen niet op de homepage.

De wijziging lost daarnaast een onjuiste verwachting op: ALI Express herkent geen
zorggegevens. De gebruiker kan wel een spreiding instellen, waarna ALI Express de opgegeven
leerlingen automatisch binnen dat maximum over de groepen verdeelt.

## Besloten afbakening

- Wijzig uitsluitend de homepage en haar gerichte tests.
- Voeg geen stepper toe. Voordat een gebruiker een verdeelmodus kiest, bestaat er nog geen
  betrouwbare, concrete route; bovendien voegt een stappenoverzicht op deze pagina te weinig
  toe naast de galerij en de drie toepassingssituaties.
- Wijzig geen wizardstepper, procespagina, processingpagina, routes, solvergedrag of andere
  gebruikersflows.
- Gebruik op de homepage niet de technische term ‘bestemmingsgroep’.
- Gebruik ‘nieuwe groep’ hier evenmin als algemene term; schrijf eenvoudig ‘groep’ of omschrijf
  de concrete situatie.
- Noem geen ‘zorgbehoefte’. Leg uitsluitend bij de uitgebreide balansinformatie uit hoe een
  door de gebruiker ingestelde spreiding kan helpen bij leerlingen die extra ondersteuning
  nodig hebben.
- De twee bestaande homepage-uitklappers verdwijnen. De volledigheid van de invoer is een
  zichtbaar selling point; de solverfasen staan al op een relevanter moment op de
  processingpagina en worden binnen deze opdracht nergens naartoe verplaatst.

## Gewenste informatievolgorde

1. Introductie en startactie
2. **Wat ALI Express je oplevert** — drie opbrengstkaarten
3. **Voor iedere manier van indelen** — drie toepassingssituaties
4. **Bekijk wat je krijgt** — bestaande galerij
5. **Wat bedoelen we met een tevreden leerling?**
6. **Alles wat voor jouw groepen telt**
7. **Je ziet hoe de indeling tot stand komt**
8. **Pas je wensen aan en reken opnieuw**

## Goedgekeurde inhoud

### Opening

Behoud de bestaande merkregel, hoofdkop en CTA. Vervang de vier introalinea's door precies
deze twee alinea's:

> Een nieuwe groepsindeling maken is een complexe puzzel. Iedere leerling heeft eigen
> voorkeuren, terwijl één verschuiving gevolgen heeft voor andere leerlingen en groepen.
> Handmatig schuiven, controleren en overleggen kost daardoor al snel veel tijd en energie.

> ALI Express overziet de hele puzzel tegelijk. Het rekent alle ingevoerde voorkeuren,
> voorwaarden en balansdoelen door en vindt binnen enkele minuten een wiskundig optimale
> groepsindeling: de best mogelijke indeling binnen de uitgangspunten die je invoert.

Plaats direct daarna de bestaande actie **Start een groepsindeling →**. Op een normale
laptop moeten hoofdkop, beide alinea's en startactie zonder scrollen zichtbaar zijn.

### Opbrengstkaarten

Behoud de bestaande drie kaarten en hun iconen. Gebruik deze titels en teksten:

1. **Tijdwinst**

   > Van uren werk naar minuten rekenen: meer tijd voor onderwijs!

2. **Iedereen tevreden**

   > ALI Express overziet veel meer combinaties dan handmatig haalbaar is en vindt de
   > wiskundig beste indeling. Daarbij verbetert het eerst de uitkomst van de minst tevreden
   > leerlingen.

3. **Een sterke start voor elke groep**

   > ALI Express brengt groepsgrootte en samenstelling zo goed mogelijk in balans. Zo krijgt
   > iedere groep een sterke start.

De kaarten verkopen de opbrengst. Plaats hier geen opsommingen van invoervelden en geen
uitleg over spreidingen of extra ondersteuning.

### Drie toepassingssituaties

Voeg na de opbrengstkaarten een vaste sectie **Voor iedere manier van indelen** toe. Toon
drie semantische, informatieve items; dit zijn geen formulieropties en er komen dus geen
radio-inputs of andere schijninteracties in.

1. **Leerlingen gaan naar de volgende groepen**

   > Bijvoorbeeld: jaarlaag 5 wordt verdeeld over de bestaande groepen 6/7/8.

2. **Bestaande groepen worden opnieuw ingedeeld**

   > Bijvoorbeeld: leerlingen uit 6A, 6B en 6C worden opnieuw verdeeld over dezelfde
   > groepen.

3. **Leerlingen gaan verder én groepen worden opnieuw ingedeeld**

   > Bijvoorbeeld: jaarlaag 5 gaat naar 6/7/8, terwijl ook de leerlingen uit jaarlaag 6 en 7
   > opnieuw worden verdeeld.

De bestaande decoratieve inline-SVG's uit templates/processes.html mogen worden
hergebruikt, met homepage-eigen CSS-klassen en aria-hidden="true". Neem het label
**Meest gekozen** niet over.

### Galerij

Behoud de bestaande drie afbeeldingen, bijschriften, toegankelijkheidsattributen,
vorige/volgende-bediening, status, toetsenbordbediening en swipebediening. Verplaats de
galerij alleen naar haar afgesproken positie na de toepassingssituaties. Verander de
bestaande JavaScriptlogica niet zonder functionele noodzaak.

### Tevredenheid

Vervang de bestaande sectie door:

#### Wat bedoelen we met een tevreden leerling?

> Voor een leerling maakt één vervulde voorkeur al veel verschil. Daarom telt de eerste
> voorkeur in de berekening het zwaarst mee. Een tweede of derde vervulde voorkeur helpt ook,
> maar voegt steeds iets minder toe. Ook het belang dat aan iedere voorkeur is gegeven, telt
> mee.

> Bij het indelen kijkt ALI Express eerst naar de minst tevreden leerlingen en probeert hun
> uitkomst te verbeteren. Zo telt iedere leerling mee.

Deze uitleg is zichtbaar en niet inklapbaar: zij onderbouwt direct de kwaliteitsbelofte uit
de tweede opbrengstkaart.

### Volledigheid van de invoer

Vervang de uitklapper **Waar houden we rekening mee?** door een permanent zichtbare sectie:

#### Alles wat voor jouw groepen telt

> Een sterke groepsindeling vraagt om meer dan voorkeuren alleen. ALI Express brengt
> leerlingwensen, schoolafspraken en balans in één berekening samen.

Toon hieronder drie scanbare onderdelen:

**Voorkeuren van leerlingen**

> Met wie een leerling graag of liever niet in de groep komt, hoe belangrijk een voorkeur is
> en een eventuele voorkeur voor een bepaalde groep.

**Afspraken en extra zekerheid**

> Leerlingen die niet samen mogen komen, groepen waarin een leerling niet geplaatst mag
> worden en extra zekerheid voor sociaal kwetsbare leerlingen.

**Balans die past bij jouw school**

> Groepsgrootte, jongens en meisjes, jaarlagen en leerlingen uit huidige groepen worden zo
> evenwichtig mogelijk verdeeld. Daarnaast kun je een spreiding instellen. Zo laat je ALI
> Express bijvoorbeeld leerlingen die extra ondersteuning nodig hebben automatisch over de
> groepen verdelen, binnen het maximum dat jij kiest.

Verwijder daarnaast de volledige uitklapper **Hoe berekenen we een optimale indeling?**.
Verplaats die tekst niet binnen deze opdracht. De noodzakelijke nuance over optimaliteit
staat al in de goedgekeurde intro.

### Functionele voordelen

Behoud de bestaande inhoud van **Je ziet hoe de indeling tot stand komt** en **Pas je wensen
aan en reken opnieuw**. Plaats beide na **Alles wat voor jouw groepen telt**.

## Vormgeving en semantiek

- Werk voort op de bestaande homepage-vormtaal en paginabreedte; introduceer geen extern
  ontwerpframework, assets of dependencies.
- Gebruik correcte h1/h2/h3-hiërarchie en koppel zelfstandige secties waar passend met
  aria-labelledby.
- Houd langere lopende tekst ongeveer op de bestaande leesbreedte van 78ch.
- Toon de drie toepassingssituaties op brede schermen in drie kolommen en op mobiel onder
  elkaar.
- Toon de drie onderdelen van **Alles wat voor jouw groepen telt** eveneens scanbaar in drie
  kolommen en op mobiel onder elkaar. Geef deze verdieping minder visuele nadruk dan de
  primaire opbrengstkaarten.
- Behoud voldoende verticale scheiding tussen hoofdsecties en voorkom horizontale overflow.

## Bestandsscope

Verwacht:

- templates/home.html
- static/home.css
- tests/test_app.py
- tests/browser/test_home_browser.py

templates/processes.html is alleen een leesbron voor de modusiconen. Kopieer desgewenst de
SVG-markup; wijzig dat template niet.

De werkboom bevat al wijzigingen van de gebruiker, ook buiten deze pagina. Behoud die, raak
geen niet-gerelateerde bestanden aan en gebruik geen destructieve Git-opdrachten.

## Teststrategie: alleen gedrag automatiseren

Leg goedgekeurde marketingcopy, afwezigheid van woorden of exacte sectievolgorde niet vast
in geautomatiseerde tests. Zulke assertions zijn verandergevoelig en testen geen gedrag.
Verwijder of versmal de bestaande copytest in tests/test_app.py als die door deze wijziging
anders slechts met nieuwe tekstassertions in stand zou worden gehouden.

Automatiseer alleen blijvend gebruikersgedrag:

- GET / rendert succesvol voor de bestaande relevante authenticatietoestanden;
- de primaire CTA is zichtbaar, navigeert naar /processes en staat op 1366×768 zonder
  vooraf scrollen in het openingsscherm;
- de galerij behoudt drie navigeerbare slides en werkt met zichtbare knoppen, toetsenbord en
  swipe/touch;
- de pagina en haar driedelige layouts veroorzaken op 390 px geen horizontale overflow.

Controleer redactionele en visuele acceptatie handmatig in de browser:

- alle goedgekeurde teksten en de afgesproken sectievolgorde staan op de pagina;
- de homepage bevat geen stepper of solveruitklappers;
- de woorden ‘zorgbehoefte’ en ‘bestemmingsgroep’ komen niet op de homepage voor;
- de drie primaire opbrengsten vallen visueel eerder op dan de verdiepende driedeling;
- de laptop- en mobiele weergave zijn rustig, leesbaar en logisch scanbaar.

## Uitvoering en verificatie

1. Lees de actuele bestanden en diff voordat je wijzigt; integreer met bestaand werk.
2. Pas template en homepage-CSS als één samenhangende wijziging aan.
3. Werk bestaande tests alleen bij waar gedrag verandert of een oude copytest moet worden
   verwijderd/versmald. Voeg geen tekstsnapshot in een andere vorm terug.
4. Draai de gerichte snelle test:

   ~~~bash
   uv run pytest tests/test_app.py --no-cov
   ~~~

5. Draai de gerichte browsertest:

   ~~~bash
   uv run pytest tests/browser/test_home_browser.py -q --no-cov
   ~~~

6. Bekijk de uiteindelijke pagina handmatig op 1366×768 en 390×844, controleer de copy en
   sectievolgorde en bedien de galerij met muis en toetsenbord.
7. Rapporteer gewijzigde bestanden, uitgevoerde controles en eventuele relevante bestaande
   problemen. Maak geen commit vóór feedback van de eigenaar.

## Acceptatie

- De homepage volgt de goedgekeurde inhoud en volgorde uit dit plan.
- Kop, kernbelofte en CTA zijn samen in het eerste laptopscherm zichtbaar.
- De opbrengstkaarten zijn enthousiasmerend en bevatten geen technische detailuitleg.
- De drie toepassingssituaties maken duidelijk voor welke indelingen ALI Express geschikt is.
- Tevredenheid wordt eenmaal kort verkocht en eenmaal zichtbaar, begrijpelijk uitgelegd.
- De volledigheid van de invoer is zichtbaar in plaats van verborgen in een uitklapper.
- De tekst suggereert nergens automatische herkenning of verwerking van zorggegevens; alleen
  een door de gebruiker ingestelde spreiding wordt automatisch uitgevoerd.
- De galerij blijft volledig bedienbaar en de pagina werkt zonder horizontale overflow op
  mobiel.
- Geautomatiseerde tests borgen alleen gedrag, niet de gekozen marketingtekst.
