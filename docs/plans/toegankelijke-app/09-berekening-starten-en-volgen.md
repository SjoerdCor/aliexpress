# Pagina 9 — Groepsindeling berekenen en volgen

**Status:** implementatiegereed na gebruikersreview

## Doel

Een leerkracht of IB'er controleert vóór de start in één scan met welke gegevens ALI
Express gaat rekenen, kan desgewenst harde bovengrenzen aan groepsverschillen instellen en
start daarna bewust de berekening. Tijdens het rekenen blijft duidelijk wat ALI Express
doet, dat de gebruiker niets hoeft te doen en dat een getoonde tussenstand nog kan
veranderen.

Deze slice omvat de klaar-, reken-, fout- en opnieuw-rekenen-toestand van `/processing`.
Solvergedrag, balansbetekenis, opslagbetekenis, verdeelmodi en de resultaat- en
sociogrampagina blijven ongewijzigd.

## Huidige toestand en relevante stash-WIP

De huidige klaarstaat gebruikt de technische kop **Groepsindeling — klaar om te rekenen**,
een lange samenvattingsregel en de uitklapper **Geavanceerd: klassenbalans-grenzen**. De
zes grensvelden zijn alleen als `Per jaarlaag` of `Totaal` benoemd; hun uitleg staat in
niet-focusbare `title`-tooltips. De uitklapper erft bovendien een eigen scrollvak van 200
px. Tijdens het rekenen zijn de fasen inhoudelijk bruikbaar, maar woorden als
`optimaal haalbare groepsbalans` en `maximaliseren` zijn onnodig technisch.

De huidige foutafhandeling toont de opgeslagen foutmelding als een eigen permanent paneel
op de klaarstaat. De polling gebruikt `setInterval`; bij een traag statusantwoord kunnen
verzoeken en terminale redirects elkaar overlappen.

De meest recente stash bij het schrijven van dit plan is object `587fe776` met de
zichtbare omschrijving `remove processed page 8 WIP`, gebaseerd op commit `8953ca0`.
Controleer de object-id opnieuw en vertrouw niet blind op het veranderlijke nummer
`stash@{0}`. Pas of pop de stash niet als geheel.

Gericht bruikbaar uit die stash:

- de gestructureerde invoersamenvatting met namen van huidige groepen en groepen in deze indeling;
- de concrete koppeling van labels aan de zes balansvelden;
- de foutmelding op basis van de al opgeslagen `Run.message`;
- de processing-specifieke responsieve stijlen;
- de duidelijkere teksten voor tussenstand, opnieuw rekenen en sociogram;
- de ideeën voor route- en browsergedragstests.

Niet overnemen:

- `run_errors.py`, `run_error.json` en wijzigingen in `tasks.py` en `process_files.py`;
  die voegen opslag en taakcode toe voor één teruglink bij `too_strict_not_together`,
  terwijl die fout al op de vorige pagina wordt gevalideerd en niet uit de berekening
  komt;
- pagina-8-restanten in `validation_messages.py`, `test_not_together_browser.py` of het
  gemengde begin van `preferences-processing.css`;
- de gedeeltelijke JavaScript-herbouw van de al servergerenderde samenvatting;
- tests die complete introducties, alle zes volledige veldteksten of CSS-klassen zonder
  gedragswaarde vastzetten.

## Klaar om te rekenen

Gebruik als h1 **Groepsindeling berekenen**.

Toon daaronder als gewone tekst:

> Controleer hieronder de gegevens waarmee ALI Express rekent. Klopt alles? Dan zoekt ALI
> Express naar de best haalbare groepsindeling: evenwichtige groepen en iedereen zo
> tevreden mogelijk.

### Jouw invoer

Toon de samenvatting vóór de start open en als een semantische lijst. Zij bevat alleen
aantallen en namen, geen volledige voorkeuren of volledige spreidingsregels:

- **Leerlingen:** `{aantal}` te verdelen leerlingen, met de aantallen jongens en meisjes;
- **Huidige groepen:** een geneste lijst met volledige groepsnaam en leerlingaantal;
- **Huidige jaarlaag/jaarlagen:** alleen tonen wanneer jaarlaaggegevens aanwezig zijn;
- **Groepen in deze indeling (`{aantal}`):** een geneste lijst met alle volledige groepsnamen;
- **Voorkeuren:** `{aantal}` leerlingen met één of meer positieve of negatieve
  voorkeuren; tel een niet-in-groep-uitsluiting of alleen extra zekerheid niet als
  voorkeur;
- **Spreidingen:** `{aantal} toegevoegd` of **Geen spreidingen toegevoegd**.

Gebruik geen formulering als `voor X van Y ingevuld`: geen voorkeur opgeven kan een
bewuste, geldige keuze zijn. Laat lange namen omlopen en behoud de volgorde uit de invoer.
Tijdens het rekenen wordt de invoersamenvatting niet opnieuw getoond: bij korte runs zou
een tweede uitklapper onrust geven. De volledige samenvatting blijft één keer
servergerenderd in de klaarstaat; bouw haar niet nogmaals of gedeeltelijk op uit
`/status`.

## Geavanceerde balansgrenzen

Gebruik een standaard gesloten uitklapper **Geavanceerd: maximale verschillen tussen
groepen**. Na een mislukte berekening met bewaarde grenzen mag hij open starten. De uitleg
krijgt geen eigen scrollvak.

Algemene uitleg:

> ALI Express maakt de groepen zo evenwichtig mogelijk. Soms is meer verschil nodig om
> leerlingen beter aan hun voorkeuren te helpen. Hieronder stel je de uiterste grenzen
> in: de berekening gaat daar nooit overheen.
>
> De voorgestelde grenzen zijn ruim en hoeven meestal niet te worden aangepast. Verander
> ze alleen als je school hierover een harde afspraak heeft. Met te strenge grenzen kan
> een geldige groepsindeling onmogelijk worden.

Geef iedere balansfamilie naast haar legend een korte uitleg.

**Groepsgrootte**

> We proberen de leerlingen zo te verdelen dat de groepen allemaal even groot worden. Je
> kunt het maximale verschil begrenzen per jaarlaag en over de hele groep.

**Jongens en meisjes**

> We streven binnen iedere groep in deze indeling naar een zo evenwichtig mogelijke verdeling
> tussen jongens en meisjes. Je kunt het verschil begrenzen per jaarlaag en over de hele
> groep.

**Leerlingen uit dezelfde huidige groep**

> We verspreiden leerlingen uit dezelfde huidige groep over de groepen in deze indeling. Je kunt
> apart begrenzen hoeveel leerlingen in totaal, en hoeveel jongens of meisjes, samen in
> één groep in de nieuwe indeling komen.

Houd de twee veldlabels per familie daarna beknopt maar zelfstandig begrijpelijk, met het
onderscheid **per jaarlaag** en **over de hele groep**. Gebruik **Geen maximum** bij alle
zes selectievakjes. Koppel ieder label met `for`/`id` aan precies het juiste veld; verwijder
de info-iconen en `title`-tooltips. De bestaande veldnamen, minimumwaarde, integerstap,
standaardwaarden, parsing en `None`-betekenis van Geen maximum blijven intact.

Gebruik de navigatieacties:

- **← Terug naar Leerlingen spreiden**;
- **Groepsindeling berekenen →**;
- bij een bestaand resultaat opnieuw **Groepsindeling berekenen →**.

Bij opnieuw rekenen blijft de waarschuwing zichtbaar dat een nieuwe berekening de huidige
groepsindeling vervangt, met de bestaande downloadactie vóór de startknop.

## Tijdens het rekenen

Gebruik als h1 **Groepsindeling berekenen** en als introductie:

> ALI Express vergelijkt veel mogelijke groepsindelingen en verbetert de beste
> tussenstand stap voor stap. Je hoeft niets te doen. Je kunt deze pagina openlaten of
> later via Jouw groepsindelingen terugkomen.

Toon de drie bestaande solverfasen in gewone taal, zonder hun betekenis of volgorde te
veranderen:

1. **Bepalen hoeveel leerlingen ten minste één voorkeur kunnen krijgen**;
2. **De groepen zo evenwichtig mogelijk maken**;
3. **De tevredenheid van alle leerlingen verder verbeteren**.

De eerste tekst mag niet worden vervangen door een nietszeggende term als `goede basis`:
dat het hier om ten minste één voorkeur per leerling gaat, is wezenlijke uitleg voor de
gebruiker. De solversemantiek voor positieve en negatieve voorkeuren blijft uiteraard
ongewijzigd.

Gebruik voor een zichtbare tevredenheidsronde bijvoorbeeld:

> De laagste tevredenheid is nu 62%. Voor 34 leerlingen zoekt ALI Express nog verder.

Maak `leerling`/`leerlingen` grammaticaal correct. Behoud het concrete percentage; maak
niet meerdere identieke voortgangsregels zonder zichtbare ontwikkeling. Gebruik voor de
slotfase **De laatste verbeteringen worden doorgerekend…**.

Tijdteksten:

- vóór een dynamische schatting: **Meestal is de berekening binnen een minuut klaar. Bij
  grotere of ingewikkelde verdelingen kan het langer duren.**;
- dynamisch: **Naar verwachting nog ongeveer {tijd}. Deze schatting kan veranderen.**

Behoud de bestaande afgeronde seconden/minuten en de bestaande 45-seconden-gate. Verander
geen ETA-formule of voortgangsdata.

Noem de uitklapper **Voorlopige groepsindeling** met de subtekst **ALI Express verbetert
deze groepsindeling nog.** Behoud de bestaande groepskaarten en popover. Gebruik onder
**Tijdens het rekenen** de secundaire actie **Bekijk de voorkeuren in het sociogram ↗**.
De sociogrampagina zelf valt buiten scope.

## Fouttoestand

Toon de bestaande `Run.message` met de bestaande flash-weergave van de app. Gebruik geen
apart foutpaneel of tweede foutweergave op deze pagina. De normale klaarstaat blijft
beschikbaar, zodat de gebruiker de bewaarde grenzen kan controleren en opnieuw kan
starten. Balansgrenzen krijgen in deze flash dezelfde volledige, begrijpelijke labels als
de velden op de pagina; interne verkortingen zoals `Zelfde stamgroep totaal` zijn niet
zichtbaar voor de gebruiker.

Laat het normale formulier, de bewaarde grenzen, de terugactie en **Groepsindeling berekenen →**
beschikbaar. Voeg geen foutcode-sidecar, contextuele foutlink, sessiestaat of nieuwe route
toe. Laat de browser bij status `error` rechtstreeks teruggaan naar `/processing`; de
route toont de opgeslagen melding via de bestaande error-flash.

## JavaScript en toegankelijkheid in gewone taal

- De volledige samenvatting staat één keer in de klaarstaat-HTML die de server verstuurt.
  Laat JavaScript haar tijdens het rekenen niet nog eens gedeeltelijk overschrijven.
- Vraag de status opnieuw op nadat het vorige antwoord is verwerkt. Zo zijn nooit twee
  statusverzoeken tegelijk bezig en kan de pagina niet meerdere keren tegelijk naar het
  resultaat proberen te gaan.
- Bereken per statusantwoord eenmaal of de 45-seconden-gate open is en gebruik die uitkomst
  voor de extra voortgang en tussenstand.
- Verwijder de debugregel uit de browserconsole.
- Kondig alleen betekenisvolle fasewisselingen via een korte `aria-live="polite"`-status
  aan; voorkom dat iedere poll dezelfde tekst opnieuw voorleest.
- Stop de pulserende/draaiende animatie wanneer de gebruiker in het besturingssysteem
  minder beweging heeft ingesteld. De tekstuele status blijft dan zichtbaar.
- Vang een tijdelijk mislukt statusverzoek rustig op en probeer na het normale interval
  opnieuw; maak daarvoor geen nieuwe gebruikersstatus, route of opslag.

Behoud de bestaande onmiddellijke eerste poll, pollingintervalconfiguratie,
statusbetekenissen en automatische overgang naar `/result` na `done`. Splits de pagina
niet op in nieuwe routes en voeg geen productiereviewmodus of kunstmatige wachttijd toe.

## Bestandsscope

Toegestaan:

- `templates/processing.html`;
- nieuw `static/processing.css`, uitsluitend voor deze pagina;
- nieuw `static/processing.js`, uitsluitend voor deze pagina;
- strikt noodzakelijke presentatiecontext en het verwijderen van `/handle-error` in
  `src/aliexpress/web/routes/results.py`;
- uitsluitend de gebruikersgerichte ETA-tekstformattering in
  `src/aliexpress/web/progress_writer.py`;
- gerichte bestaande tests in `tests/test_results.py`,
  `tests/test_progress_writer.py`, `tests/test_wizard_distribution.py` en
  `tests/browser/test_distribution_browser.py`;
- zo nodig één gerichte processing-layouttest in
  `tests/browser/test_accessible_layout_browser.py`;
- dit plan en de statusregel in `docs/plans/toegankelijke-app/README.md`.

Niet toegestaan: `wizard.py`, `tasks.py`, `process_files.py`, `run_errors.py`,
`validation_messages.py`, pagina 8, resultaat- of sociogramtemplates, solvercode,
voortgangsdata, ETA-formules, datamodel, sessiebetekenis of opslagbetekenis.

## Gedragsacceptatie

- Een gewone GET in de klaarstaat is alleen-lezen en start geen `Run` of
  voortgangsbestand.
- De open samenvatting toont de afgesproken categorieën als lijst, met correcte aantallen,
  namen van huidige groepen en groepen in de nieuwe indeling en volledig omlopende lange
  namen. Zij suggereert niet dat een
  leerling zonder voorkeur onvolledig is.
- Tijdens pending/running ontbreekt de klaar-samenvatting en het startformulier; alle drie
  fasen blijven aanwezig terwijl hun toestand wordt bijgewerkt.
- De zes balansvelden hebben bruikbare toegankelijke namen; Geen maximum schakelt alleen
  het gekoppelde getal uit en herstelt de eerdere waarde bij terugschakelen.
- De 45-seconden-gate, dynamische tijdschatting, tevredenheidsrondes en voorlopige
  groepsindeling blijven volgens bestaand gedrag verschijnen.
- Er is maximaal één statusverzoek tegelijk. `done` veroorzaakt één overgang naar het
  resultaat; `error` één overgang terug naar de klaarstaat met een error-flash.
- De foutmelding verschijnt via de normale flash-weergave en behoudt invoer en grenzen.
- Een bestaand resultaat kan eerst worden gedownload en daarna bewust worden vervangen
  door een nieuwe berekening.
- Op laptopbreedte, 390 px en 320 CSS-px, bij 200% zoom en met lange namen ontstaat geen
  horizontale paginascroll of afgeknotte informatie. Alle bediening werkt met toetsenbord,
  focus is zichtbaar en de verminderde-bewegingsvoorkeur wordt gerespecteerd.

Tests controleren deze gedragingen via zichtbare rollen, namen, toestanden, routes en
opgeslagen gevolgen. Ze mogen enkele betekenisdragende labels gebruiken om elementen te
vinden, maar dupliceren geen volledige intro, alle statische familieteksten, template-HTML,
helperfuncties of CSS-implementatiedetails zonder duidelijke regressiewaarde.

## Verificatie en review

Voer na iedere wijziging de kleinste relevante test uit. Rond de kandidaat af met ten
minste:

```bash
uv run pytest tests/test_results.py tests/test_progress_writer.py \
  tests/test_wizard_distribution.py --no-cov -n 4 --dist load
uv run pytest tests/browser/test_distribution_browser.py \
  tests/browser/test_accessible_layout_browser.py -q --no-cov -n 4 --dist load
```

Controleer daarnaast in een echte browser de klaar-, reken-, fout- en opnieuw-rekenen-
toestand op laptopbreedte, 390 px en 320 px, met toetsenbord, 200% zoom, lange namen en
verminderde beweging.

Omdat een echte kleine berekening te snel kan zijn voor een rustige review, maakt de
implementerende agent zonder productiecode te veranderen vijf vaste browserbeelden door
in Playwright alleen het antwoord van `/status` te vervangen:

1. klaar om te rekenen;
2. eerste fase actief;
3. balansfase actief;
4. tevredenheidsfase met zichtbare voorlopige groepsindeling;
5. fouttoestand met zichtbare error-flash.

Bewaar de screenshots buiten de repository, bijvoorbeeld onder `/tmp`, en rapporteer hun
paden aan de eigenaar. De getoonde invoer gebruikt lange maar fictieve leerling- en
groepsnamen. Dit vertraagt de echte solver niet en voegt geen reviewroute of testdata aan
het product toe.

Rapporteer alle tests en browserbevindingen, maar maak geen commit. Vraag eerst de eigenaar
de uiteindelijke pagina en de vijf vaste beelden te beoordelen. Na expliciet akkoord kan
de slice worden gecommit. Audit daarna stashobject `587fe776`: laat het pas verwijderen
nadat ieder resterend relevant onderdeel aantoonbaar is geïmplementeerd of bewust is
afgewezen en alle oudere stashes intact blijven.
