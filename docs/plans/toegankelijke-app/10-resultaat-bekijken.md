# Pagina 10 — Resultaat bekijken

**Status:** implementatiegereed na gebruikersreview

## Doel

Een nieuwe, niet-technische leerkracht of IB'er kan de gevonden groepsindeling zelfstandig
beoordelen: eerst de groepen zelf, daarna de omvang en plaats van zichtbare
groepsverschillen en vervolgens, indien gewenst, de uitkomst per leerling en de herkomst
van leerlingen. De gebruiker kan de groepsindeling met collega's bespreken, krijgt concrete
routes om invoer bij te sturen en kan een goed bevonden resultaat als Excel-bestand
downloaden en afronden.

Deze slice omvat alleen `/result` en het foutpad van de bestaande Excel-download. De
sociogrampagina, de afrondingspagina, de invoerpagina's, solvergedrag, gewichten,
verdeelmodi en opslagbetekenis blijven ongewijzigd.

## Uitgangspunt en gerealiseerd ontwerp

De resultaatpagina gebruikt de bestaande `groepsindeling_view.json` als bron voor de
groepskaarten en leidt daar tijdelijk de drie native analyses uit af. De oude brede
klassenoverzichtweergave, pandas-tabs en prominente sociogramactie zijn vervangen door
semantische HTML, rustige uitklappers en een duidelijke beoordelings- en bijstuurroute.

De primaire balanssamenvatting is altijd volledig zichtbaar binnen de beschikbare breedte.
De twee detailtabellen mogen bij een dynamisch aantal groepen lokaal horizontaal scrollen;
de pagina zelf loopt bij 320 CSS-px niet horizontaal over.

## Informatievolgorde en teksten

Gebruik als h1:

> Je groepsindeling is klaar!

Toon daaronder:

> ALI Express heeft op basis van je invoer de best haalbare groepsindeling gemaakt.
> Bekijk de groepen en bespreek met je collega's of deze indeling ook in de praktijk goed
> werkt.

De uitleg dat je op een leerling kunt klikken staat in de standaard gesloten uitleg bij de
groepskaarten, niet in de algemene introductie.

De inhoudsvolgorde is:

1. een standaard gesloten uitleg bij de groepskaarten;
2. de groepskaarten;
3. de samenvatting **Hoe evenwichtig zijn de groepen?**;
4. een gesloten uitklapper met het volledige leerlingoverzicht;
5. een gesloten uitklapper met de herkomstmatrix;
6. een rustige tekstlink naar het sociogram;
7. de vraag **Ben je tevreden met deze groepsindeling?**, met bijstuurhulp, Excel-download en
   afrondingsactie.

De groepskaarten blijven de kern en staan altijd open. Secundaire analyse begint gesloten
om de lange pagina behapbaar te houden.

## Groepskaarten

Behoud de bestaande gestructureerde groepskaarten en leerlingdetails. Gebruik de huidige
groep als term; vervang op deze pagina geen tekst door *oude groep* of *klas*.

Plaats boven de kaarten één standaard gesloten uitklapper **Uitleg bij de groepskaarten**.
Neem daarin ook de zin op dat alleen nieuw ingedeelde leerlingen met naam op de kaarten
staan, terwijl reeds aanwezige leerlingen meetellen in totalen en balans. Voeg daarnaast
de bestaande legenda, de uitleg over de afkorting van de huidige groep, voorkeurstatussen,
belangtekens, badges voor extra zekerheid, de tevredenheidsscore en de betekenis van
`—` toe. De uitleg dat je op een leerling kunt klikken staat in dezelfde uitklapper.
Toon de aantallen nieuw ingedeelde en reeds aanwezige leerlingen alleen op basis van het
tijdelijke paginaviewmodel; sla ze niet apart op.

Maak de kaarten op de resultaatpagina passend bij 320 px. Op smalle schermen mogen de
kolommen Jongens en Meisjes onder elkaar staan en opent de bestaande leerlingdetailkaart
onder het leerlingkaartje in plaats van buiten de viewport. Verander het gedeelde gedrag
van de voorlopige groepsindeling op `/processing` niet.

## Hoe evenwichtig zijn de groepen?

Toon direct onder de kop één compacte tabel met de expliciete kolommen **Hele groep of
jaarlaag**, **Verschil tussen grootste en kleinste groep** en **Grootste verschil tussen
jongens en meisjes binnen een groep**. Gebruik één rij voor de hele groep en één rij per
aanwezige jaarlaag. Maak alleen de uitkomsten vetgedrukt; rij- en kolomnamen blijven
normaal gewicht. Deze primaire tabel past zonder horizontaal scrollen op smalle schermen.

Plaats daaronder de standaard gesloten uitklapper **Bekijk waar de verschillen zitten**.
Gebruik hierin twee afzonderlijke native tabellen: één met aantallen per groep en één met
compacte jongens/meisjes-aantallen. Gebruik dezelfde rij- en groepsindeling, markeer per
rij alleen de grootste en kleinste groepsgrootte of het grootste jongens/meisjesverschil,
en laat alleen deze detailtabellen lokaal scrollen als dat nodig is.

## Tevredenheid en voorkeuren per leerling

Vervang de pandas-tabs **Leerlingtevredenheid** en **VervuldeVoorkeuren** door één gesloten
uitklapper:

> Tevredenheid en voorkeuren per leerling

Inleiding:

> Tevredenheid laat zien hoe goed de voorkeuren van een leerling zijn uitgekomen.
> Belangrijkere voorkeuren tellen zwaarder. Daarom is het percentage niet hetzelfde als
> het aantal gehonoreerde voorkeuren.

Bouw dit overzicht als servergerenderde, semantische HTML uit de leerlinggegevens die al
in `groepsindeling_view.json` staan. Gebruik geen pandas-HTML, Excelopmaak of parser op de
opgeslagen HTML.

Per leerling toont één compacte, responsieve rij in dezelfde visuele basis als het
overzicht op `preference_form`:

- de volledige naam;
- het afgeronde tevredenheidspercentage in dezelfde badgevorm als de groepskaartpopover;
- alle voorkeuren direct in dezelfde rij, met de bestaande compacte doel- en belangweergave.
  Gebruik naast kleur ook een vinkje of kruis voor het resultaat; schrijf de status niet
  opnieuw uit en toon geen losse telling.

Gebruik voor het belang dezelfde tekens als in de groepskaartpopover: `~`, `↑` en `♥`.
Toon Extra zekerheid en Niet-in-groep-uitsluitingen niet in deze compacte lijst; ze zijn
geen voorkeuren en blijven beschikbaar in de bestaande invoer en groepskaartdetails.

Sorteer leerlingen van hoog naar laag op tevredenheid en daarna op naam. Sorteer voorkeuren
eerst op type (positief vóór de overige), daarna van belangrijk naar minder belangrijk en
bij gelijk type en belang eerst de gehonoreerde voorkeuren.

Leerlingen zonder voorkeuren komen onderaan en tonen **Geen voorkeuren ingevuld**. Gebruik
voor 100% een rustig groen met voldoende contrast, niet het huidige felle `#00ff00`.

Herhaal geen huidige groep, jaarlaag of jongen/meisje in dit overzicht. Die informatie
staat al in de groepskaarten en verklaart de tevredenheid niet.

## Herkomst van leerlingen

Vervang de pandas-tab **Overgangsmatrix** door een gewone sectie **Herkomst van
leerlingen**. Toon daarin altijd, dus buiten en boven de gesloten matrixuitklapper:

> Maximaal `{n}` leerlingen uit dezelfde huidige groep komen samen in één groep in deze
> groepsindeling.

Bereken `{n}` als de hoogste celwaarde in de herkomstmatrix. Dit is de gerealiseerde
kliekgrootte van de groepsindeling, niet een ingestelde bovengrens.

Plaats daaronder één gesloten uitklapper:

> Waar komen de leerlingen vandaan?

Met de uitleg:

> Hier zie je vanuit welke huidige groepen de leerlingen over de groepen in deze
> groepsindeling zijn ingedeeld.

Leid de matrix server-side af uit `origin_full` op ieder leerlingkaartje en de groepkaart
waarin dat kaartje staat. Render een compacte semantische HTML-tabel met huidige groepen
als rijen en groepen in de groepsindeling als kolommen. Voeg geen rij- of kolomtotalen
toe. Gebruik geen pandas-HTML en wijzig `display_transition_matrix()` of de Excel-sheet
niet.

## Sociogram

Verwijder de prominente oranje actie bovenaan. Plaats na de twee analyse-uitklappers een
rustige secundaire tekstlink:

> Wil je zien welke leerlingen graag wel of niet bij elkaar willen zitten? **Bekijk het
> sociogram ↗**

Behoud de bestaande route, het nieuwe tabblad en de sociogrampagina ongewijzigd.

## Tevreden, downloaden, bijsturen en afronden

Gebruik onderaan:

> Ben je tevreden met deze groepsindeling?

De Excel-download is belangrijk, maar verschijnt pas na de inhoudelijke beoordeling.
Maak downloaden niet verplicht en zet de download direct onder de vraag of de
groepsindeling bewaard of gedeeld moet worden:

> Wil je de **groepsindeling** bewaren of delen?

Actie direct daaronder: **Download als Excel-bestand**. Behoud de bestandsnaam
`results.xlsx` en alle bestaande sheets.

Laat de verderactie de vraag beantwoorden:

> Ja, ik ben tevreden!

Deze link blijft naar de bestaande afrondingspagina gaan; verander die pagina niet.

Gebruik daarnaast links van de verderactie een gesloten native uitklapper die als primaire,
gevulde oranje knop is vormgegeven. De uitklapper bevat vier afzonderlijke scenario's:

> ← Nog niet helemaal... opnieuw invoeren

Inhoud en bestaande bestemmingen:

- **Eén leerling verdient meer aandacht.** Vul extra voorkeuren in of kies Extra
  zekerheid voor deze leerling. Link **Voorkeuren aanpassen** naar het formulier of de
  Excel-route volgens de al opgeslagen invoermethode.
- **Een groep lijkt pedagogisch te zwaar.** Voeg een spreidingsregel toe voor leerlingen
  die je over de groepen wilt verspreiden. Link **Leerlingen spreiden aanpassen** naar de
  bestaande pagina.
- **Te weinig voorkeuren zijn gehonoreerd.** Geef ALI Express meer ruimte door grotere
  verschillen tussen groepen toe te staan. Link **Ruimte voor verschillen tussen groepen
  aanpassen** naar de bestaande berekenpagina met een queryparameter die de uitklapper met
  deze velden direct opent.
- **De verschillen zijn te groot.** Maak de toegestane verschillen tussen groepen kleiner
  om de groepen evenwichtiger te maken. Gebruik dezelfde bestaande berekenpagina en
  queryparameter.

Voeg hiervoor geen modal, nieuwe route of sessiestaat toe. Lees alleen de bestaande
invoermethode om de juiste voorkeurenroute te kiezen. Gebruik voor de laatste link een
duidelijke, in code Engelstalige queryparameter, bijvoorbeeld `?edit=differences`, en laat
de bestaande `balance_limits_open`-presentatiecontext daarop reageren. Bewaar de keuze
niet en wijzig de betekenis of standaardwaarden van de velden niet.

## Route- en datagedrag

Maak uit het geladen `groepsindeling_view` alleen een tijdelijk paginaviewmodel met:

- de vlakke, gesorteerde leerlingresultaten;
- de absolute voorkeursaantallen en bestaande voorkeursdetails;
- de balanssamenvattingen en extrema per groep;
- de herkomstmatrix en haar grootste celwaarde;
- het totale aantal nieuw ingedeelde en al aanwezige leerlingen.

Sla dit paginaviewmodel niet op. Wijzig `groepsindeling_view.json`,
`result_tables.json`, `results.xlsx`, solveruitvoer en opslagbetekenis niet.

De normale resultaatpagina gebruikt het gestructureerde viewmodel; er blijft geen
legacyweergave van oude voorgerenderde resultaat-HTML in de lucht.

De normale download blijft rechtstreeks `results.xlsx` versturen. Wanneer dat bestand
onverwacht ontbreekt, moet `/download` niet zelf een tweede, lege variant van
`result.html` renderen. Toon de bestaande fout als flash en redirect naar de normale
`/result`-route, zodat de zichtbare groepsindeling behouden blijft. Voeg geen route toe.

## JavaScript en toegankelijkheid

- Verwijder de niet-semantische tab-`div`s en de inline `showTab()`-functie; native
  uitklappers hebben daarvoor geen JavaScript nodig.
- Voeg geen JavaScript toe voor de bijstuurhulp, balansdetails of analyses.
- Behoud het bestaande gedeelde leerlingpopover-script. Los de mobiele plaatsing op met
  resultaatpagina-specifieke layout/CSS en verander de goedgekeurde tussenstand niet.
- Gebruik echte headings, links, knoppen, lijsten, `<details>/<summary>` en tabellen met
  correcte rij- en kolomkoppen.
- Betekenis rust niet alleen op rood/groen; voorkeuruitkomsten hebben zichtbare tekst.
- Focus blijft zichtbaar en de hele pagina werkt met toetsenbord.
- Bij 320 CSS-px, 390 px en 200% zoom ontstaat geen horizontale paginascroll. Lange
  leerling- en groepsnamen blijven volledig leesbaar.

## Bestandsscope

Toegestaan:

- `templates/result.html`;
- nieuw `static/result.css`, uitsluitend geladen op de resultaatpagina;
- strikt noodzakelijke pure presentatiehelpers en download-foutafhandeling in
  `src/aliexpress/web/routes/results.py`;
- het openen van de bestaande uitklapper op `/processing` via presentatiecontext in
  `src/aliexpress/web/routes/results.py`; wijzig daarvoor `templates/processing.html`
  niet;
- gerichte tests in `tests/test_results.py`;
- gerichte resultaatgedragstests in `tests/browser/`, bij voorkeur een klein nieuw
  `test_result_browser.py` in plaats van meer verwerkingstests in het bestaande grote
  bestand;
- een kleine actualisering van `docs/adr/0016-groepsindeling-gestructureerde-viewmodel.md`:
  de webpagina leidt haar native analyses af uit het bestaande viewmodel, terwijl de
  opgeslagen pandas-tabellen en Excel-export behouden blijven;
- dit plan en de statusregel in `docs/plans/toegankelijke-app/README.md`.

Niet toegestaan:

- solver- of optimalisatiecode;
- `src/aliexpress/main.py`, `tasks.py`, `process_files.py` of het datamodel;
- wijzigingen aan gewichten, tevredenheidsberekening, balansbetekenis, verdeelmodi of
  opgeslagen artifacts;
- inhoudelijke of visuele wijzigingen aan de sociogram-, processing-, invoer- of
  afrondingspagina;
- een nieuwe route, sessiesleutel, modal of frontend datamodel;
- een nieuwe bestandsnaam voor de Excel-download.

## Gedragsacceptatie

- De groepskaarten blijven volledig zichtbaar en functioneren zoals nu; eventuele
  blijvende bezetting wordt één keer algemeen uitgelegd.
- De altijd zichtbare balanssamenvatting toont per totaal en aanwezige jaarlaag alleen het
  verschil tussen de grootste en kleinste groep en het grootste verschil tussen jongens en
  meisjes binnen een groep. Alleen de uitkomsten zijn vetgedrukt en de volledige tabel past
  op smalle schermen; de detailtabellen tonen de onderliggende aantallen.
- Na openen toont het native leerlingoverzicht alle leerlingen van hoog naar laag, het
  afgeronde percentage en de concrete voorkeuruitkomsten zonder dubbele stamgroep- of
  geslachtskolommen. Extra zekerheid, Niet-in-groep-uitsluitingen en een losse `x van y`-
  telling worden niet in deze compacte lijst getoond. De voorkeuren staan eerst positief,
  vervolgens van hoog naar laag belang en bij gelijke waarde eerst gehonoreerd.
- Na openen toont de native herkomstmatrix dezelfde tellingen als de bestaande
  overgangsmatrix, met begrijpelijke koppen en zonder rij- of kolomtotalen. De altijd
  zichtbare samenvatting erboven noemt de hoogste celwaarde als maximaal aantal
  leerlingen uit dezelfde huidige groep dat samen in één groep komt.
- **Download als Excel-bestand** levert `results.xlsx`; een ontbrekend bestand geeft een
  foutmelding en keert terug naar de normale resultaatpagina.
- **Ja, ik ben tevreden!** gaat naar de bestaande afrondingspagina.
- De vier bijstuuracties gaan zonder nieuwe routes naar de juiste bestaande invoerstap;
  de voorkeurenlink respecteert formulier versus Excel en de link naar de ruimte voor
  groepsverschillen opent het bestaande relevante veldblok.
- Op laptopbreedte, 390 px en 320 px, bij 200% zoom en met lange namen is er geen
  horizontale paginascroll of afgeknotte informatie.
- Alles is bruikbaar met toetsenbord en zichtbare focus; open/dicht-toestanden en
  voorkeuruitkomsten zijn semantisch waarneembaar.

De visuele volgorde, rustige presentatie van het sociogram en technische keuze voor
native HTML zonder tab-JavaScript worden handmatig en tijdens code-review gecontroleerd;
voeg daarvoor geen nieuwe gedragstests toe.

Tests zoeken op betekenisvolle rollen, namen, routes, toestanden en uitkomsten. Ze mogen
enkele labels gebruiken om bediening te vinden, maar dupliceren geen volledige statische
teksten, template-HTML, CSS-keuzes of helperimplementaties. Test de native matrix met
enkele betekenisvolle cellen en de zichtbare maximale kliekgrootte, niet door de hele
tabel als snapshot vast te zetten.

## Verificatie en review

Voer na iedere wijziging de kleinste relevante test uit. Rond de kandidaat ten minste af
met:

```bash
uv run pytest tests/test_results.py --no-cov -n 4 --dist load
uv run pytest tests/browser/test_result_browser.py \
  tests/browser/test_distribution_browser.py -q --no-cov -n 4 --dist load
```

Als de nieuwe browsertest anders wordt genoemd, pas alleen het pad aan. Een integratierun
is niet nodig zolang solver-, data- en opslaggedrag werkelijk ongewijzigd blijven.

Controleer daarnaast in een echte browser:

1. een resultaat van Doorzetten met blijvende bezetting;
2. een resultaat van Herindelen met meerdere jaarlagen;
3. formulier- en Excelinvoer voor de juiste bijstuurroute;
4. laptopbreedte, 390 px, 320 px en 200% zoom;
5. lange leerling- en groepsnamen;
6. bediening met alleen toetsenbord;
7. een ontbrekend `results.xlsx` met behoud van de normale resultaatpagina;
8. de bijstuuractie voor groepsverschillen, waarbij de bestaande velduitklapper na
   navigatie direct open staat.

Maak reviewbeelden op laptopbreedte en 390 px met fictieve gegevens en bewaar die buiten
de repository, bijvoorbeeld onder `/tmp`. Rapporteer de paden en alle testresultaten.
Vraag de eigenaar om de uiteindelijke pagina te beoordelen en maak pas na expliciete
goedkeuring een commit.
