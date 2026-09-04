---
status: slice-5-complete
branch: feature/sociogram-cytoscape
---

# TDD-implementatieplan: interactief sociogram met Cytoscape.js

## Doel

ALI Express toont na het opslaan van de voorkeuren een interactief **Sociogram** van
de relaties tussen leerlingen. Nauwe positieve verwantschappen liggen gemiddeld
dichter bij elkaar; negatieve voorkeuren liggen gemiddeld verder uit elkaar en
blijven zichtbaar als rode gerichte pijlen. De weergave staat los van de
**Verdeling** en mag die nooit blokkeren.

Dit plan werkt met verticale TDD-slices. Iedere slice doorloopt lokaal
`RED -> GREEN -> eventueel REFACTOR` en eindigt in precies één groene, zelfstandig
reviewbare commit. Er komen geen aparte commits met alleen tests of alleen
infrastructuur. Commit 1 is de tracer bullet en levert direct een werkende route
met echte procesdata; volgende commits breiden dat resultaat zichtbaar uit.

## Voortgang

- [x] Slice 1 — echte voorkeuren in de browser. Nodes en gerichte positieve/negatieve
  voorkeuren worden direct uit `voorkeuren.json` gebouwd en met lokaal Cytoscape
  gerenderd. De eenvoudige grid-layout en de read-only browserdiagnose zijn aanwezig.
- [x] Slice 2 — sociometrische layoutprojectie.
- [x] Slice 3 — gewicht, focus en lokale leesbaarheid. Begrensde interactie, focus
  met popup en leesbare referentiedataset zijn gereviewd en geïmplementeerd.
- [x] Slice 4 — direct beschikbaar in de Procesflow. Canonieke voorkeuren ontsluiten
  het sociogram vóór, tijdens en na de Verdeling; de solverflow is onafhankelijk.
- [x] Slice 5 — objectieve layoutkwaliteit op `voorkeuren.xlsx`. De referentietest
  haalt bij 1440 × 1000 deterministisch 23 proxy-kruisingen, geen node-overlap en
  mediane afstanden van ongeveer 58,4 / 71,6 / 92,7 voor wederzijds positief /
  eenzijdig positief / negatief; alle zes negatieve paren zijn langer dan de
  positieve mediaan. De standaardweergave is visueel gecontroleerd.
- [ ] Slice 6 — schaal, offlinegarantie en verwijderen oude stack.

## Vaststaande keuzes

- Cytoscape.js 3.34.0 verzorgt browserlayout, rendering en interactie.
- Cytoscape wordt lokaal gevendord; de browser gebruikt geen CDN en er komt alleen
  voor deze feature geen npm/buildstraat.
- SHA-256 van het gecontroleerde bestand:
  `9C2A3BF2592E0B14A1F7BEC07C03A54F16DEDF32AF9CD0AF155C716AA6C87BC3`.
- De applicatie blijft proprietair; Cytoscape blijft afzonderlijk onder de
  MIT-licentie beschikbaar. De licentietekst blijft in het bestand en komt in de
  third-partyadministratie.
- De Pythonkant bouwt alleen een klein, JSON-serialiseerbaar sociogram-viewmodel.
  Er komt geen eigen graafbibliotheek of layoutoptimizer.
- Alleen leerling-naar-leerlingvoorkeuren worden getoond. Voorkeuren voor een
  bestemmingsgroep, huidige groep en de uiteindelijke Verdeling beïnvloeden het
  Sociogram niet.
- Iedere oorspronkelijke Voorkeur blijft een afzonderlijke, gerichte zichtbare
  pijl. Per ongeordend leerlingenpaar komt daarnaast één afgeleide, onzichtbare
  layoutrelatie.
- De bestaande nodegrootte op basis van de inkomende, per voorkeur op `[-2, 2]`
  begrensde voorkeurssom blijft in de eerste versie behouden. Noem dit in de UI
  een *ontvangen voorkeursscore*, niet een formele sociometrische status.
- Geen formele PNG-, PDF- of Excel-export, geen opgeslagen handmatige posities en
  geen sociometrische statusclassificaties in deze versie.
- `stash@{0}` wordt niet toegepast. De tests daarin mogen als historische
  inspiratie dienen; de eigen Python/NumPy-SMACOF-optimizer wordt niet hergebruikt.

## Waarom CoSE — en wat CoSE hier niet doet

CoSE staat voor *Compound Spring Embedder*. Een compound graph kan nodes in
bovenliggende, eventueel geneste groepsnodes bevatten. Dat is een hiërarchie van
containers; het betekent niet dat CoSE gewone relaties automatisch als een
hiërarchie tekent. Dit Sociogram heeft geen compound nodes. De compound-uitbreidingen
van CoSE worden hier dus niet benut: voor onze invoer is CoSE in de kern een gewone
force-directed spring-layout, net als NetworkX' Fruchterman-Reingold-layout.

De verwachte kwaliteitswinst wordt daarom niet toegeschreven aan een intrinsiek
superieur of hiërarchisch CoSE-algoritme. De vergelijking met de oude implementatie
verandert tegelijkertijd de representatie en de layoutmotor:

| Oude implementatie | Nieuwe implementatie |
|---|---|
| 102 ruwe, gerichte voorkeurspijlen sturen de layout | 73 afgeleide relaties per leerlingenpaar sturen de layout |
| één globale voorkeursafstand `k` | `idealEdgeLength` kan per layoutrelatie verschillen |
| gewicht verandert vooral de aantrekkingskracht | categorie en gewicht bepalen een begrensde rustlengte |
| zichtbare en positionerende relaties zijn hetzelfde | zichtbare pijlen en onzichtbare layoutrelaties zijn gescheiden |

De sociometrische projectie — wederzijds, eenzijdig en negatief vertalen naar
verschillende ideale afstanden — is dus de belangrijkste inhoudelijke verbetering.
CoSE is gekozen als eerste uitvoeringsmotor omdat het al in Cytoscape.js zit, zonder
extra dependency per edge een ideale lengte accepteert en opties heeft voor
nodeafstoting, overlap en componentafstand. Het is daarmee een pragmatische en
vervangbare keuze, geen domeinbeslissing.

Een werkelijk eerlijke algoritmevergelijking zou zowel de ruwe als de afgeleide
relaties met meerdere motoren testen. Dat is voor de MVP niet nodig zolang de
objectieve criteria uit slice 5 worden gehaald. Haalt CoSE die criteria niet, dan
blijven `SociogramView` en de sociometrische projectie staan en vergelijken we pas
een andere Cytoscape-layout; we bouwen niet meteen een eigen optimizer.

Zie ook de officiële
[Cytoscape.js-documentatie voor CoSE](https://js.cytoscape.org/#layouts/cose) en de
[NetworkX-documentatie voor `spring_layout`](https://networkx.org/documentation/stable/reference/generated/networkx.drawing.layout.spring_layout.html).

## Publieke interfaces

De tests spreken alleen gedrag aan via deze grenzen:

1. `build_sociogram_view(preference_data) -> SociogramView` bouwt een vlak,
   `dataclasses.asdict`-baar viewmodel uit de canonieke `PreferenceData`.
2. `GET /sociogram` laadt de voorkeuren van het actieve Proces en rendert het
   viewmodel. Authenticatie en `require_process` blijven gelden.
3. De sociogrampagina biedt als observeerbaar browsergedrag: nodes en pijlen,
   layout, selectie/focus, details, zoomen en slepen.
4. Een kleine read-only browserdiagnose, bijvoorbeeld
   `window.sociogramSnapshot()`, geeft na `sociogram:ready` uitsluitend nodecentra,
   bounding boxes en relatie-eindpunten terug. De Playwright-acceptatietest gebruikt
   dit om afstanden, overlap en kruisingen te meten zonder private helperfuncties te
   testen.

Voorgesteld vlak datacontract; voeg velden pas toe in de slice die ze gebruikt:

```python
@dataclass(frozen=True)
class SociogramNode:
    id: str
    label: str
    received_preference_score: float
    size: float

@dataclass(frozen=True)
class PreferenceEdge:
    id: str
    source: str
    target: str
    weight: float
    kind: str                    # "positive" | "negative"

@dataclass(frozen=True)
class LayoutRelation:
    source: str
    target: str
    kind: str                    # mutual_positive | positive | negative
    strength: float
    ideal_distance: float

@dataclass(frozen=True)
class SociogramView:
    nodes: list[SociogramNode]
    preferences: list[PreferenceEdge]
    layout_relations: list[LayoutRelation]
```

De concrete veldnamen mogen tijdens de eerste slice eenvoudiger blijken, zolang de
interface klein blijft, de waarden plat en JSON-veilig zijn en tests gedrag in
plaats van de interne bouwstappen beschrijven.

## Relatieregels

De onzichtbare layoutrelatie combineert beide richtingen als volgt:

- eenzijdig positief: de kracht is het gewicht van die Voorkeur;
- wederzijds positief: de kracht is het gemiddelde van beide positieve gewichten;
- positief plus negatief: voor de layout is de relatie negatief;
- negatief: de sterkste absolute negatieve Voorkeur bepaalt de kracht;
- wederzijds negatief blijft de negatieve categorie met de grootste gewenste
  afstand;
- de categorie bepaalt eerst de afstandsband; gewicht nuanceert de afstand binnen
  die band via een begrensde, niet-lineaire schaal;
- afwezigheid van een relatie is neutraal: algemene nodeafstoting voorkomt overlap,
  maar er komt geen zichtbare neutrale pijl.

Daarmee geldt op categorieniveau:

```text
wederzijds positief < eenzijdig positief < negatief
```

Een expliciete negatieve Voorkeur wordt nooit weggemiddeld tegen een positieve.

## Slice 1 — tracer bullet: echte voorkeuren in de browser

**Gebruikersresultaat:** `GET /sociogram` toont voor het actieve Proces direct alle
leerlingen en hun gerichte positieve/negatieve Voorkeuren met Cytoscape. Gebruik in
deze eerste slice een eenvoudige ingebouwde layout; perfecte afstanden en interactie
volgen later. De bestaande resultaatlink vormt al een klikbaar pad naar deze pagina.

**RED**

- Vervang de routeverwachting “lees opgeslagen `sociogram.html`” door gedrag:
  een Proces met `voorkeuren.json` krijgt HTTP 200 en gerenderde sociogramdata.
- Test via de route dat leerlingen zonder persoonlijke Voorkeur ook als node worden
  geleverd, groepsdoelen geen pijl worden en richting/teken/gewicht behouden blijven.
- Voeg één Playwright-tracer toe die een klein echt Proces opent en waarneemt dat
  Cytoscape nodes en gerichte pijlen rendert.
- Voer de smalle tests uit en constateer dat ze op de bestaande Plotly-route falen.

**GREEN**

- Voeg de lokaal gecontroleerde `cytoscape-3.34.0.min.js` toe onder `static/vendor/`.
- Voeg meteen de MIT-vermelding toe aan `THIRD_PARTY_NOTICES.md` en nuanceer de
  algemene licentietekst in de README voor componenten van derden.
- Introduceer het minimale `SociogramView` en `build_sociogram_view` in
  `src/aliexpress/sociogram.py`: alleen nodes en zichtbare voorkeurspijlen.
- Laat `results.show_sociogram` `load_voorkeuren` gebruiken en het viewmodel aan
  `templates/sociogram.html` geven.
- Render lokaal met een simpele Cytoscape-layout, namen, pijlpunten, doorgetrokken
  positieve lijnen en rode gestreepte negatieve lijnen.
- Laat de oude achtergrondgenerator tijdelijk bestaan als veilig, maar ongebruikt
  pad; opruimen gebeurt pas nadat de nieuwe flow afzonderlijk bewezen is.

**Reviewpunt:** echte voorkeuren zijn zichtbaar en onderzoekbaar; nog geen claim
over optimale afstanden, kruisingen of definitieve styling.

**Commit:** `feat(sociogram): render preferences with local Cytoscape`

## Slice 2 — sociometrische layoutprojectie

**Gebruikersresultaat:** wederzijdse positieve relaties trekken het sterkst samen,
eenzijdige positieve relaties minder sterk en negatieve relaties worden op grotere
afstand gelegd. De gerichte bronpijlen blijven ongewijzigd zichtbaar.

**RED**

- Voeg per relatieregel één gedragsgerichte test toe via
  `build_sociogram_view`: wederzijds positief, eenzijdig positief, gemengd teken,
  wederzijds negatief en een neutraal paar.
- Test het afgesproken gemiddelde voor wederzijds positief en het sterkste negatieve
  gewicht bij een negatieve relatie.
- Test dat categoriebanden altijd domineren over gewichtsnuances.
- Voer na iedere toegevoegde test eerst de smalle rode test uit; implementeer steeds
  alleen het zojuist beschreven gedrag.

**GREEN**

- Voeg één `LayoutRelation` per ongeordend leerlingenpaar met een Voorkeur toe.
- Gebruik een begrensde niet-lineaire omzetting van kracht naar ideale afstand.
- Geef uitsluitend nodes plus onzichtbare layoutrelaties aan Cytoscape CoSE voor de
  positionering; de zichtbare gerichte pijlen sturen de layout niet nogmaals.
- Gebruik algemene nodeafstoting voor neutrale paren en overlapvermijding.

**Reviewpunt:** de testnamen lezen als de afgesproken domeinregels en de browser laat
voor het eerst herkenbare sociale clusters en langere negatieve verbindingen zien.

**Commit:** `feat(sociogram): positioneer leerlingen op sociale relaties`

## Slice 3 — gewicht, focus en lokale leesbaarheid

**Gebruikersresultaat:** de totale sociale structuur blijft zichtbaar, terwijl een
leerkracht één leerling of pijl gericht kan onderzoeken.

**RED**

- Playwright test dat een hoger absoluut gewicht een dikkere lijn geeft, zonder een
  exacte interne formule vast te pinnen.
- Test dat positieve/negatieve betekenis niet alleen door kleur maar ook door
  doorgetrokken/gestreept wordt aangegeven.
- Test dat wederzijdse pijlen afzonderlijk herkenbaar zijn.
- Test dat selectie van een leerling alle inkomende en uitgaande pijlen accentueert,
  de rest dimt en dat klikken op de achtergrond de selectie wist.
- Test dat selectie van een pijl bron, doel, soort en exact gewicht toont.
- Test dat zoomen binnen de afgesproken onder- en bovengrens blijft en dat de resetknop
  het volledige Sociogram terug in beeld brengt.
- Test dat details als popup binnen de tekenarea verschijnen en met achtergrondklik of
  Escape verdwijnen.

**GREEN**

- Voeg een begrensde niet-lineaire lijndikteschaal toe; `5` is duidelijk dikker dan
  `2`, maar niet letterlijk 2,5 maal zo dik.
- Gebruik gebogen lijnen waar twee leerlingen elkaar kiezen, zodat beide richtingen
  zichtbaar blijven.
- Voeg focus/dimmen en een compacte popup binnen de tekenarea toe; laat die popup de
  onderliggende pijlen niet blokkeren.
- Behoud pan, zoom en handmatig slepen tijdens de geopende pagina, maar begrens zoom
  met `minZoom: 0.35`, `maxZoom: 2.5` en een lagere `wheelSensitivity`; voeg een
  resetknop toe die het volledige Sociogram fit.
- Gebruik naast rood altijd streepstijl en tekstuele betekenis voor toegankelijkheid.

**Reviewpunt:** controleer in de browser zowel het totaalbeeld als één geselecteerde
leerling met wederzijdse en negatieve Voorkeuren. Controleer ook dat de popup niet buiten
de tekenarea valt en dat een extreem muiswielgebaar het overzicht niet onbruikbaar maakt.

**Commit:** `feat(sociogram): make relationships inspectable`

## Slice 4 — direct beschikbaar in de Procesflow

**Gebruikersresultaat:** het Sociogram is beschikbaar zodra geldige voorkeuren zijn
opgeslagen. Het wacht niet op de Verdeling en een visualisatiefout kan de Verdeling
niet raken.

**RED**

- Route-/browsertest dat de sociogramlink na opgeslagen voorkeuren zichtbaar is op
  het voorkeurenoverzicht, de processingpagina en de resultaatpagina.
- Test dat het starten van de Verdeling slechts de solverthread start.
- Test dat `/status` en de processingpagina niet meer afhankelijk zijn van
  `sociogram_ready` of een `sociogram.html`-artifact.
- Test dat ontbrekende of onleesbare voorkeuren een begrijpelijke fout op de
  sociogrampagina geven zonder de status van een solverrun te veranderen.

**GREEN**

- Maak de links in beide voorkeureninvoerpaden zichtbaar zodra canonieke
  `voorkeuren.json` bestaat; behoud de links op processing en resultaat.
- Toon de processinglink direct in “Terwijl je wacht”, zonder readiness-poll.
- Verwijder het starten en de import van `create_sociogram_thread` uit `wizard.py`.
- Verwijder `sociogram_ready` uit `/status` en bijbehorende browser-JavaScript.
- Verwijder de achtergrondfunctie uit `web/tasks.py` en stop met schrijven/lezen van
  het gegenereerde `sociogram.html`-artifact.
- Haal `sociogram.html` uit `reset_result_files`; canonieke voorkeuren zijn voortaan
  de enige procesinvoer voor het Sociogram.

**Reviewpunt:** vóór het starten, tijdens het rekenen en na het resultaat opent
dezelfde route hetzelfde actuele Sociogram. De solverflow blijft groen.

**Commit:** `feat(sociogram): maak visualisatie direct beschikbaar`

## Slice 5 — objectieve layoutkwaliteit op voorkeuren.xlsx

**Gebruikersresultaat:** het referentiebestand levert een bruikbaar totaaloverzicht
zonder nodekluwen en met weinig lijnkruisingen.

De referentiedataset `testdata/voorkeuren.xlsx` is hierbij leidend: die bevat echte
subgroepstructuren (43 leerlingen, 102 leerling-naar-leerlingvoorkeuren en 73
layoutparen) en is daarom relevanter voor layoutbeoordeling dan willekeurig gegenereerde
voorkeuren. Kleine of gegenereerde bestanden blijven nuttig voor snelle smoke-tests.

**RED**

- Voeg een Playwright-acceptatietest toe die `testdata/voorkeuren.xlsx` via de echte
  invoer-/routegrens rendert en wacht op `sociogram:ready`.
- Meet via `sociogramSnapshot()` uitsluitend observeerbare geometrie na de layout.
- Definieer een kruising als een snijding tussen de binnenkanten van twee zichtbare
  gerichte relaties zonder gemeenschappelijk eindpunt. Iedere zichtbare pijl telt;
  de berekening gebruikt de rechte verbinding tussen nodecentra als stabiele proxy
  voor de gerenderde curve.
- Controleer node-bounding-boxes op overlap en nodecentrumafstanden per categorie.

**GREEN**

- Tune alleen de gedocumenteerde Cytoscape-opties voor ideale afstand,
  elasticiteit, nodeafstoting, overlap en componentafstand.
- Geef leerlingen in vaste volgorde eenvoudige deterministische beginposities en
  schakel een willekeurige start uit als Cytoscape daarmee zonder extra toestand
  reproduceerbaar blijft.
- Voeg geen eigen optimalisatielus, seed-injectie in `Math.random`, opslagmodel of
  meervoudige layoutzoektocht toe. Als exacte reproduceerbaarheid daarmee niet
  haalbaar is, blijft die een wens en geen blokkerend criterium.

**Harde datasetcriteria bij een viewport van 1440 × 1000:**

- 43 leerlingnodes;
- 102 zichtbare gerichte pijlen: 96 positief en 6 negatief;
- 73 layoutparen: 26 wederzijds positief, 41 eenzijdig positief en 6 negatief;
- geen overlappende nodes;
- alle 43 leerlinglabels zijn in de standaardweergave aanwezig en bij menselijke
  controle leesbaar; zoomen is niet nodig om vast te stellen welke clusters er zijn;
- minder dan 30 lijnkruisingen;
- mediane afstand strikt geordend als
  `wederzijds positief < eenzijdig positief < negatief`;
- minimaal 4 van de 6 negatieve layoutrelaties langer dan de mediane positieve
  relatie;
- alle zes negatieve pijlen rood, gestreept en gericht zichtbaar.

De prototypevalidatie gaf bij vijf willekeurige starts 12–25 kruisingen; de grens
`< 30` is dus ambitieus maar al empirisch haalbaar. Een screenshot blijft een
menselijk reviewhulpmiddel en wordt geen pixel-perfect regressietest.

**Reviewpunt:** voeg de actuele screenshot en de gemeten criteria aan de PR of
commitreview toe.

**Commit:** `feat(sociogram): borg leesbare layout op referentiedata`

## Slice 6 — schaal, offlinegarantie en verwijderen oude stack

**Gebruikersresultaat:** de nieuwe implementatie is het enige sociogrampad, werkt
offline en blijft responsief voor de afgesproken omvang.

**RED**

- Test dat de sociogrampagina geen verzoek buiten de eigen origin uitvoert.
- Voeg een niet-timinggevoelige browsertest toe die 150 leerlingen en 1.000
  Voorkeuren rendert en binnen een ruime testdeadline `sociogram:ready` bereikt.
- Meet lokaal afzonderlijk dat 100 leerlingen op een normale ontwikkellaptop in
  circa 3 seconden een eerste layout krijgen; leg de waarneming vast, maar maak van
  een strakke wandklokgrens geen flakey CI-test.
- Voeg een repositorytest of gerichte controle toe dat productiecode geen
  `networkx`, `matplotlib` of `plotly` meer importeert.

**GREEN**

- Verwijder de oude `SociogramMaker`, Matplotlib/NetworkX-rendering en handmatige
  Plotly-traces uit `sociogram.py`.
- Verwijder NetworkX, Matplotlib en Plotly uit `pyproject.toml` en werk het lockbestand
  bij. Deze dependencies zijn uitsluitend voor deze feature aanwezig.
- Verwijder of herschrijf de oude backend- en artifacttests.
- Werk README-uitleg, docstrings en eventuele routearchitectuurcommentaren bij.
- Controleer dat het gevendorde bestand nog exact de vastgelegde SHA-256 en volledige
  MIT-melding heeft.

**Reviewpunt:** een lokale/offline browserrun werkt, de dependencydiff is negatief
en de volledige suite is groen.

**Commit:** `refactor(sociogram): verwijder de Python-grafiekstack`

## TDD-regels per slice

Voor iedere slice geldt:

1. Begin met precies één eerstvolgend observeerbaar gedrag.
2. Voer de nieuwe smalle test uit en noteer dat deze om de bedoelde reden rood is.
3. Schrijf alleen genoeg productiecode om die test groen te maken.
4. Herhaal test voor test binnen dezelfde slice; schrijf niet alle slicetests vooraf.
5. Refactor alleen terwijl alle tests groen zijn.
6. Test via `build_sociogram_view`, Flaskroutes en de echte browser; mock geen eigen
   sociogramhelpers. Mock alleen echte systeemgrenzen als dat onvermijdelijk is.
7. Draai vóór de commit de relevante regressiesuite en lint.
8. Maak één groene commit met uitsluitend de slice; begin de volgende slice pas na
   review of expliciete toestemming.

## Verificatie per commit

Gebruik tijdens de cyclus eerst de kleinst mogelijke test, daarna minimaal:

```text
uv run pytest tests/test_sociogram.py tests/test_results.py -q --no-cov
uv run pytest tests/test_wizard_distribution.py tests/test_process_files.py -q --no-cov
uv run pytest tests/browser -q --no-cov
uv run pylint src/aliexpress app.py
```

Niet iedere slice raakt alle vier commando’s. Draai wel vóór iedere commit alle
direct geraakte suites. Na slice 4 en slice 6 draait ook de volledige quick suite:

```text
uv run pytest tests/ --ignore=tests/integration --ignore=tests/browser -q --no-cov
```

Voor de laatste commit:

```text
uv run pytest tests/ -q
uv run pylint src/aliexpress app.py
```

## Definitie van klaar

- Alle zes slices zijn afzonderlijk gereviewd en groen gecommit.
- De harde datasetcriteria uit slice 5 slagen op `testdata/voorkeuren.xlsx`.
- Het Sociogram is vóór, tijdens en na een Verdeling bereikbaar zodra voorkeuren
  geldig zijn opgeslagen.
- Een fout in de visualisatie verandert geen runstatus en blokkeert geen Verdeling.
- De browser gebruikt alleen lokale assets en verstuurt geen leerlinggegevens naar
  derden.
- Cytoscapeversie, hash en MIT-licentie zijn traceerbaar vastgelegd.
- NetworkX, Matplotlib en Plotly zijn uit productiecode en directe dependencies weg.
- Geen eigen layoutoptimizer, exportfunctie, opgeslagen handmatige layout,
  groepering op huidige/bestemmingsgroep of formele sociometrische status is
  toegevoegd.
- `stash@{0}` is niet toegepast of verwijderd; opruimen daarvan blijft een aparte,
  expliciete gebruikersbeslissing.
