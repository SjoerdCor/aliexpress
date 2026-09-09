# Voorkeuren invullen

**Status:** implementatiegereed na gebruikersreview

## Doel

Een leerkracht of IB'er neemt per leerling over bij wie die graag of liever niet in de
nieuwe groep komt, kan uitzonderingen en extra zekerheid vastleggen en kan de actuele
voorkeuren tussentijds in het sociogram bekijken. De pagina moet warm, begrijpelijk en
snel bruikbaar zijn voor iemand die ALI Express nog niet kent.

Dit is uitsluitend de paginaslice voor het webformulier `/preferences_form`. De
Excel-invoer, de volgende pagina voor het spreiden van leerlingen, het sociogram zelf,
solvergedrag, gewichten en de betekenis van opgeslagen voorkeuren blijven ongewijzigd.

## Huidige toestand

- De uitleg staat in een afwijkend geel/perzikkleurig blok en mengt de hoofdtaak met
  uitzonderingen, technische uitleg en een verwijzing naar een niet-bestaand
  `Niet-samen-bestand`.
- De mogelijkheid om een nieuwe groep als voorkeur te kiezen staat achter een info-i.
  Juist het belangrijke geval — de gewenste leerling zit al in bijvoorbeeld Blauw en
  kan daarom niet bij naam worden gekozen — is daardoor gemakkelijk te missen.
- `sociogram_available` kijkt alleen of eerder canonieke voorkeuren zijn opgeslagen.
  Modalopslaan schrijft bewust alleen de draft. Na nieuwe invoer kan de zichtbare link
  daardoor nog een oud of leeg sociogram openen.
- De leerlingregels zijn `div`-elementen met `role="button"`, maar missen native
  knopgedrag voor Enter en Spatie. De modal mist volledige dialogsemantiek.
- Geen voorkeuren is technisch geldig, maar onwenselijk en onwaarschijnlijk. De huidige
  aanduiding `nog niet ingevuld` mag daarom bewust als openstaande taak blijven werken.
- De teksten en toelichting bij extra zekerheid zijn te technisch. De zichtbare niveaus
  blijven inhoudelijk **Geen extra eis**, **Minstens tevreden** en
  **Alle voorkeuren gehonoreerd**.

## Goedgekeurde tekst en interactie

### Pagina

- H1 en vierde voortgangsstap: **Voorkeuren invullen**.
- Intro, als gewone tekst zonder gekleurd kader:

  **Open iedere leerling en neem over bij wie die graag in de nieuwe groep wil komen —
  en bij wie liever niet. Voeg waar mogelijk meerdere positieve voorkeuren toe. Zo geef
  je ALI Express de ruimte om voor iedereen een zo prettig en evenwichtig mogelijke
  groepsindeling te maken.**

- Plaats daarna een standaard, gesloten uitklapper **Hoe gebruikt ALI Express de
  voorkeuren?** met:

  **Ga uit van wat de leerling zelf aangeeft. ALI Express weegt alle voorkeuren samen;
  een voorkeur is dus geen garantie. Een eerste vervulde voorkeur telt het zwaarst.
  Volgende voorkeuren en het belang dat je kiest tellen ook mee.**

  **Wil de school dat leerlingen beslist niet samen komen? Geef dat in de volgende stap
  aan.**

- Behoud de groepering per huidige groep en de teller met het aantal leerlingen waarvoor
  voorkeuren zijn opgegeven. Een lege leerlingregel blijft neutraal maar aansporend:
  **nog niet ingevuld**. Toon als regelactie **Invullen** en na invoer **Wijzigen**.

### Bewerkmodal

- Kop: volledige leerlingnaam; subregel: **Huidige groep: {groepsnaam}**.
- Gebruik de veldkoppen **Graag bij** en **Liever niet bij**.
- Gebruik bij beide zoekvelden het zichtbare label **Leerling of nieuwe groep zoeken**
  en de placeholder **Zoek een leerling of nieuwe groep…**.
- Toon onder **Graag bij** als gewone, altijd zichtbare tekst:

  **Wil deze leerling bij iemand komen die al in een nieuwe groep zit? Kies dan die
  groep; de leerling die daar al zit, kun je hier niet kiezen.**

  Zet dit niet in de intro en niet achter een info-i. Hoewel de tekst in iedere modal in
  de DOM staat, ziet de gebruiker steeds maar één modal; verbergen zou tientallen extra
  handelingen veroorzaken.
- Noem de harde groepsuitsluiting **Mag niet naar** en leg uit:

  **Gebruik dit alleen als deze leerling echt niet in een bepaalde nieuwe groep mag
  komen, bijvoorbeeld omdat er al een broer of zus zit. Dit geldt altijd. Is het alleen
  een voorkeur, kies de groep dan bij ‘Liever niet bij’.**

- Toon de bestaande vaste voorkeursintensiteiten met hun bestaande gewichten. Gebruik bij
  een negatieve voorkeur de vraag **Hoe belangrijk is dit?** met de labels
  **Belangrijk** en **Heel belangrijk**; de opgeslagen waarden blijven 1 en 2.
- Houd **Extra zekerheid voor deze leerling** standaard gesloten. Gebruik de korte uitleg:

  **Bedoeld voor een leerling die sociaal kwetsbaarder is en zeker bij een vertrouwd
  iemand moet komen. Extra zekerheid beperkt de ruimte bij het indelen van andere
  leerlingen en kan ervoor zorgen dat geen geldige groepsindeling mogelijk is. Gebruik
  het daarom alleen waar het echt nodig is.**

  Behoud de zichtbare keuzes en opgeslagen waarden:

  - **Geen extra eis** — leeg;
  - **Minstens tevreden** — 50%;
  - **Alle voorkeuren gehonoreerd** — 100%.

- Acties: **Wijzigingen annuleren** en **Voorkeuren opslaan**.
- Alleen de expliciete knop **Wijzigingen annuleren** verwerpt de wijzigingen. Esc en een
  klik op de backdrop sluiten de modal bewust niet, zodat een toevallige handeling geen
  invoer weggooit. Behoud dit bestaande gedrag en de bijbehorende gedragstests.
- Gebruik een native knop voor iedere leerlingregel. Geef de modal `role="dialog"`,
  `aria-modal="true"` en een koppeling met de modaltitel. Behoud focus op de titel bij
  openen, focusherstel naar de leerlingregel bij sluiten, `inert` op de achtergrond en de
  scroll-lock.
- Gebruik geen info-i of bijbehorende popover-JavaScript op deze pagina. Geef
  verwijderknoppen een naam die het concrete voorkeurdoel of de concrete groep noemt.

### Navigatie en sociogram

- Terugacties volgen de werkelijk voorafgaande zichtbare pagina:

  | Verdeelmodus | Terugactie | Route |
  | --- | --- | --- |
  | Doorzetten | **← Terug naar groepen voor volgend jaar** | `/groups_to` |
  | Herindelen met dezelfde groepen | **← Terug naar leerlingen controleren** | `/roster` |
  | Herindelen met doorzetten | **← Terug naar nieuwe groepen kiezen** | `/select_groups` |

- Primaire vervolgactie: **Verder naar leerlingen spreiden →**. De redirect blijft
  `/not_together`; wijzig die volgende pagina niet in deze slice.
- Toon **Bekijk deze voorkeuren in het sociogram ↗** altijd als secundaire formulieractie.
  Deze actie:

  1. post de actuele inhoud van het bestaande formulier;
  2. gebruikt dezelfde servervalidatie en draftopslag als de primaire vervolgactie;
  3. overschrijft bij succes het canonieke `voorkeuren.json` met de actuele invoer;
  4. redirect daarna in een nieuw tabblad naar de bestaande `/sociogram`-route;
  5. toont bij een fout de normale flashmelding en behoudt de invoer.

  Voeg hiervoor geen nieuwe route, sessiestaat, sociogramvariant of tweede
  opslagbetekenis toe. Zorg dat de `beforeunload`-bescherming in het oorspronkelijke
  tabblad na deze doelgerichte formulier-submit correct blijft werken.

## Documentatiecorrecties binnen deze slice

- Corrigeer `CONTEXT.md` en ADR 0003 zodat 100% extra zekerheid zichtbaar en inhoudelijk
  **Alle voorkeuren gehonoreerd** betekent. Beschrijf 50% als **Minstens tevreden** en
  verander geen gewichten of solvergedrag.
- Corrigeer ADR 0007 naar de latere bewuste modalkeuze: alleen de expliciete
  annuleerknop verwerpt wijzigingen; Esc en backdrop doen dat niet. Verwijder ook de
  achterhaalde vermelding van een sluitkruis als annuleerroute.
- Pas alleen documentatie aan die door deze twee correcties aantoonbaar onjuist is.

## Onnodige complexiteit die niet wordt overgenomen

- Geen gedeeld `preferences-processing.css`: haal uitsluitend de voorkeurenregels uit de
  gestashte variant en plaats ze in `static/preferences-form.css`.
- Geen bijna volledige kopie van de statische template-tekst in een routetest.
- Geen extra JavaScript om `min_satisfaction` opnieuw te prefllen; de GET-route levert de
  opgeslagen waarde al aan de template.
- Geen JavaScript-keydown-emulatie voor een `div role="button"`; gebruik een native knop.
- Geen info-popovers, tooltipmacroparameter of bijbehorende globale click-handler.
- Geen aanpassingen aan `/not_together`, verwerking, resultaat of het sociogram zelf.
- Behoud het in ADR 0007 gekozen ene formulier met verborgen editors en de bestaande
  draftopslag. Introduceer geen client-side datamodel, per-leerlingroute of nieuwe
  sessievelden.

## Relevante stash-WIP

De nieuwste stash bij het schrijven van dit plan is commit `a95fafd` met omschrijving
`accessible app: remaining later-page WIP after groups-to`, gebaseerd op `ff07649`.
Bekijk steeds de commit-id opnieuw; vertrouw niet blind op het veranderlijke nummer
`stash@{0}`.

Neem gericht over en pas aan:

- de pagina-7-hunks uit `templates/preferences_form.html`;
- alleen de modusafhankelijke terugroute van `/preferences_form` uit
  `src/aliexpress/web/routes/wizard.py`;
- alleen het eerste, voorkeurenspecifieke deel uit het ongetrackte gestashte
  `static/preferences-processing.css`, onder de nieuwe naam `static/preferences-form.css`;
- bruikbare gedragsaanpassingen uit `tests/browser/test_preferences_form_browser.py` en
  `tests/test_preferences_form_route.py`;
- uitsluitend de voorkeurenvariant uit het gestashte
  `tests/browser/test_accessible_layout_browser.py`.

Neem niet over:

- wijzigingen voor `not_together`, processing, results, tasks, progress of solver;
- de volledige statische tekstassertie `test_get_uses_approved_preferences_copy`;
- de gestashte wijzigingen van de extra-zekerheidbetekenis;
- de gestashte vervanging van `nog niet ingevuld` door `Geen voorkeuren opgegeven`;
- de redundante client-side prefill van extra zekerheid;
- pagina-8- en pagina-9-CSS uit het gemengde stylesheet.

## Bestandsscope

- `CONTEXT.md` — uitsluitend de gecorrigeerde betekenis/labels van extra zekerheid;
- `docs/adr/0003-betekenisvolle-niveaus-ipv-vrije-getallen.md`;
- `docs/adr/0007-voorkeuren-overzicht-met-bewerkmodal.md`;
- `docs/plans/toegankelijke-app/README.md`;
- `docs/plans/toegankelijke-app/07-voorkeuren-invullen.md`;
- `templates/base.html` — uitsluitend het label van de vierde voortgangsstap;
- `templates/preferences_form.html`;
- nieuw `static/preferences-form.css`;
- strikt noodzakelijke voorkeurenformulier-hunks in
  `src/aliexpress/web/routes/wizard.py`;
- `tests/test_preferences_form_route.py`;
- `tests/browser/test_preferences_form_browser.py`;
- alleen wanneer een bestaande flowassertie door de nieuwe zichtbare knoptekst breekt:
  de kleinste selector-/tekstcorrectie in `tests/browser/test_herindelen_browser.py`.

## Gedragstests

- De drie verdeelmodi tonen de juiste terugactie en volgen de juiste route.
- Een leerlingregel is met muis, Enter en Spatie te openen; de modal heeft een
  toegankelijke naam en beheert focus zoals beschreven.
- **Wijzigingen annuleren** herstelt de momentopname. Esc en backdrop behouden de invoer
  en laten de modal open.
- Tab vanuit de combobox kiest niet ongemerkt de gemarkeerde suggestie; Enter wel.
- Een deelnemende leerling en een nieuwe groep kunnen als voorkeur worden gekozen en
  behouden hun bestaande opgeslagen identiteit en gewicht.
- Een handmatig ongeldige POST geeft een normale flashmelding en bewaart de draft.
- Vanuit een toestand met een oud of leeg canoniek voorkeurenbestand opent de
  sociogramactie een nieuw tabblad dat de zojuist ingevulde voorkeur bevat. Test alleen
  deze overdracht; dupliceer geen sociogram-layouttests.
- Lege regels blijven zichtbaar als `nog niet ingevuld`; de primaire vervolgactie blijft
  een lege voorkeurenset accepteren.
- De pagina en geopende modal hebben op 1280, 390 en 320 CSS-px en bij 200% zoom geen
  horizontale overflow. Lange leerling- en groepsnamen blijven volledig leesbaar.
- Tests controleren gedrag, routes en enkele betekenisdragende labels. Dupliceer geen
  volledige tekstblokken uit de template en test geen interne JavaScriptfuncties.

## Review-, commit- en stashworkflow

1. Controleer voor iedere handeling de actuele werkboom en de commit-id van de nieuwste
   stash. Behoud bestaande wijzigingen van de eigenaar.
2. Extraheer alleen de hierboven genoemde pagina-7-hunks; pas of pop nooit de volledige
   stash.
3. Implementeer, formatteer en draai de kleinste relevante route- en browsertests. Maak
   nog geen commit en herschrijf nog geen stash.
4. Laat de eigenaar de uiteindelijke pagina in een echte browser beoordelen. Stop bij
   deze reviewpoort en verwerk feedback.
5. Maak alleen na expliciet akkoord één kleine, self-contained paginacommit, inclusief de
   ADR- en contextcorrecties.
6. Bouw daarna een nieuwe bovenste stash met omschrijving
   `accessible app: remaining later-page WIP after preferences form`. Deze opvolgstash
   bevat alleen nog WIP voor pagina 8 en later. Filter pagina-7-template-, route-, test- en
   CSS-hunks volledig uit gemengde bestanden.
7. Bewaar de oorspronkelijke stash `a95fafd` ongewijzigd als backup. Controleer de nieuwe
   actieve stash met `git stash show --stat`, `--name-status`, de derde/untracked parent en
   gerichte diffs van alle gemengde bestanden. Rapporteer aantoonbaar dat deze kleiner is.

## Acceptatie

- De goedgekeurde introductie staat zonder geel/perzikkleurig kader op de pagina.
- De belangrijke uitleg over een leerling die al in een nieuwe groep zit staat kort en
  zichtbaar bij **Graag bij**, zonder info-i.
- Extra zekerheid legt zowel het doel voor een sociaal kwetsbare leerling als het risico
  voor de totale groepsindeling uit; 100% betekent **Alle voorkeuren gehonoreerd**.
- Esc en backdrop sluiten de modal niet; expliciet annuleren werkt en verliest geen eerder
  opgeslagen invoer.
- `nog niet ingevuld` blijft een rustige aansporing, geen foutmelding.
- De sociogramactie gebruikt aantoonbaar de actuele formulierinvoer en opent in een nieuw
  tabblad.
- Validaties gebruiken de bestaande flashconventie en behouden invoer.
- Geen wijzigingen aan toekomstige pagina's, solvergedrag, gewichten of opslagbetekenis.
- Alleen bestanden uit de afgebakende scope zijn gewijzigd en de gerichte tests en
  browsercontroles zijn groen.
- De eigenaar heeft de pagina beoordeeld voordat wordt gecommit.
- Na akkoord en commit is de nieuwste actieve WIP-stash aantoonbaar kleiner en vrij van
  pagina-7-WIP; de oorspronkelijke stash blijft als backup bestaan.
