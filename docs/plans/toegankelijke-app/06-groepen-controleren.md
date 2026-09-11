# Groepen controleren

**Status:** implementatiegereed na gebruikersreview

## Doel

Een leerkracht controleert welke leerlingen al in de groepen in deze indeling zitten,
kan een groep buiten de groepsindeling laten en kan een ontbrekende lege groep toevoegen.
ALI Express gebruikt de gecontroleerde aantallen om de leerlingen daarna over zo
evenwichtig mogelijke groepen te verdelen.

Deze pagina wordt alleen getoond bij **Doorzetten**. Beide herindelmodi blijven de pagina
automatisch overslaan. Routes, verdeelmodi, solvergedrag en de betekenis van
`groups.xlsx` en `groups_to_state.json` veranderen niet.

## Huidige toestand

De pagina heeft geen gevulde h1 en zet de uitleg in een afwijkende gele instructiebox.
De tekst vraagt om aantallen terwijl de leerkracht leerlingen aan- en uitvinkt, gebruikt
de technische term *bestemming*, spreekt de gebruiker met *u* aan en toont *jaargroep*
in plaats van *jaarlaag*. De keuze tussen het aanbevolen formulier en Excel is formeler
geschreven dan de rest van de wizard.

Het JavaScript gebruikt een knop, een verborgen groepsveld en CSS met `pointer-events`
om een groep uit te zetten. Daardoor lijkt de leerlingenlijst uitgeschakeld terwijl de
checkboxes met het toetsenbord nog gewijzigd kunnen worden. De gestashte oplossing met
disabled checkboxes en gekloonde hidden inputs bewaart de vinkjes, maar maakt dit gedrag
onnodig ingewikkeld.

## Goedgekeurde tekst en interactie

- H1 en derde voortgangsstap: **Groepen controleren**.
- Intro, zonder gekleurd kader:
  **In deze groepen zitten al leerlingen die in deze indeling blijven. ALI Express telt hen mee,
  zodat de groepsindeling straks zo evenwichtig mogelijk kan worden.**
- Instructiekop: **Controleer welke leerlingen blijven**.
- Instructie:
  **De leerlingen uit de schooladministratie staan alvast aangevinkt. Laat het vinkje
  staan als een leerling in deze groep blijft. Haal het vinkje weg
  als de leerling deze groep verlaat. De aantallen bij de groep worden meteen bijgewerkt.**
- Uitleg bij groepen uitschakelen:
  **Staat een groep niet in deze groepsindeling? Zet ‘Deze groep
  gebruiken’ dan uit.**
- Gebruik per bestaande groep een native checkbox **Deze groep gebruiken**. Uitgevinkt
  betekent dat de groep niet in `group` wordt gepost. De eerder gekozen leerlingen
  blijven bewaard wanneer de groep later weer wordt aangezet.
- Toon de telling als één niet-vetgedrukte live-regio, met correcte enkelvoud/meervoud,
  bijvoorbeeld **Blijven in deze groep: 12 leerlingen · 6 jongens · 6 meisjes**.
- Toon bij een leerling de volledige naam en, wanneer beschikbaar,
  **jaarlaag {nummer}**.
- Kop toevoegactie: **Ontbreekt er een groep?**
- Uitleg toevoegactie:
  **Voeg een groep toe die nog niet in de lijst staat en in deze indeling moet komen.
  De groep begint zonder leerlingen die er al blijven.**
- Knop: **+ Lege groep toevoegen**.
- Een toegevoegde groep krijgt een duidelijke automatische naam `Nieuwe groep N`.
  Kies een nog niet gebruikte volgnummervariant, plaats de focus in het naamveld en
  maak de naam direct wijzigbaar. Het veldlabel is **Naam van de groep**.
- Kop routekeuze: **Hoe wil je de voorkeuren invullen?**
- Uitleg routekeuze:
  **We raden het formulier in ALI Express aan. Dat gaat sneller en je hoeft niets te
  downloaden. Werk je liever offline in een spreadsheet? Kies dan Excel.**
- Primaire actie, zichtbaar gemarkeerd als **Aanbevolen**:
  **Voorkeuren invullen in ALI Express →**.
- Secundaire actie: **Voorkeuren invullen via Excel →**.
- Terugactie: **← Terug naar Leerlingen controleren**.

Gebruik voor server- en clientvalidatie deze herstelteksten:

- **Kies minimaal twee groepen voor deze indeling.**
- **Geef iedere nieuwe groep een naam.**
- **Iedere groep heeft een unieke naam nodig. Pas de dubbele groepsnaam ‘{name}’ aan.**

## Concrete wijziging

- Geef de pagina een eigen wrapper, h1 en `static/groups-to.css`. Neem alleen bruikbare
  groups-to-regels uit de stash over; introduceer geen gedeeld `setup-pages.css` en pas
  geen stijlen van toekomstige pagina's aan.
- Vervang de aan/uitknop voor een bestaande groep door de goedgekeurde native checkbox.
  Gebruik minimaal JavaScript om de bijbehorende details te tonen of te verbergen; laat
  de leerlingcheckboxes succesvolle form controls blijven zodat hun toestand zonder
  extra hidden clones wordt opgeslagen.
- Behoud JavaScript alleen voor live tellen, details tonen/verbergen, een lege groep
  toevoegen/verwijderen, focusbeheer en directe toegankelijke validatie. Gebruik geen
  `alert()`, maar één zichtbare melding met `role="alert"` waarop de focus wordt gezet.
- Render de begintellingen ook server-side. Gebruik één live-regio per groep, zodat één
  wijziging niet meerdere losse aankondigingen veroorzaakt.
- Laat de server de bron van waarheid blijven. Sla de draftstaat ook op bij te weinig
  groepen, een lege groepsnaam en dubbele groepsnamen, zodat de ingevulde toestand na de
  redirect behouden blijft. Voeg hiervoor geen sessiestaat toe.
- Behoud beide bestaande submit-acties en redirects naar het formulier en Excel. Wijzig
  geen teruglinks of inhoud van de volgende pagina's.
- Beperk CSS-selectors tot deze pagina. Los de bestaande botsing met de generieke
  `.group-header` lokaal op zonder de voorkeurenpagina te redigeren.

## Relevante stash-WIP

De bron-WIP staat bij het schrijven van dit plan in de nieuwste stash met commit
`605fd58da314a46350d0ebf825147ee7ce10d568` (*accessible app: remaining later-page WIP
after roster*). Bekijk en neem alleen relevante hunks over uit:

- `templates/groups_to.html`;
- de `/groups_to`-hunk in `src/aliexpress/web/routes/wizard.py`;
- groups-to-selectors in `static/style.css` en in het afzonderlijk gestashte
  `static/setup-pages.css`;
- `tests/test_wizard_groups_to.py`;
- `tests/browser/test_groups_to_browser.py`;
- de groups-to-test uit het gestashte `tests/browser/test_accessible_layout_browser.py`.

Neem niet over:

- de bijna volledige statische tekstassertie uit `test_get_renders_approved_copy_and_existing_count`;
- de gestashte browsertestselector die niet overeenkomt met de templateknop;
- algemene `.instructions-box`-wijzigingen;
- rosterregels of wijzigingen voor voorkeuren, spreidingen, verwerking en resultaten;
- de gestashte hidden-inputconstructie voor uitgeschakelde leerlingcheckboxes.

## WIP- en stashworkflow

1. Controleer voor iedere handeling de actuele werkboom en resolveer de stash op commit-id;
   vertrouw niet blind op het veranderlijke nummer `stash@{0}`.
2. Extraheer alleen de relevante pagina-6-hunks. Pop of apply de volledige stash niet.
3. Implementeer en test de pagina, maar maak geen commit en herschrijf nog geen stash.
4. Laat de eigenaar de uiteindelijke pagina in de browser beoordelen. Stop bij deze
   reviewpoort en verwerk eventuele feedback.
5. Maak uitsluitend na expliciet akkoord de kleine, self-contained paginacommit.
6. Bouw daarna een nieuwe bovenste stash met omschrijving
   `accessible app: remaining later-page WIP after groups-to`. Deze opvolgstash bevat
   uitsluitend nog WIP voor pagina 7 en later; hij bevat geen groups-to-template,
   groups-to-tests, pagina-6-routehunks of pagina-6-CSS. Filter ook pagina-6-regels uit
   bestanden met gemengde hunks.
7. Controleer de nieuwe stash met ten minste `git stash show --stat`, `--name-only` en
   gerichte diffs van gemengde bestanden. Laat de oorspronkelijke stash als ongewijzigde
   backup bestaan; de nieuwe bovenste/actieve stash is de kleinere werkstash.

## Bestandsscope

- `docs/plans/toegankelijke-app/README.md`
- `docs/plans/toegankelijke-app/06-groepen-voor-volgend-jaar.md`
- `templates/base.html` — alleen het label van de derde voortgangsstap
- `templates/groups_to.html`
- nieuw `static/groups-to.css`
- strikt noodzakelijke `/groups_to`-hunks in `src/aliexpress/web/routes/wizard.py`
- uitsluitend de relevante groepsnaamvalidatie in
  `src/aliexpress/web/validation_messages.py`, indien hergebruik daarvan de route eenvoudiger houdt
- `tests/test_wizard_groups_to.py`
- `tests/browser/test_groups_to_browser.py`

## Gedragstests

- Routekeuzes blijven respectievelijk naar `/preferences_form` en `/preferences_excel`
  gaan en bewaren dezelfde invoermethode en groepsaantallen.
- De begintelling klopt zonder op JavaScript te wachten en verandert bij aan- of uitvinken.
- Een uitgeschakelde groep wordt niet opgeslagen, maar herstelt haar eerdere leerlingvinkjes
  wanneer zij opnieuw wordt gebruikt, ook na een serverredirect en na terugkeren.
- Een nieuwe groep krijgt een vrije automatische naam, kan worden hernoemd en verwijderd
  en wordt met nul blijvende jongens en meisjes opgeslagen.
- Alle drie validatiefouten behouden de overige invoer en geven een zichtbare,
  gefocuste herstelmelding; de server valideert hetzelfde gedrag zonder JavaScript.
- Volledige leerling- en groepsnamen blijven leesbaar. De pagina werkt met toetsenbord,
  op 1280, 390 en 320 CSS-px en bij 200% zoom zonder horizontale overflow.
- Assert niet de volledige statische tekst uit de template en test geen functienamen,
  verborgen clones of andere interne JavaScriptdetails.

## Acceptatie

- De goedgekeurde teksten en native groepscheckbox zijn zichtbaar en begrijpelijk.
- De intro staat niet in een geel kader; jongens en meisjes zijn niet onnodig vetgedrukt.
- Formulierinvoer is duidelijk aanbevolen, terwijl Excel zichtbaar beschikbaar blijft.
- Opslagbetekenis, solvergedrag, verdeelmodi en routes zijn ongewijzigd.
- Alleen bestanden uit de pagina-afgebakende scope zijn gewijzigd.
- Gerichte route-, parser- en browsertests zijn groen en de browsercontrole is uitgevoerd.
- De eigenaar heeft de uiteindelijke pagina beoordeeld voordat een commit wordt gemaakt.
- Na akkoord en commit is de nieuwste actieve WIP-stash aantoonbaar kleiner en bevat die
  uitsluitend werk voor pagina 7 en later; de oorspronkelijke stash blijft als backup bestaan.
