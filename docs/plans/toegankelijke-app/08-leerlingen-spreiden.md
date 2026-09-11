# Pagina 8 — Leerlingen spreiden

**Status:** implementatiegereed na gebruikersreview

## Doel

Een leerkracht of IB'er kan optioneel vastleggen welke leerlingen bewust over de groepen
in de nieuwe indeling moeten worden gespreid. De pagina legt in gewone taal uit waarom dat
nuttig kan zijn, maakt het ingestelde maximum ondubbelzinnig en houdt duidelijk dat ALI Express dit
maximum altijd respecteert.

Deze slice verandert geen solvergedrag, opslagbetekenis, verdeelmodus of vervolgstap. Een
geldige POST blijft dezelfde `not_together.json` schrijven en gaat naar de nog niet
startende verwerkingspagina.

## Huidige toestand en bruikbaar stashwerk

De huidige pagina gebruikt de technische termen `niet-samen-groep` en `Max samen`, zet de
volledige uitleg in een perzikkleurige instructiebox en toont fouten uit de
clientvalidatie met `alert()`. De bestaande expliciete cyclus **Bevestigen → Bewerken** is
wel gewenst.

De meest recente stash met omschrijving `accessible app: remaining later-page WIP after
preferences form` bevat een richting voor deze pagina, maar mag niet als geheel worden
toegepast:

- bruikbaar als vertrekpunt zijn de hertaling naar spreidingen, twee voorbeelden,
  regelnummering, toegankelijk benoemde acties, de deelnemerskiezer en de correctie van
  de terugroute bij ontbrekende voorkeureninvoer;
- de gestashte template en tests spreken elkaar op meerdere zichtbare teksten tegen;
- de nieuwe custom combobox dupliceert voorlopig bewust de bediening van de al
  beoordeelde voorkeurenpagina; maak er in deze slice geen gedeelde refactor van;
- `preferences-processing.css` vermengt stijlen voor deze pagina met de toekomstige
  verwerkingspagina; neem alleen relevante ideeën over in een pagina-eigen stylesheet;
- wijzigingen aan processing, resultaten, progress, tasks, run errors, balansvelden en
  `too_many_niet_in_form` vallen buiten deze slice;
- de gestashte tests die complete statische teksten of verborgen veldnamen controleren
  worden niet overgenomen.

Pas of pop de stash niet in zijn geheel en wijzig de stash tijdens deze implementatieronde
niet. Na gebruikersreview en een latere commit kan het verbruikte pagina-8-werk apart uit
de stash worden verwijderd, met behoud van alle latere-page-WIP.

## Tekst en presentatie

Gebruik als h1 **Leerlingen spreiden**.

Toon de intro als gewone lopende tekst, dus zonder perzikkleurige box:

> Wil je voorkomen dat deze leerlingen allemaal in dezelfde groep komen?
> Maak hieronder een spreiding. Voeg de leerlingen toe en kies hoeveel van hen er maximaal
> per groep mogen komen. ALI Express houdt zich bij de groepsindeling altijd aan
> dat maximum.
>
> Geen spreiding nodig? Dan kun je meteen verder.

Toon beide voorbeelden direct onder de intro, niet in een uitklapper. Gebruik het rustige
patroon met een oranje verticale lijn uit het appbrede ontwerp, zonder perzikkleurig vlak:

> **Twee leerlingen uit elkaar houden**
>
> Twee leerlingen leiden elkaar vaak af als ze samen zitten. Voeg hen toe en kies maximaal
> 1. Ze komen dan niet samen in één groep.
>
> **Extra ondersteuning over de groepen verdelen**
>
> Zes leerlingen hebben extra ondersteuning nodig. Voeg hen toe en kies maximaal 2. Per
> groep komen dan maximaal twee van hen.

Gebruik in iedere spreiding het doorlopende veldlabel **Maximaal [invoer] van deze
leerlingen per groep**. De toegankelijke naam van de cijferinvoer beschrijft het
volledige maximum, niet alleen `Maximaal`.

Gebruik verder deze korte acties:

- **Spreiding toevoegen**;
- **Bevestigen** en, in bevestigde toestand, **Bewerken**;
- **Spreiding verwijderen**;
- **← Terug naar Voorkeuren invullen**;
- **Verder naar Groepsindeling berekenen →**.

## Interactie en validatie

- Een nieuwe spreiding opent in bewerktoestand en krijgt een zichtbare titel zoals
  **Spreiding 1**. De knop **Spreiding toevoegen** blijft ook zichtbaar wanneer een andere
  spreiding wordt bewerkt.
- Dupliceer voor deze pagina alleen de noodzakelijke, pagina-afgebakende versie van de
  deelnemerscombobox uit `preferences_form.html`: openen bij focus, filteren tijdens
  typen, kiezen met muis of toetsenbord en alleen bekende deelnemers accepteren. Een
  gekozen optie verschijnt direct als verwijderbare chip. Raak pagina 7 niet aan en maak
  nu geen gedeeld component.
- Binnen dezelfde spreiding kan een leerling maar eenmaal worden toegevoegd. Bestaand
  gedrag waarbij een leerling in meer dan één verschillende spreiding kan staan blijft
  intact.
- Na iedere toegevoegde leerling wordt het maximum bewust opnieuw berekend als
  `max(1, floor(aantal_leerlingen / aantal_groepen_in_deze_indeling) + 1)`. Deze waarde mag een
  handmatige keuze overschrijven. Het verwijderen van een leerling verandert het maximum
  niet.
- **Bevestigen** is pas succesvol met minstens twee leerlingen en een geldig, uitvoerbaar
  maximum. Daarna verdwijnen de invoer- en chipverwijderacties, wordt het maximum alleen
  lezen en wordt **Bevestigen** vervangen door **Bewerken**. Bewerken heropent dezelfde
  waarden.
- Een lege of onvolledige spreiding wordt bij **Verder naar Groepsindeling berekenen →** niet stil verwijderd. Verwijs
  de gebruiker naar bevestigen, aanvullen of verwijderen. Zonder enige spreiding mag de
  gebruiker wel direct verder.
- Nummer kaarten en geposte velden opnieuw na het verwijderen van een hele spreiding,
  zodat meerdere resterende spreidingen correct worden opgeslagen. Dit is geen zichtbaar
  opslagmodel en wordt alleen via het opgeslagen gedrag getest.
- Vervang `alert()` door een foutbanner met dezelfde vormgeving en positie als de bestaande
  flashmeldingen. Deze mag volledig client-side zijn, krijgt `role="alert"`, wordt
  gefocust en laat alle invoer intact. Toon steeds één concreet hersteladvies en focus
  daarna waar zinvol het betreffende veld of de betreffende actie.
- Servervalidatie en Flask-flashes blijven als defensieve terugval bestaan. Herschrijf
  uitsluitend de niet-samen-validatiemeldingen naar `spreiding`, `groep` en concreet
  hersteladvies; verander de validatieregels zelf niet.
- Bereken of start de verdeling niet op deze pagina. Een geldige submit bewaart exact
  dezelfde regels en redirect naar de idle verwerkingspagina.

## Route en data

- Behoud de bestaande sessiesleutel, formulierveldstructuur, parser en JSON-betekenis.
- Sorteer namen in eerder opgeslagen spreidingen uitsluitend voor stabiele weergave; de
  setbetekenis blijft gelijk.
- Bepaal de voorkeuren-terugroute eenmaal uit `input_method.json` en gebruik dezelfde URL
  voor de zichtbare terugactie en voor herstel bij ontbrekende voorkeurenbestanden. Voeg
  geen nieuwe sessiestaat of route toe.

## Bestandsscope

Toegestaan:

- `templates/not_together.html`;
- een nieuw pagina-eigen `static/not-together.css` en het verwijderen van uitsluitend het
  daardoor overbodige niet-samen-blok uit `static/style.css`;
- strikt noodzakelijke pagina-8-code in `src/aliexpress/web/routes/wizard.py`;
- uitsluitend de niet-samen-meldingen in
  `src/aliexpress/web/validation_messages.py`;
- gerichte route-/berichttests in `tests/test_wizard_preferences.py` en zo nodig
  `tests/test_validation_messages.py`;
- een gerichte browsertest `tests/browser/test_not_together_browser.py`;
- dit plan en de statusregel in `docs/plans/toegankelijke-app/README.md`.

Niet toegestaan: processing/resultaat/sociogram, solver- of datamodelcode, parsers,
opslagbetekenis, andere wizardpagina's, gedeelde comboboxrefactors of algemene
designwijzigingen.

## Gedragsacceptatie

- De lege pagina legt taak en optionaliteit begrijpelijk uit en **Verder naar Groepsindeling berekenen →** bewaart een
  lege lijst zonder een berekening te starten.
- De deelnemerskiezer gedraagt zich voor muis en toetsenbord hetzelfde als op de
  voorkeurenpagina; onbekende of dubbele leerlingen worden niet toegevoegd.
- Toevoegen maakt een chip en herberekent het maximum; verwijderen haalt de chip weg maar
  laat het gekozen maximum staan.
- Bevestigen vergrendelt een geldige spreiding, bewerken heropent haar en verwijderen haalt
  de juiste hele spreiding weg.
- Meerdere spreidingen, inclusief verwijderen van een tussenliggende spreiding, worden met
  de juiste leerlingen en maxima opgeslagen en opnieuw getoond.
- Een onbevestigde, te kleine, ongeldige of onuitvoerbare spreiding toont een toegankelijke
  flashvormige fout met hersteladvies en behoudt alle ingevoerde waarden.
- De terugactie en de foutredirect bij ontbrekende bestanden volgen zowel de formulier- als
  de Excelroute correct.
- Bij laptopbreedte, 390 px en 320 CSS-px ontstaan geen horizontale paginascroll of
  afgeknotte namen en acties; lange namen lopen om. De hele taak werkt met alleen het
  toetsenbord en zichtbare focus.

Tests controleren deze gedragingen via zichtbaar gedrag. Ze dupliceren niet de volledige
intro of voorbeeldteksten en testen geen helperfuncties, tijdelijke CSS-klassen, verborgen
veldnummering of andere interne implementatiedetails zonder regressiewaarde.

## Verificatie en overdracht

Voer na iedere wijziging de kleinste relevante test uit. Rond de kandidaat af met ten
minste:

```bash
uv run pytest tests/test_wizard_preferences.py tests/test_validation_messages.py \
  --no-cov -n 4 --dist load
uv run pytest tests/browser/test_not_together_browser.py -q \
  --no-cov -n 4 --dist load
```

Controleer daarnaast in een echte browser laptopbreedte, 390 px en 320 px, toetsenbord,
lange leerlingnamen, geen regels, één regel en meerdere regels. Rapporteer tests en
bevindingen, maar maak geen commit. Vraag eerst de eigenaar de uiteindelijke pagina te
beoordelen; commit en gerichte stashopschoning volgen pas na expliciet akkoord.
