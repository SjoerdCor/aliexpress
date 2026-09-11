# Leerlingen controleren

**Status:** implementatiegereed na gebruikersreview

## Doel

Een leerkracht controleert welke leerlingen aan deze groepsindeling meedoen, kan een
ontbrekende leerling toevoegen en gaat daarna naar de juiste volgende stap. De pagina
moet voor alle drie verdeelmodi begrijpelijk zijn zonder kennis van de techniek.

## Huidige toestand

`/roster` is één gedeelde pagina met drie routevarianten:

| Verdeelmodus | Vorige stap | Volgende stap |
| --- | --- | --- |
| Doorzetten | Schoolinformatie | Groepen controleren |
| Herindelen met dezelfde groepen | Groepen kiezen | Voorkeuren invullen |
| Herindelen met doorzetten | Schoolinformatie | Groepen controleren |

De pagina heet nu `Leerlingen controleren`; de gedeelde progress-strip gebruikt die naam ook.
Nieuwe leerlingen krijgen nu stilzwijgend de eerste huidige groep als geen groep is
gekozen. De server vergelijkt namen al op een volledige interne naamssleutel. De korte
displaynaam wordt pas later automatisch afgeleid; een leerkracht hoeft geen voorletters
of andere onderscheidende tekens toe te voegen.

## Concrete wijziging

- Noem de pagina en de tweede stap in de progress-strip **Leerlingen controleren**.
- Geef per verdeelmodus een korte uitleg van wat aangevinkt blijft en wanneer een leerling
  wordt uitgevinkt.
- Maak `Huidige groep` een expliciete keuze met een placeholder; `Anders` blijft de keuze
  voor een leerling die niet uit een genoemde groep komt. Valideer dit ook server-side.
- Toon bij herindelen voor een handmatig toegevoegde leerling ook `Huidige jaarlaag`.
- Behoud de bestaande opslag en routevolgorde. Gebruik per modus de terug- en verderlabels
  uit de routekaart hierboven.
- Laat invoerwaarden staan wanneer servervalidatie de pagina afwijst; voeg hiervoor geen
  sessie-opslag toe.
- Gebruik als duplicaatmelding uitsluitend:
  `Er staat al een leerling met de naam ‘{name}’ in de lijst.`
  Deze melding geldt voor dezelfde volledige voor- en achternaam na de bestaande
  normalisatie, zowel bij een bestaande als bij een nieuw toegevoegde leerling. Verschillen
  in displaynaam zijn geen reden voor de leerkracht om de naam aan te passen.
- Houd de clientcheck en servercheck gelijk in hun naamnormalisatie. De server blijft de
  bron van waarheid.
- Verwijder de live teller en overbodige editor-state uit de stash-WIP. Gebruik duidelijke
  labels, expliciete `type="button"`-attributen en toegankelijke verwijderlabels.
- Neem alleen de rosterregels uit de gestashte `setup-pages.css` over in een eigen
  `static/roster.css`; neem geen stijlen voor latere pagina's mee.

## WIP- en stashworkflow

De relevante roster-WIP wordt uit `stash@{0}` gehaald, aangepast en onderdeel van deze
pagina gemaakt. Na de gebruikersreview en het akkoord op de uiteindelijke pagina wordt de
pagina gecommit. Daarna wordt een kleinere opvolgende stash opgebouwd met alleen de nog
niet behandelde latere pagina's. De oorspronkelijke stash blijft als ongewijzigde backup
behouden; hij wordt niet gepopt of verwijderd. Omdat Git-stashes immutable zijn, wordt de
stash niet in-place aangepast.

## Bestandsscope

- `docs/plans/toegankelijke-app/README.md`
- `docs/plans/toegankelijke-app/05-leerlingen-controleren.md`
- `templates/base.html` — alleen het label van de tweede progress-stap
- `templates/roster.html`
- `static/roster.css`
- `src/aliexpress/web/routes/roster.py` — alleen noodzakelijke tekst/context
- `src/aliexpress/data/form_parsers.py` — servervalidatie van de expliciete huidige groep
- alleen roster-validatiemeldingen in `src/aliexpress/web/validation_messages.py`
- `tests/test_roster.py`
- `tests/browser/test_roster_browser.py`
- gerichte tekst- of selectorhunken in `tests/browser/test_herindelen_browser.py`

## Acceptatie

- Alle drie modi tonen de juiste uitleg, terugactie en volgende actie en behouden hun
  bestaande redirects en opslaggedrag.
- De progress-strip noemt de pagina `Leerlingen controleren`.
- Een nieuwe leerling kan niet zonder expliciete huidige groep worden bevestigd of opgeslagen.
- Dezelfde volledige naam wordt afgewezen met alleen de afgesproken eerste zin; verschillende
  volledige namen blijven toegestaan en displaynamen worden ongemoeid gelaten.
- Een validatiefout behoudt de ingevulde selecties en velden.
- De flow werkt met toetsenbord, lange namen, 320 px, 390 px en 200% zoom zonder
  horizontale overflow.
- Tests controleren gedrag en routes, niet volledige statische tekstblokken of interne
  JavaScriptdetails.
- Geen wijzigingen aan solvergedrag, opslagbetekenis of toekomstige pagina's. Na akkoord
  bevat de actieve opvolgende stash geen rosterwijzigingen meer; de oorspronkelijke stash
  blijft onaangeroerd beschikbaar als backup.
- Geen commit voordat de eigenaar de uiteindelijke pagina heeft beoordeeld en akkoord heeft
  gegeven.
