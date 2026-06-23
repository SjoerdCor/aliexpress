---
status: accepted
---

# Voorkeuren-formulier als read-only overzicht met bewerk-modal per leerling

De pagina `preferences_form` toonde elke leerling als een volledig uitgeklapt bewerk-blok
(twee comboboxes met intensiteit-pills, niet-in-groep, extra zekerheid), allemaal tegelijk
`open`. Voor een klas van tientallen leerlingen is dat overweldigend. We herontwerpen de
pagina tot een **rustig overzicht** waarin elke leerling één compacte, read-only regel is,
en het bewerken gebeurt in een **modal voor één leerling tegelijk**.

De regel toont: de korte weergavenaam, een eventuele extra-zekerheid-badge náást de naam,
en de voorkeuren als read-only chips. Een chip codeert het soort via kleur + leidend teken
(`+` graag met, `−` liever niet met, `⊘` niet-in-groep) en de intensiteit via een
achteraan-slot (`~` een beetje, niets normaal, `↑` heel graag / echt niet, `♥` hartsvriend)
**én** via een oplopende kleurverzadiging (zwak gedempt → normaal leesbaar → sterk dieper →
top diepst). Leerlingen zonder enige invoer krijgen een neutraal "nog niet ingevuld"-accent
(geen waarschuwing — leeg is een geldige eindtoestand, zie Tevredenheid in CONTEXT.md).

## Considered Options

**Bewerk-surface — modal vs aparte pagina vs inline uitklappen.** Gekozen: **modal**. Een
aparte pagina per leerling zou nieuwe routes en per-leerling-opslag vergen en het bestaande
single-form-model (één `<form>`, client-side chips, prefill uit de draft) breken. Inline
uitklappen laat de pagina "springen" en houdt twee soorten "klik om te openen" naast elkaar.
De modal houdt de bestaande opslag-/prefill-laag intact: het bewerk-blok van elke leerling
**blijft in het ene formulier staan** (verborgen) en wordt via CSS tot overlay "gepromoveerd"
— er worden geen DOM-nodes verplaatst, dus de verborgen inputs blijven gewoon meeposten.

**Korte namen — alleen weergave vs opgeslagen identiteit.** Gekozen: **alleen weergave**.
De korte unieke naam (`create_unique_name`) is cosmetisch; de opgeslagen voorkeur-`target`
blijft de volledige naam, die de canonieke identiteit is voor server-matching, het sociogram
en de integratietests. De korte naam als opgeslagen identiteit zou al die plekken raken voor
wat in de kern een weergavewens is.

**Afsluiten van de modal — twee uitkomsten.** De modal kent **"Opslaan"** (bevestigen) én een
**annuleren** via ×, Esc of backdrop-klik. Annuleren draait de wijzigingen sinds het openen
terug en sluit zonder op te slaan. (Een eerdere variant met enkel "Opslaan" — geen annuleren —
is verworpen: een echte annuleer-uitgang is voor leerkrachten de verwachte conventie, en omdat
er toch al per leerling wordt opgeslagen kan annuleren veilig terugdraaien.)

**Opslaan — automatische autosave vs expliciet bij sluiten.** Gekozen: **expliciet bij
"Opslaan"**. De gedebouncede autosave-trigger (`MutationObserver` + timer) en de aparte
"Tussentijds opslaan"-knop verdwijnen; elke modal-"Opslaan" post de draft. Omdat de
achtergrond-save het **hele** formulier post, neemt de modal bij openen een momentopname van de
bewerkbare staat; annuleren herstelt die. Zo geldt de invariant: *buiten een open modal is de
DOM gelijk aan de laatst opgeslagen draft*, zodat een geannuleerde bewerking niet alsnog
meelekt bij het opslaan van een volgende leerling.

## Consequences

- De DOM blijft even groot als voorheen (alle bewerk-blokken bestaan nog, maar verborgen).
  Dat is bewust: voor één jaargroep (~30–90 leerlingen) is het laadverschil niet voelbaar,
  en het lichtere alternatief (een client-side datamodel) zou de geteste opslaglaag
  herschrijven. Bij honderden leerlingen per scherm is dát het moment om te meten.
- Het **POST-mechanisme** voor opslaan blijft (de "Opslaan"-knop hangt eraan); alleen de
  automatische trigger gaat weg. De `beforeunload`-waarschuwing blijft als goedkoop vangnet
  voor het enige resterende verliespad: de tab sluiten middenin een open modal.
- De **validatiepoort** blijft ongewijzigd bij "Volgende → Niet samen" (daar wordt
  `voorkeuren.json` gebouwd en gevalideerd); de modal-save is best-effort draft, net als de
  oude autosave.
- Geen kleur-alleen: betekenis hangt altijd óók aan een teken/glyph of badge, consistent met
  de bestaande chip-filosofie en toegankelijkheid. De intensiteit-niveaus zelf en hun
  gewichten zijn ongewijzigd (zie ADR-0003); dit ADR legt alleen hun visuele codering vast.
- Modal-toegankelijkheid: bij openen focus op de modal-titel (niet het eerste veld, dat zou
  meteen een combobox-lijst openklappen), bij sluiten focus terug naar de aangeklikte regel,
  achtergrond `inert` (gratis focus-trap + verborgen voor schermlezers) en scroll vergrendeld.
- De browser-tests voor deze pagina (`tests/browser/`) moeten herschreven worden: de
  chips zijn read-only op het overzicht, en bewerken verloopt via de klik-op-regel → modal.
