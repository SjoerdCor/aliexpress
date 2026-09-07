# Ontwerpuitgangspunten voor een toegankelijke ALI Express

**Status:** leidend voor nieuwe wijzigingen; geen goedkeuring van alle huidige
branchwijzigingen. De homepage en **Jouw groepsindelingen** zijn al door de gebruiker
bekeken. De overige pagina's worden pas na afzonderlijke beoordeling definitief gemaakt.

## Doelgroep en doel

ALI Express moet bruikbaar zijn voor een leerkracht of IB'er die de app en de techniek
erachter niet kent. Op iedere pagina moet die gebruiker zelfstandig kunnen begrijpen:

1. wat hier gebeurt;
2. wat die nu kan of moet doen;
3. waarom die handeling nodig is;
4. wat er na de handeling gebeurt.

De app mag enthousiast en zelfverzekerd zijn over haar kracht: een grote, onderling
afhankelijke puzzel snel en consequent doorrekenen. Die belofte wordt concreet gemaakt
met drie opbrengsten: tijdwinst, leerlingen die zo tevreden mogelijk zijn en goede,
evenwichtige groepen. Objectiviteit en het best haalbare resultaat binnen de ingevoerde
uitgangspunten ondersteunen die drie opbrengsten; het zijn geen onbeperkte garanties.

## Taal

- Spreek de gebruiker aan met **je/jij**. Gebruik **we/wij** voor begeleiding en
  **ALI Express** voor productuitleg.
- Schrijf warm, eenvoudig en professioneel. Vermijd technische termen wanneer de
  gebruiker ze niet nodig heeft om een keuze te maken.
- Gebruik consequent: **groepsindeling**, **voorkeur**, **jaarlaag**, **huidige groep**,
  **nieuwe groep**, **spreiding** en **tevredenheid**.
- Een voorkeur weegt mee. Een uitsluiting of spreidingsmaximum geldt altijd. Noem dat
  in gewone taal en gebruik alleen waar nodig het begrip *harde voorwaarde*.
- Beschrijf eerst de normale taak. Toon regels voor ongeldige invoer pas als ze relevant
  zijn, tenzij de regel nodig is om het veld überhaupt goed te kunnen invullen.
- Knoppen zijn kort en passen bij de context. Een echte handeling krijgt een werkwoord,
  zoals **Gegevens inlezen** of **Berekenen**. Op een overzicht kan **Verder →** voldoende
  zijn. Maak knopteksten niet automatisch langer om de hele volgende pagina uit te leggen.
- Schrijf foutmeldingen als hersteladvies: wat ging er mis, wat moet de gebruiker
  aanpassen en waar kan dat?

De betekenis van tevredenheid blijft nauwkeurig: de eerste vervulde voorkeur maakt veel
verschil, volgende voorkeuren voegen minder toe, het opgegeven belang telt mee en de
berekening probeert eerst de minst tevreden leerlingen te helpen. Dit wordt op de homepage
samengevat als **Iedereen zo tevreden mogelijk**.

## Informatie en indeling

- Eén duidelijke h1 benoemt de taak of uitkomst. Herhaal die niet direct in een h2.
- Zet een korte uitleg van het *waarom* vóór de handeling als die uitleg vertrouwen geeft
  of de keuze beïnvloedt. Zet voorbeelden, uitzonderingen en technische verdieping daarna
  of in een benoemde uitklapper.
- Lopende tekst mag een beperkte leesbreedte hebben. Acties, kaarten, lijsten en beelden
  volgen één herkenbaar paginaraster. Laat niet zonder reden tekst op twee derde van het
  vlak eindigen terwijl een knop, kaarten of gallery over een andere breedte lopen.
- Witruimte maakt structuur zichtbaar, maar mag een tweede geldige route niet laten
  verdwijnen. Een lege of bijna lege pagina moet nog steeds alle logische vervolgpaden
  duidelijk tonen.
- Eén primaire actie per toestand. Terug, annuleren, verwijderen en alternatieve routes
  zijn zichtbaar maar visueel secundair.
- Namen van leerlingen, groepen en groepsindelingen blijven volledig leesbaar. Laat ze
  omlopen en zet herhaalde acties op vaste plekken.

## Look-and-feel

De branch is visueel te ver van de vertrouwde app af geraakt. Het doel is geen nieuw
designsysteem, maar een vriendelijkere en beter leesbare versie van de bestaande app.

- De uitstraling blijft warm, speels en menselijk. Vermijd een rechte, zakelijke of
  overwegend grijze vormgeving.
- **Comic Sans is de referentie voor karakter en dikte, niet automatisch het technisch
  te bundelen font.** De nu gebundelde Comic Neue-fallback is nog niet geaccepteerd: hij
  oogt in de huidige toepassing te recht, dun en saai. Kies pas een definitief lokaal font
  na een vergelijking in echte appschermen op Windows en Linux. Gebruik geen externe
  fontdienst en bundel geen font zonder passende licentie.
- Het oorspronkelijke heldere oranje heeft de voorkeur boven het huidige lichte perzik.
  Witte tekst op het oude oranje had te weinig contrast. Onderzoek daarom eerst een
  verzadigd oranje met donkere tekst en duidelijke hover/focus-toestanden, in plaats van
  het hele vlak lichter en slapper te maken. De precieze kleuren zijn nog een visueel
  besluit, geen vrijbrief voor een nieuwe palette-wide wijziging.
- Knoppen en invoer zijn minstens 16 px, hebben een goed zichtbaar klikdoel en gebruiken
  onderling dezelfde basisvorm. Een terugactie die als knop hoort te werken krijgt ook
  een knopvorm.
- Iconen ondersteunen herkenning en staan naast tekst. Gebruik eenvoudige SVG-iconen in
  één stijl; geen emoji of icoon zonder toegankelijke naam als het icoon betekenis draagt.
- Behoud groen en rood waar die kleuren in voorkeuren al een betekenis hebben. Betekenis
  mag nooit alleen door kleur worden overgebracht.

## Toegankelijkheid en gedrag

- Documenttaal is Nederlands; de viewport werkt op smalle schermen.
- Tekst en interactieve toestanden voldoen minimaal aan WCAG AA-contrast. Focus is altijd
  zichtbaar en de hele flow werkt met toetsenbord.
- Geen horizontale paginascroll bij 320 CSS-px en geen verlies van informatie bij 200%
  zoom. Uitleg krijgt geen eigen scrollvak.
- Verborgen onderdelen zijn daadwerkelijk verborgen. Een modal beheert focus en geeft
  focus na sluiten terug aan de juiste actie.
- Fouten behouden ingevulde waarden. Browser-terug en zichtbare terugacties brengen de
  gebruiker naar de verwachte toestand.
- Solvergedrag, gewichten, opslagbetekenis en verdeelmodi veranderen niet als onderdeel
  van een tekst- of layoutverbetering.

## Veranderbudget per commit

Een paginacommit is een kleine verticale verbetering: de template, strikt noodzakelijke
routecode, pagina-afgebakende CSS/assets en gerichte tests. Geen opportunistische redactie
van volgende pagina's en geen brede CSS-opruiming omdat dat toevallig handig lijkt.

Gedeelde wijzigingen aan `base.html`, globale stijlen, fonts of kleurvariabelen vormen één
kleine, vooraf beoordeelde basiscommit. Daarna wordt steeds maar één pagina actief gemaakt,
beoordeeld, getest en gecommit. Zo blijft zichtbaar welke verandering welk effect heeft.
