---
status: proposed
---

# "Wie gaat mee" wordt een eigen, gedeelde wizardstap vóór het doorgeven van voorkeuren

Het bepalen van de populatie — welke leerlingen meegaan en welke [Verlengers](../../CONTEXT.md) worden uitgevinkt, plus het zeldzame toevoegen van een ingestroomde of versneld doorgaande leerling — wordt een **eigen wizardstap "Wie gaat mee"**, geplaatst ná "Groepen naartoe" en vóór het doorgeven van voorkeuren. De stap is **gedeeld door beide invoerroutes** (web-formulier én Excel): de populatie staat los van hóé je daarna voorkeuren invoert.

Aanleiding: in de web-formulier-route stonden populatie bepalen én voorkeuren invullen op één pagina, met "nieuwe leerling toevoegen" als prominente *Stap 1* bovenaan. Dat is scheef: een nieuwe leerling komt meestal niet voor (alleen instroom van een andere school of versneld doorgaan), terwijl de bestaande lijst al uit het bestand is ingeladen. De prominentie wekte de neiging om iederéén te gaan invoeren, en de pagina legde niet uit dát de lijst al klaarstond. De Excel-route scheidde deze twee handelingen al wél ("Stap 1: selecteer leerlingen" vóór het downloaden van het invulbestand); deze beslissing trekt beide routes gelijk.

## Considered Options

**Toevoeg-blok onderaan of gedemoveerd op één pagina houden**: minst ingrijpend, maar lost de kernvermenging (populatie vs. voorkeuren) niet op. Onderaan kan bovendien niet, want een net toegevoegde leerling moet al kiesbaar zijn als voorkeurdoel hoger op de pagina. Verworpen.

**Twee pagina's binnen dezelfde stepper-stap 3** (sub-stappen, stepper ongewijzigd): scheidt de handelingen wel, maar geeft de roster-stap geen eigen plek in de voortgangsbalk en laat de Excel-route's eigen selectie ongemoeid. Verworpen ten gunste van een volwaardige, gedeelde stap die beide routes unificeert.

**Aparte stap alleen voor de formulier-route**: minder werk, maar laat een inconsistentie staan (een stepper-stap die niet voor Excel geldt, terwijl Excel zijn eigen selectie houdt). Verworpen ten gunste van één gedeeld model.

## Consequences

- De stepper groeit van 7 naar 8 stappen; "Wie gaat mee" wordt stap 3 en alles erna schuift één op. Alle `current_step`-waarden in de templates moeten mee verschuiven.
- De keuze van invoermethode (voorkeuren via web-formulier of via Excel) verhuist van de "Groepen naartoe"-stap naar de roster-stap: je bepaalt eerst de populatie en kiest pas daarna hóé je voorkeuren invoert. De roster-stap legt `input_method.json` vast (twee verzendknoppen) en stuurt door naar de bijbehorende voorkeuren-pagina; "Groepen naartoe" heeft nog maar één doorgaan-knop.
- Er komt één canoniek roster-artifact (de aangevinkte bestaande leerlingen + de handmatig toegevoegde) dat zowel het voorkeuren-formulier als de Excel-download voedt. De bestaande `student_selection.json` (Excel) en de `going_over`-vlaggen in `preferences_form_state.json` worden hierdoor vervangen/gestroomlijnd.
- De web-formulier-pagina wordt puur voorkeuren; de Excel-pagina wordt puur download + upload. Beide lezen de afgeronde roster in.
- Het uitgebreide "af / niet-af"-kaartmodel voor nieuwe leerlingen (B5-herziening) vervalt: omdat voorkeuren nu een aparte pagina zijn die de afgeronde lijst inleest, volstaan simpele invoerregels (voornaam/achternaam/geslacht/herkomstgroep), gelijk aan wat de Excel-route al heeft. De naam-botsing- en compleetheidscontrole verhuist naar verzendtijd.
- Nieuw rand-geval: terugkeren naar "Wie gaat mee" en een leerling uitvinken kan een al ingevulde voorkeur laten bungelen. Dit wordt opgelost bij het (her)laden van de voorkeuren-pagina: bungelende voorkeuren worden opgeschoond en bovenaan vriendelijk gemeld ("Anna gaat niet meer mee — de voorkeur van Bo naar Anna is verwijderd"). De live "ongedaan maken"-balk binnen één pagina vervalt daarmee.
- De roster-pagina opent met een geruststellende uitleg die expliciet maakt dat de lijst al is ingeladen (niemand opnieuw invoeren), wat de stap doet en waarom, en dat toevoegen een zeldzame uitzondering is.
