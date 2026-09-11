# Implementatieplannen toegankelijke app

Deze map bevat één zelfstandig implementatieplan per **beoordeelde** pagina. Lees in een
werksessie alleen:

1. het [appbrede ontwerpdocument](../../toegankelijke-app-ontwerp.md);
2. dit overzicht;
3. het plan van de actieve pagina.

De werkboom bevatte bij de start 59 ongecommitte bestanden met wijzigingen voor vrijwel de
hele wizard. Die wijzigingen zijn niet automatisch goedgekeurd of klaar om samen te
committen. Bestaand werk wordt per pagina beoordeeld, behouden of bijgestuurd.

| Volgorde | Plan | Status |
| --- | --- | --- |
| 1 | [Homepage](01-homepage.md) | Vervolgverbetering implementatiegereed |
| 2 | [Jouw groepsindelingen](02-jouw-groepsindelingen.md) | Geïmplementeerd op deze branch |
| 3 | [Schoolinformatie](03-leerlinggegevens-ophalen.md) | Geïmplementeerd op deze branch |
| 4 | [Groepen kiezen](04-groepen-kiezen.md) | Geïmplementeerd op deze branch |
| 5 | [Leerlingen controleren](05-leerlingen-controleren.md) | Geïmplementeerd op deze branch |
| 6 | [Groepen controleren](06-groepen-controleren.md) | Geïmplementeerd op deze branch |
| 7 | [Voorkeuren invullen](07-voorkeuren-invullen.md) | Geïmplementeerd op deze branch |
| 8 | [Leerlingen spreiden](08-leerlingen-spreiden.md) | Geïmplementeerd op deze branch |
| 9 | [Groepsindeling berekenen en volgen](09-berekening-starten-en-volgen.md) | Geïmplementeerd op deze branch |

Resultaat, sociogram, afronding, login en beheer krijgen pas een eigen plan wanneer ze aan
de beurt zijn. De homepage-gallery mag beelden van resultaat en sociogram gebruiken, maar
verandert die pagina's niet.

Vóór de homepage wordt alleen de noodzakelijke gedeelde visuele basis vastgesteld: een
vriendelijker font en een krachtiger oranje knopvariant met voldoende contrast. De huidige
Comic Neue/perzikcombinatie is niet akkoord. De uitgangspunten en begrenzing staan in het
ontwerpdocument; dit wordt geen brede herinrichting van nog niet beoordeelde pagina's.

## Vaste werkwijze

- Werk één pagina en met één Luna xhigh-agent tegelijk af.
- Controleer vóór wijziging de actuele diff; een lopende agent kan een punt al hebben opgelost.
- Raak alleen de bestandsscope uit het actieve plan aan.
- Maak geen tekst- of ontwerpkeuzes voor nog niet beoordeelde pagina's.
- Behoud solvergedrag, opslagbetekenis, gewichten en alle drie verdeelmodi.
- Draai de gerichte tests en controleer de pagina in een echte browser op laptopbreedte,
  390 px, met toetsenbord en met lange namen.
- Lever één kleine commitkandidaat op en rapporteer wijzigingen buiten de pagina apart.
- Een plan is pas uitvoerbaar wanneer de status **Implementatiegereed** is en er geen open
  product- of tekstkeuzes meer in staan.
- Maak voor de eerstvolgende pagina pas na de gebruikersreview een nieuw, kort planbestand
  met doel, huidige toestand, concrete wijziging, bestandsscope en acceptatie.
