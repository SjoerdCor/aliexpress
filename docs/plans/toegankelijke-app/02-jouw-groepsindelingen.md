# Implementatieplan — Jouw groepsindelingen

**Status:** beoordeeld, nog niet implementatiegereed. De presentatie van verwijderen moet
nog in de paginasessie worden gekozen.

## Doel van deze commit

De pagina maakt twee gelijkwaardige routes zichtbaar: verder met bestaand werk of een nieuwe
groepsindeling beginnen. Het nieuwe formulier legt kort uit waarom een herkenbare naam helpt.

## Wat er al staat

- Afzonderlijke overzichts- en nieuw-toestand.
- Hervatknoppen in een vaste actiekolom.
- Een zichtbare actie voor een nieuwe groepsindeling.
- Behoud van naam en modus na validatiefouten.
- Drie aanklikbare situatiekaarten met SVG-iconen en voorbeelden.
- Een teruglink, expliciete verwijderknop en permanente hulp met toegestane naamtekens.

De twee toestanden, kaartteksten, standaardkeuze en foutbehoud hoeven niet opnieuw te worden
ontworpen.

## Wat moet gebeuren

1. Maak op het overzicht beide routes overtuigend zichtbaar: **Verder** met een bestaande
   indeling en **Nieuwe groepsindeling**. Vul de lege ruimte niet met algemene uitleg; een
   korte introductie en duidelijke secundaire actie of kaart zijn voldoende.
2. Behoud de verbeterde uitlijning. Verkort **Verdergaan →** tot **Verder →**. De
   toegankelijke naam blijft **Verder met [naam]**.
3. Geef **← Jouw groepsindelingen** op de huidige positie een secundaire knopvorm.
4. Gebruik voor de situatiekaarten een vooruitpijl voor doorzetten, twee ronde pijlen voor
   herindelen en een herkenbare combinatie voor herindelen plus doorzetten. Tekst blijft
   de betekenis dragen.
5. Verwijder de dubbele naamuitleg. Leg één keer uit dat een herkenbare naam helpt om de
   indeling later terug te vinden en verder te gaan. Toon het voorbeeld één keer bij het veld.
6. Toon “Gebruik alleen letters ...” pas nadat daadwerkelijk een ongeldig teken is gebruikt.
   Servervalidatie blijft leidend; eventuele clientvalidatie gebruikt dezelfde regel.
7. Verander verder niets aan deze pagina.

## Open vóór uitvoering

Kies één compacte verwijderpresentatie. Een **X** is rustig maar kan ook “sluiten” betekenen;
**Verwijderen** is duidelijker maar visueel zwaarder. In beide gevallen noemt de
toegankelijke naam de groepsindeling en blijft de bevestiging vóór definitief verwijderen.

## Bestandsscope

- `templates/processes.html`
- `src/aliexpress/web/routes/processes.py`
- alleen processpecifieke regels in `static/setup-pages.css` en `static/style.css`
- `tests/test_processes.py`
- `tests/browser/test_processes_browser.py`

## Acceptatie

- Gevuld en leeg overzicht tonen duidelijk het pad naar een nieuwe indeling.
- De nieuwe toestand bevat geen dubbele uitleg.
- Alleen een ongeldig teken activeert de specifieke tekenhulp.
- Lange namen verschuiven acties niet.
- Terug, browser-terug, hervatten en verwijderen behouden hun gedrag.
