# Groepen kiezen

**Status:** afgerond na gebruikersreview

## Doel

Een leerkracht kiest de bestaande groepen uit het EDEXML-bestand die opnieuw worden
ingedeeld of waarin leerlingen volgend schooljaar starten. De pagina moet de keuze op
320 px, bij 200% zoom en met toetsenbord volledig bruikbaar houden.

## Huidige toestand

De pagina had verouderde teksten, een live teller en inline JavaScript. De twee
verdeelmodi gebruikten niet dezelfde duidelijke structuur.

## Concrete wijziging

Gebruik per modus de goedgekeurde kop, uitleg, terugactie, knop en hersteltekst. Toon de
EDEXML-groepen in één eenvoudige checklist met grote aanklikbare labels. Toon een
modusafhankelijke validatiefout als normale flashmelding. Gebruik alleen pagina-eigen CSS
en geen teller of JavaScript voor deze keuze.

## Bestandsscope

- `docs/plans/toegankelijke-app/README.md`
- `docs/plans/toegankelijke-app/04-groepen-kiezen.md`
- `templates/select_groups.html`
- `static/select-groups.css`
- strikt noodzakelijke select-groups-hunks in `src/aliexpress/web/routes/wizard.py`
- `tests/test_select_groups.py`
- gerichte browseracceptatie voor deze pagina; alleen tekstasserties in de bestaande
  herindelen-browserflow die door deze wijziging breken

## Acceptatie

- Beide modi tonen exact de goedgekeurde Nederlandstalige teksten en behouden hun routes.
- Alleen EDEXML-groepen verschijnen; er is geen teller, zoekveld, uitklapper of inline
  keuze-JavaScript.
- Een fout vraagt in een normale flashmelding om minimaal twee groepen.
- De checklist, lange groepsnamen en navigatie werken met toetsenbord, op 320 en 390 px
  en bij 200% zoom zonder horizontale overflow.
- Gerichte route-, unit- en browsertests en relevante formattingchecks zijn groen.
