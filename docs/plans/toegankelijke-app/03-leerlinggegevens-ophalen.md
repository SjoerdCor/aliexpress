# Leerlinggegevens ophalen

Status: Implementatiegereed na gebruikersakkoord op inhoud, modi en afbakening.

## Doel

Maak het uploaden van EDEXML begrijpelijk voor iemand die ALI Express en EDEXML nog
niet kent, met een compacte, toegankelijke pagina voor alle drie verdeelmodi.

## Huidige toestand

De pagina gebruikt nog algemene uploadtekst, deelt WIP-stijlen met latere
setup-pagina’s en toont geen goedgekeurde modusafhankelijke hulp en knopteksten.

## Concrete wijzigingen

- Vervang de pagina-inhoud door de goedgekeurde H1, introductie, EDEXML-instructie,
  ingeklapte hulp, jaarlaagkeuzes en modusafhankelijke primaire knoppen.
- Gebruik een eigen `upload-edexml.css` voor normale flow, smalle schermen, focus en
  bedieningsruimte; wijzig `base.html`, `style.css` en de stepper niet.
- Houd de drie bestaande dispatchroutes, opslagbetekenis, foutmeldingen en redirects
  intact.
- Werk de route- en paginatests bij en voeg een gerichte browseracceptatietest toe voor
  responsiviteit en toetsenbordbediening.

## Bestandsscope

- `docs/plans/toegankelijke-app/README.md`
- `docs/plans/toegankelijke-app/03-leerlinggegevens-ophalen.md`
- `templates/upload_edexml.html`
- `static/upload-edexml.css`
- `tests/test_wizard_edexml.py`
- `tests/browser/test_upload_edexml_browser.py`
- de bestaande uploadassertie in `tests/browser/test_herindelen_browser.py`, alleen als
  die door de goedgekeurde knoptekst anders faalt

## Acceptatie

- Alle drie modi tonen exact de goedgekeurde teksten en gaan naar de juiste volgende
  stap: `/roster` of `/select_groups`.
- Alleen `.xml` wordt geaccepteerd; labels, legends, hulpteksten en externe link zijn
  toegankelijk gekoppeld.
- Upload- en validatiefouten behouden hun bestaande foutmeldingen en redirects; een
  bestandskeuze wordt niet nagebootst.
- De pagina werkt zonder horizontale overflow op 320 en 390 px, bij 200% zoom en met
  toetsenbord; gerichte tests en de drie opgegeven verificatiecommando’s slagen.
