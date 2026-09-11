# Gedeelde toegankelijke basis

Status: Implementatiegereed na gebruikersakkoord op font en primaire knopvariant.

## Goedgekeurde basis

- Lopende tekst: lokaal gebundelde Nunito Sans.
- Koppen: lokaal gebundelde Noto Sans.
- Primaire actie: helder gevuld oranje (`#e36c00`) met witte, vetgedrukte tekst van
  20 px; de hoverkleur is donkerder oranje.
- Secundaire actie: witte knop met oranje rand en oranje tekst (`#c45100`).
- Alle interactieve elementen krijgen een zichtbare toetsenbordfocus; `[hidden]` blijft
  verborgen en reserveert geen ruimte.
- De documenttaal is Nederlands en de viewport is vastgelegd in de gedeelde layoutbasis.

## Bestandsscope

De basis staat geïsoleerd in `static/shared-base.css`, zodat de bestaande gemengde
`static/style.css` niet opnieuw als geheel hoeft te worden goedgekeurd. De stylesheet wordt
geladen door `templates/base.html` en de zelfstandige loginpagina. De lokale fontbestanden
en hun SIL Open Font License-bestanden staan in `static/fonts/`.

Deze basis wijzigt geen paginateksten, pagina-indelingen, wizardinteracties of
domeingedrag. Paginaspecifieke kleuren, componenten en layout blijven onderdeel van het
plan en de review van de betreffende pagina.

## Acceptatie voor vervolgwerk

Controleer de basis op login, homepage, Jouw groepsindelingen, één wizardpagina en
resultaat bij normale breedte, 320 px, 200% zoom en toetsenbordfocus. Voer daarna per
pagina alleen de wijzigingen uit die in het eigen plan zijn goedgekeurd.
