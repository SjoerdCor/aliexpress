---
status: proposed
---

# Voorkeur-intensiteit en min. tevredenheid als vaste niveaus i.p.v. vrije getallen

In het web-formulier voor voorkeuren (`preferences_form`) voerde de leerkracht het gewicht van een voorkeur in als vrij getal en de minimale tevredenheid als vrij percentage (0–100). Beide zijn solver-concepten: het gewicht is een onbegrensde schaal zonder betekenis op zichzelf, en het percentage verwijst naar de verzadigingsintegraal van de tevredenheidsfunctie — onbegrijpelijk voor een leerkracht zonder uitleg. We vervangen ze door een kleine set **betekenisvolle, vaste niveaus**: voor "graag met" → *ook wel leuk / normaal / heel graag / hartsvriend* (gewicht 0.5 / 1 / 2 / 5), voor "liever niet met" → *liever niet / echt niet*, en voor extra zekerheid → *geen eis / minstens één voorkeur / belangrijkste voorkeur* (intern 0% / 50% / 100%). De leerkracht kiest een intentie, niet een getal.

## Considered Options

**Vrije getallen/percentages houden** (de oude situatie): maximale fijnmazigheid, maar die fijnmazigheid heeft voor een leerkracht geen betekenis en is de belangrijkste bron van verwarring en invoerfouten op deze pagina. Verworpen omdat begrijpelijkheid hier zwaarder weegt dan precisie.

**Dropdown i.p.v. zichtbare keuzes**: verbergt de schaal achter een extra klik. Verworpen ten gunste van een rij segment-knopjes (pills) die de hele schaal in één oogopslag toont, met "normaal" voorgeselecteerd (≈80% van de gevallen).

## Consequences

De vaste waarden 0.5 / 1 / 2 / 5 zijn **magische constanten** — iets wat de projectfilosofie (CLAUDE.md) normaal verbiedt. Ze zijn hier bewust en gerechtvaardigd: ze vertalen de denkcategorieën van een leerkracht ("ook wel leuk" … "hartsvriend") naar de gewogen-voorkeur-schaal die de solver al gebruikt. De keuze bakt zich in het contract tussen het formulier (`preferences_form.html`), de route die de POST verwerkt, en de tevredenheidsfunctie. Wie de niveaus of hun waarden wijzigt, raakt alle drie. De fijnmazige Excel-invoerroute blijft ongemoeid; wie wél vrije gewichten wil, gebruikt die.
