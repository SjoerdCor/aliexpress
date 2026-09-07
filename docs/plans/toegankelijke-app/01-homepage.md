# Implementatieplan — homepage

**Status:** beoordeeld, nog niet implementatiegereed. Alleen de precieze korte teksten voor
de drie opbrengsten moeten nog worden afgestemd.

## Doel van deze commit

De homepage maakt het probleem herkenbaar, laat direct zien wat ALI Express oplevert en
maakt de uitkomst concreet met drie bedienbare voorbeeldbeelden. De wijziging blijft
herkenbaar als de bestaande homepage; dit is geen nieuw totaalontwerp.

## Wat er al staat

- Merkregel, bestaande hoofdkop en een herschreven intro.
- Knop **Start een groepsindeling →** vroeg op de pagina.
- Drie opbrengstkaarten.
- Een statische gallery met drie nieuwe PNG's.
- Uitleg over tevredenheid, resultaat en opnieuw rekenen, plus twee uitklappers.
- Homepage-CSS en route-/rendertests.

Controleer dit opnieuw aan het begin van de opdracht: een lopende agent kan onderdelen
intussen hebben veranderd.

## Wat moet gebeuren

1. Laat de intro exact beginnen met:

   > Een nieuwe groepsindeling maken is altijd een enorme puzzel. Iedere leerling komt met
   > eigen voorkeuren waar we zoveel mogelijk aan willen voldoen.

   Leg daarna kort uit dat één verschuiving gevolgen heeft voor andere leerlingen en groepen.
2. Plaats **Start een groepsindeling →** direct boven **Wat ALI Express je oplevert**.
3. Gebruik één paginaraster. Lopende tekst mag een leesbreedte hebben; CTA, opbrengsten en
   gallery lijnen bewust met elkaar uit. Maak de CTA niet automatisch paginabreed.
4. Bouw de opbrengsten op rond precies drie resultaten:
   **tijdwinst**, **iedere leerling zo tevreden mogelijk** en **goede, evenwichtige groepen**
   (onder meer grootte, jongens/meisjes en zorgbehoefte). Objectief en best haalbaar mogen
   dit ondersteunen, maar worden geen vierde groot thema.
5. Vervang de kop **Ook de minst tevreden leerling telt mee** door
   **Iedereen zo tevreden mogelijk**.
6. Vervang **Daarom telt die in de berekening het zwaarst** door een zin die begint met
   **Daarom telt de eerste voorkeur ...**.
7. Maak van de gallery een handmatig bedienbare, swipebare reeks van precies drie beelden,
   zonder autoplay:

   1. voorkeuren voor één leerling invoeren;
   2. de voltooide groepsindeling met die leerling geselecteerd, inclusief tevredenheid en
      vervulde wensen;
   3. het sociogram.

   Gebruik `testdata/integration/voorkeuren.xlsx` als inhoudelijke basis en uitsluitend
   fictieve gegevens. Voeg zichtbare vorige/volgende-bediening, “1 van 3”, alttekst en
   onderschriften toe.
8. Laat de bestaande verdieping verder ongemoeid, behalve waar bovenstaande termen exact
   terugkomen.

## Open vóór uitvoering

De structuur van de opbrengsten staat vast, maar de drie korte kaartteksten nog niet. Stem
die eerst met de gebruiker af. Begin daarna pas met implementeren.

## Bestandsscope

- `templates/home.html`
- alleen `.home-*`-regels in `static/style.css`, of een eigen homepage-stylesheet
- `static/images/home-gallery-*`
- `tests/test_app.py`
- alleen een browsertest als die gallerybediening of responsive gedrag daadwerkelijk test

## Acceptatie

- De intro begint letterlijk met de goedgekeurde tekst.
- CTA en secties lijnen logisch uit op 1280 en 1366 px.
- De gallery werkt met muis, touch en toetsenbord en communiceert de actieve slide.
- De drie opbrengsten zijn zonder detailtekst te lezen direct duidelijk.
- Ingelogde en niet-ingelogde rendering en bestaande links blijven werken.
