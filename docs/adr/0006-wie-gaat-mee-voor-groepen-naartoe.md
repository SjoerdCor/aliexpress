---
status: accepted
amends: ADR-0005
---

# "Wie gaat mee" komt vóór "Groepen naartoe"; de invoermethode-keuze schuift mee

De wizardvolgorde wordt `Schoolinformatie → Wie gaat mee → Groepen naartoe → Voorkeuren → Niet samen → …`: het bepalen van de populatie (wie meegaat) komt nu vóór het vastleggen van de beginsamenstelling van de doelgroepen. Dit draait de volgorde uit [ADR-0005](0005-wie-gaat-mee-als-eigen-gedeelde-stap.md) om, waar "Wie gaat mee" juist ná "Groepen naartoe" stond.

Aanleiding: voor een leerkracht leest "bepaal eerst wíé we verdelen, daarna het doellandschap" als een natuurlijker verhaal. De twee stappen zijn data-onafhankelijk (de overgangers en de achterblijvers zijn verschillende leerlingen), dus de volgorde is vrij te kiezen; we kiezen voor de volgorde die het verhaal vertelt.

## Consequences

- De invoermethode-keuze (voorkeuren via web-formulier of via Excel) verhuist van de roster-stap **terug naar "Groepen naartoe"**, omdat dát na de wissel de directe voorganger van de voorkeuren-stap is. Het principe van ADR-0005 — kies de invoermethode vlak vóór je hem gebruikt — blijft intact; alleen welke pagina dat is, verandert mee met de volgorde. "Groepen naartoe" legt nu `input_method.json` vast (twee verzendknoppen) en stuurt door naar de bijbehorende voorkeuren-pagina; de roster-stap krijgt één neutrale "Volgende →"-knop.
- De redirect-keten verschuift: `upload_edexml` → roster, roster → `groups_to`, `groups_to` → de gekozen voorkeuren-pagina. De `current_step`-nummers van roster (3→2) en groups_to (2→3) wisselen, evenals hun onderlinge vorige/volgende-links.
- Het bungelende-voorkeur-randgeval uit ADR-0005 blijft ongewijzigd: dat hangt aan het terugkeren naar "Wie gaat mee", niet aan de positie van de stap.

## Niet in deze ADR

De herstructurering van de voortgangsbalk (8 losse bolletjes → 3 fase-clusters Voorbereiden / Voorkeuren / Resultaat) is bewust géén ADR: het is omkeerbare presentatie zonder gevolgen voor routes of data.
