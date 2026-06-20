---
status: proposed
---

# Een bestemmingsgroep mag het doel van een voorkeur zijn; duplicaatregel verschilt per doeltype

Een positieve of negatieve voorkeur ("graag met" / "liever niet met") richt zich meestal op een klasgenoot, maar mag in het web-formulier ook een **bestemmingsgroep** als doel hebben. Dat dekt het geval waarin de gewenste klasgenoot al in die groep zit en dus niet zelf wordt verdeeld: we kennen die leerling niet bij naam, alleen de groep. Het datamodel stond een groep-doel al toe ([datareader.py](../../src/aliexpress/datareader.py): "a Graag met/Liever niet met target must be a known group *or* a known student"); de UI biedt het nu expliciet aan in beide richtingen, via dezelfde keuzelijst als leerlingen maar visueel onderscheiden.

De **duplicaatregel wordt afhankelijk van het doeltype**: dezelfde *leerling* mag hoogstens één keer voorkomen over "graag met" en "liever niet met" samen (twee keer is redundant of tegenstrijdig), maar dezelfde *groep* mag onbeperkt voorkomen, in beide richtingen. Voorkeuren naar een groep stapelen en strepen tegen elkaar weg — "liever niet met X uit Blauw, maar wel graag met Y uit Blauw" is een geldige netto-afweging.

## Consequences

De pandera-uniciteitscheck die nu elke dubbele `Waarde` binnen een leerling verbiedt, moet worden versoepeld tot **alleen leerling-doelen**, niet groep-doelen. De solver moet meerdere voorkeuren naar dezelfde groep optellen; dat is nog niet empirisch geverifieerd en moet worden getest vóór het wordt losgelaten — desnoods voegt de route ze tijdelijk samen. Onderscheid bewaken: "liever niet met → groep" (zacht, gewogen) is iets anders dan de [Niet-in-groep-uitsluiting](../../CONTEXT.md) (hard, voor gevallen waarin het écht niet kan, zoals een al-zittend familielid).
