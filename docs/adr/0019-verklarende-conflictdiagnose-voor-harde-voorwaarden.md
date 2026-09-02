---
status: accepted
---

# Verklarende conflictdiagnose voor harde voorwaarden

Dit ADR bouwt voort op ADR-0008. De bestaande familieniveaudiagnose blijft als
fallback bestaan; dit besluit voegt een specifieke, verklarende diagnose toe.

## Context

Als een indeling onmogelijk is, onderscheidt de huidige diagnose alleen de
voorwaardenfamilies `Niet in`, niet-samen en extra zekerheid. Dat helpt de gebruiker
naar het juiste invoerscherm, maar verklaart niet welke concrete voorwaarden elkaar
tegenspreken.

Een concrete wijziging adviseren is hier onwenselijk. Anders dan bij een balansgrens
heeft het versoepelen van een individuele voorwaarde een normatief gevolg voor een
leerling. Bovendien kunnen meerdere wijzigingen het probleem oplossen. Een leerling
of regel als oplossing aanwijzen zou daarom schijnzekerheid geven.

## Besluit

De diagnose geeft een **verklaring**, geen voorgeschreven reparatie. Zij toont één
concreet conflict per mislukte run als een kleine verzameling gebruikersvoorwaarden
die niet tegelijk waar kunnen zijn. Bijvoorbeeld:

- Piet mag niet in Blauw.
- Piet moet minimaal één voorkeur vervuld krijgen.
- Piets enige voorkeur is Sam.
- Sam kan alleen in Blauw.

De tekst concludeert dat deze voorwaarden niet tegelijk uitvoerbaar zijn, maar zegt
niet welke voorwaarde de gebruiker moet aanpassen.

De getoonde conflictcore moet **subset-minimaal (irreducibel)** zijn: als één getoonde
voorwaarde wordt weggelaten, vormt de resterende verzameling niet langer dezelfde
bewezen tegenspraak. We zoeken niet naar de conflictcore met absoluut het kleinste
aantal voorwaarden en ook niet naar alle mogelijke of onafhankelijke conflicten.

Bij een extra-zekerheidsvoorwaarde probeert de presentatielaag een eenvoudig gevolg van
de voorkeuren kort af te leiden, bijvoorbeeld “Piet kan 100% alleen halen als hij bij
Sam komt”. Dit gebeurt alleen wanneer de afleiding kort en ondubbelzinnig kan worden
bewezen. Anders blijft de diagnose exact maar abstracter: “Piets extra zekerheid van
50% kan binnen deze andere voorwaarden niet worden gehaald”, met de relevante
voorkeuren als context. De diagnose gokt nooit een eenvoudig klinkende oorzaak.

In de praktijk worden doorgaans niet meer dan drie positieve en één negatieve voorkeur
per leerling ingevoerd. Dat is een optimalisatie-aanname voor de begrijpelijke
presentatie, geen nieuw invoermaximum. Grotere of ingewikkeldere voorkeurensets moeten
correct naar de abstractere tekst kunnen terugvallen.

De implementatie gebruikt CP-SAT-assumptions om betekenisvolle
gebruikersvoorwaarden te labelen. Na een `INFEASIBLE`-status levert
`sufficient_assumptions_for_infeasibility()` een voldoende conflictcore. Alleen die
gevonden core wordt vervolgens door herhaalde feasibility-checks verkleind tot iedere
overgebleven voorwaarde noodzakelijk is.

De bestaande foutweergave blijft behouden. Het gestructureerde diagnoseobject wordt in
`validation_messages.py` naar één platte tekstmelding met regeleinden en opsommingstekens
vertaald. De bestaande flashmelding bewaart die opmaak al met `white-space: pre-wrap`.
Een apart conflictpaneel, nieuwe databasevelden en directe links naar invoerregels vallen
buiten de eerste versie.

De detaildiagnose krijgt een begrensd tijdsbudget. Alleen een volledig irreducibel
bewezen core mag als concrete conflictverklaring worden getoond. Als core-extractie of
-verkleining niet binnen het budget slaagt, blijft de bestaande diagnose op familieniveau
de veilige fallback. Een gedeeltelijk verkleinde core wordt niet als irreducibel
gepresenteerd. De concrete tijdslimiet is voor deze slice als eindige functieparameter
vastgelegd op 10 seconden, met een eenmalige lokale meting als onderbouwing.

Een kleine core wordt volledig uitgeschreven. Daarbij mag de formatter alleen
mechanisch groeperen, bijvoorbeeld twee uitsluitingen samenvoegen tot “Piet mag niet in
Blauw en Rood”. Er komt in de eerste versie geen catalogus die nieuwe semantische
conflicttypen probeert af te leiden.

De bullets volgen een vaste vertelvolgorde: extra zekerheid, de bijbehorende gewone
voorkeuren als ingesprongen context, niet-samen-regels en ten slotte de harde `Niet in`-
feiten. Zonder extra zekerheid begint de melding bij de niet-samen-regel. Deze volgorde
is alleen presentatie; zij kent geen schuld of aanbevolen wijziging toe.

De eerste implementatieslice toont bij iedere extra-zekerheidsvoorwaarde de gewone
voorkeuren van die leerling als ruwe context. De tekst maakt duidelijk dat deze
voorkeuren de tevredenheid bepalen, maar beweert nog niet welke voorkeur noodzakelijk
moet worden vervuld. Het afleiden van korte noodzakelijke gevolgen uit de
tevredenheidsgrens hoort bij een aparte tweede slice.

Een grote core wordt niet afgekapt en ook niet volledig uitgeschreven. De formatter
geeft dan een concrete inventaris van de betrokken invoer, bijvoorbeeld: “Het gevonden
conflict betreft niet-samen-regel 2, de extra zekerheid van Piet en Noor, en de
`Niet in`-uitsluitingen van Anna, Piet, Sam en Noor.” De precieze grens tussen klein en
groot wordt als benoemde functieparameter met default gekozen en met representatieve
voorbeelden getoetst.

## Gevolgen

- De gebruiker krijgt één behapbare verklaring per mislukte run.
- De feedback vermijdt taal als “de oorzaak” en schrijft geen benadeelde leerling of
  concrete versoepeling voor.
- Meerdere runs kunnen na opeenvolgende invoerwijzigingen verschillende conflicten
  zichtbaar maken.
- Het verkleinen van de core kost extra feasibility-solves. Er wordt niet gezocht naar
  de absoluut kleinste core, omdat dat meer rekentijd en complexiteit vraagt zonder
  overeenkomstige gebruikerswaarde.
- Een korte afgeleide uitleg bij extra zekerheid is best effort; de bewezen conflictcore
  blijft de bron van waarheid en heeft altijd een correcte fallback.
- De weblaag hoeft voor de eerste versie geen nieuw presentatie- of opslagmodel te
  krijgen; de bestaande `FeasibilityError`-context en tekstmelding worden uitgebreid.
- Een diagnose die te lang duurt verslechtert gecontroleerd naar de bestaande melding;
  de solver presenteert geen onbewezen detail als feit.

## Nog te besluiten

- Welke exact bewijsbare tevredenheidsgevolgen in de optionele tweede slice compact
  genoeg zijn om te tonen.

Voor slice 1 is de diagnose-deadline vastgesteld op 10 seconden en de grens tussen een
kleine en grote core op acht voorwaarden. Een eenmalige lokale meting op 2 september 2026 mat maximaal
0,0545 seconde en elf solves op de geteste scenario's; de deadline blijft daarom ruim
maar eindig voor grotere invoer. De deadline wordt als functieparameter aangeboden; de
worker count en seed zijn lokale reproduceerbaarheidsinstellingen. De grens van acht is
een presentatiekeuze en geen invoerlimiet.
