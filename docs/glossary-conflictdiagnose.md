# Begrippenlijst conflictdiagnose

## Conflictverklaring

Een concrete verzameling voorwaarden die niet allemaal tegelijk uitvoerbaar zijn. De
verklaring benoemt de tegenspraak, maar schrijft niet voor welke voorwaarde moet worden
aangepast.

## Gebruikersvoorwaarde

Een betekenisvolle regel uit de invoer die aan de gebruiker kan worden terugverteld,
zoals “Piet mag niet in Blauw”, “Piet heeft extra zekerheid 100%” of
“van deze leerlingen mogen er maximaal twee samen in een groep”. Interne
modelconstraints zijn geen afzonderlijke gebruikersvoorwaarden.

## Assumption

Een CP-SAT-label waarmee een gebruikersvoorwaarde voor een solve wordt ingeschakeld.
Bij een onmogelijk model kan CP-SAT een verzameling ingeschakelde labels teruggeven die
de onmogelijkheid bewijst.

## Conflictcore

Een door de solver gevonden verzameling assumptions die samen voldoende is om
onoplosbaarheid te bewijzen. Er kunnen meerdere verschillende conflictcores bestaan.

## Irreducibele conflictcore

Een conflictcore waarin iedere genoemde gebruikersvoorwaarde nodig is: na het
weglaten van één voorwaarde is de resterende verzameling niet langer onoplosbaar. Dit
heet ook subset-minimaal. Het betekent niet dat er nergens een andere conflictcore met
minder voorwaarden bestaat.

## Kleinste conflictcore

Een conflictcore met het absoluut kleinste mogelijke aantal voorwaarden. Hiernaar
zoeken valt buiten de gekozen scope.

## Reparatievoorstel

Een concrete invoerwijziging waarmee het model weer oplosbaar wordt. Voor harde
voorkeurvoorwaarden wordt dit bewust niet gegeven, omdat zo'n keuze normatief is en er
alternatieven kunnen bestaan.

## Afgeleide uitleg

Een korte vertaling van extra zekerheid en de onderliggende gewogen voorkeuren naar een
begrijpelijk gevolg, zoals “Piet moet bij Sam komen om 100% te halen”. Zij wordt alleen
getoond wanneer dat gevolg ondubbelzinnig is bewezen.

## Veilige fallback

Een minder specifieke maar nog steeds correcte uitleg die wordt gebruikt wanneer de
voorkeuren niet compact naar één natuurlijk gevolg kunnen worden vertaald. De fallback
zegt dat de vereiste tevredenheid binnen de overige genoemde voorwaarden niet haalbaar
is en verzint geen oorzaak of reparatie.

## Concrete inventaris

Een compacte opsomming van de specifieke invoerelementen in een grote conflictcore,
gegroepeerd naar niet-samen-regel, extra zekerheid en betrokken leerlingen met
`Niet in`-uitsluitingen. Zij wijst precies naar de relevante invoer, maar pretendeert
niet de volledige logische keten in natuurlijke taal te reconstrueren.
