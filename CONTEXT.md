# ALI Express

Webapplicatie waarmee basisscholen leerlingen verdelen over nieuwe groepen, op basis van voorkeuren van leerlingen en balans-eisen per groep.

## Language

**Leerling**:
Een kind dat wordt ingedeeld in een nieuwe groep.
_Avoid_: student, kind, pupil

**Verlenger**:
Een leerling die niet meegaat in deze verdeling maar nog een jaar in dezelfde (kleuter)groep blijft. Op de stap "Wie gaat mee" staat elke leerling standaard aangevinkt; een verlenger wordt uitgevinkt en doet dan niet mee aan de verdeling. Geen apart soort leerling — een eigenschap van deze ene verdeling.
_Avoid_: blijver, zittenblijver

**Jaarlaag**:
Het leerjaar van een leerling (bijv. 6, 7 of 8), los van de Groep waarin die zit: één gecombineerde groep bevat meerdere jaarlagen. Bij Herindelen wordt de balans per jaarlaag over de groepen bewaakt, zodat niet de ene groep vooral jaarlaag 6 krijgt en de andere vooral 7.
_Avoid_: jaargroep, leerjaar, cohort

**Verdeelmodus**:
Waarvoor een Proces dient: Doorzetten of Herindelen. Gekozen bij het aanmaken van het proces en bepaalt de wizard-stappen en welke balans-eisen gelden.
_Avoid_: modus, type, soort

**Doorzetten**:
De verdeelmodus waarin één instromende Jaarlaag over de gecombineerde groepen van de volgende bouw wordt verdeeld; de blijvende oudere leerlingen vormen de vaste bezetting van die Bestemmingsgroepen. Eén bewegende jaarlaag.
_Avoid_: promoveren, overgang

**Herindelen**:
De verdeelmodus waarin alle nog-aanwezige leerlingen van een aantal bestaande groepen (één of meer jaarlagen, welke bouw dan ook) opnieuw over diezelfde groepen worden verdeeld, met balans per Jaarlaag. Bestemmingsgroepen starten leeg; wie vertrekt doet niet mee.
_Avoid_: herverdelen, herschikken, mengen

**Groep**:
Een klas leerlingen. In een verdeling worden leerlingen vanuit hun huidige groep over bestemmingsgroepen verdeeld; "groep" is op zichzelf rolneutraal — of een groep herkomst of bestemming is, volgt uit de context, niet uit het woord.
_Avoid_: klas (ambigu tussen oud en nieuw)

**Bestemmingsgroep**:
De groep waarin een leerling door de verdeling wordt geplaatst. Een bestemmingsgroep kan al leerlingen bevatten die er blijven; de verdeling verdeelt de overige leerlingen over de bestemmingsgroepen, rekening houdend met die huidige aantallen jongens en meisjes.
_Avoid_: nieuwe groep, doelgroep

**Voorkeur**:
Het verlangen van een leerling om wel of niet bij een bepaald doel in dezelfde groep te zitten. Het doel is meestal een klasgenoot, maar kan ook een bestemmingsgroep zijn — bijv. wanneer de gewenste klasgenoot al in die groep zit en dus niet zelf wordt verdeeld (we kennen die niet bij naam, alleen de groep). Een voorkeur komt vanuit de leerling en gaat als gewogen factor de tevredenheid in: de verdeling mag een voorkeur schenden als het geheel daar beter van wordt. Staat tegenover de Niet-samen-regel.
_Avoid_: wens, keuze

**Niet-samen-regel**:
Een harde eis vanuit de school/leerkracht dat hoogstens een bepaald aantal van een groepje leerlingen samen in één bestemmingsgroep komt — bijv. zorgleerlingen die samen te veel van de groep vragen, of een dynamiek die de leerkracht onwenselijk vindt. Anders dan een Voorkeur (vanuit de leerling, meegewogen) komt deze regel vanuit de school en wordt altijd gerespecteerd.
_Avoid_: harde voorkeur, blacklist

**Tevredenheid**:
De mate waarin de voorkeuren van een leerling in de verdeling zijn ingewilligd, op een verzadigende schaal (iedereen krijgt voorkeur 1 voor wie dan ook voorkeur 2 krijgt). Een leerling zonder voorkeuren geldt als volledig tevreden. De optimalisatie maximaliseert de tevredenheid lexicografisch over de minst tevreden leerlingen.
_Avoid_: score, geluk

**Extra zekerheid**:
Een ondergrens die de leerkracht per leerling kan eisen aan diens Tevredenheid, in drie betekenisvolle niveaus: *geen eis*, *minstens één voorkeur* (de belangrijkste of een willekeurige voorkeur wordt vervuld) of *belangrijkste voorkeur* (juist de zwaarst gewogen voorkeur wordt gegarandeerd). Anders dan Tevredenheid, die de verdeling maximaliseert maar mag schenden, is een gevraagde extra zekerheid een harde eis. Te veel of te hoge eisen kunnen de verdeling onmogelijk maken.
_Avoid_: minimale tevredenheid, garantie

**Niet-in-groep-uitsluiting**:
Een harde eis dat een specifieke leerling niet in een bepaalde bestemmingsgroep geplaatst mag worden — bedoeld voor gevallen waarin het écht niet kan, met name als er al een familielid (bijv. een oudere broer of zus) in die groep zit. Hard, dus altijd gerespecteerd. Onderscheidt zich van een negatieve Voorkeur naar een groep ("liever niet naar Blauw"), die zacht is en mag wijken.
_Avoid_: niet in, verbod

**Verdeling**:
Het resultaat van de optimalisatie: de toewijzing van alle leerlingen aan groepen.
_Avoid_: indeling, uitkomst

**Sociogram**:
Een visualisatie van de relaties tussen leerlingen, afgeleid uit hun voorkeuren. Het is een tweede analyse over dezelfde invoer als de Verdeling en staat daar los van: de optimalisatie maakt een Verdeling, het sociogram maakt een relatiegrafiek. Beide zijn "wat je met de ingelezen voorkeuren doet" — peers, geen onderdeel van elkaar.
_Avoid_: grafiek, netwerk, plaatje

**Proces**:
Één verdelingsrun voor een school, geïdentificeerd door naam. Bevat de invoerbestanden, status en resultaten van die run.
_Avoid_: run, sessie, taak

**School**:
De basisschool die de app gebruikt. Een school logt in met een eigen account en is eigenaar van al haar processen. De school ís de geauthenticeerde entiteit — er bestaat geen aparte gebruiker.
_Avoid_: klant, gebruiker, organisatie

**Beheerder**:
De systeembeheerder die scholen aanmaakt en namens een school in de app kan kijken (impersonatie). Een beheerder heeft geen eigen processen.
_Avoid_: admin, superuser

**Processenlijst**:
De pagina waarop een ingelogde school al haar processen ziet. Fungeert als veilig ankerpunt: bij fouten en onbekende pagina's wordt de gebruiker hierheen gestuurd.
_Avoid_: homepage, dashboard, overzicht
