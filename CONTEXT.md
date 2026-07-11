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
Het leerjaar van een leerling (bijv. 6, 7 of 8), los van de Groep waarin die zit: één gecombineerde groep bevat meerdere jaarlagen. Bij Herindelen wordt de balans per jaarlaag over de groepen bewaakt, zodat niet de ene groep vooral jaarlaag 6 krijgt en de andere vooral 7. Rond een Overgang is "de jaarlaag" van een leerling dubbelzinnig: de invoer vraagt de Huidige jaarlaag, de resultaten tonen de Nieuwe jaarlaag.
_Avoid_: jaargroep, leerjaar, cohort

**Huidige jaarlaag**:
De Jaarlaag waarin een leerling nú zit — wat de leerkracht bij de invoer opgeeft en wat door de hele verwerking als de identiteit van de leerling geldt. Tegenhanger van de Nieuwe jaarlaag.
_Avoid_: bron-jaarlaag, oude jaarlaag

**Nieuwe jaarlaag**:
De Jaarlaag waarin een leerling ná de Overgang zit: één hoger dan de Huidige jaarlaag. De resultaten en de Excel-export tonen bij Doorzetten en Herindelen met doorzetten (beide rond een Overgang) de nieuwe jaarlaag, want dáárin zit de leerling in de nieuwe indeling. Bij Herindelen met dezelfde groepen is er geen Overgang en vallen huidige en nieuwe jaarlaag samen.
_Avoid_: doeljaarlaag, bestemmingsjaarlaag, volgende jaarlaag

**Verdeelmodus**:
Waarvoor een Proces dient: Doorzetten, Herindelen met dezelfde groepen of Herindelen met doorzetten. Gekozen bij het aanmaken van het proces (drie gelijkwaardige opties) en bepaalt de wizard-stappen en welke balans-eisen gelden.
_Avoid_: modus, type, soort

**Doorzetten**:
De verdeelmodus waarin één instromende Jaarlaag over de gecombineerde groepen van de volgende bouw wordt verdeeld; de blijvende oudere leerlingen vormen de vaste bezetting van die Bestemmingsgroepen. Eén bewegende jaarlaag.
_Avoid_: promoveren, overgang

**Overgang**:
De jaarovergang waarbij elke leerling één Jaarlaag opschuift en de oudste jaarlaag buiten de nieuwe indeling valt (verlaat de school of de bouw). Een eigenschap van de situatie, geen Verdeelmodus: zowel Doorzetten als Herindelen met doorzetten spelen zich af rond een overgang, Herindelen met dezelfde groepen niet. Kenmerk: de bron-jaarlagen verschillen van de groepslabels van de bestemming.
_Avoid_: promotie, jaarwissel, doorschuiven

**Herindelen**:
Verzamelnaam voor de twee Verdeelmodi waarin nog-aanwezige leerlingen opnieuw over een set leeg startende bestemmingsgroepen worden verdeeld, met balans per Jaarlaag: *Herindelen met dezelfde groepen* en *Herindelen met doorzetten*. Wie vertrekt doet niet mee. Geen eigen keuze in de app — je kiest altijd één van de twee concrete modi.
_Avoid_: herverdelen, herschikken, mengen

**Herindelen met dezelfde groepen**:
De Verdeelmodus waarin de huidige bewoners van de gekozen groepen opnieuw over diezelfde groepen worden verdeeld, met balans per Jaarlaag: bron en bestemming zijn dezelfde groepen. Wie vertrekt doet niet mee. Speelt zich niet af rond een Overgang.
_Avoid_: herindelen zonder doorzetten, in-place

**Herindelen met doorzetten**:
De Verdeelmodus waarin niet de huidige bewoners van de gekozen groepen worden herverdeeld, maar de leerlingen van een (doorgaans aaneengesloten) reeks Jaarlagen — bijv. 5-6-7 — over een apart gekozen set bestemmingsgroepen: de huidige groepen van 6-7-8, die leeg starten. Meerdere jaarlagen schuiven tegelijk op (zie Overgang). Backend-identiek aan Herindelen met dezelfde groepen; alleen de invoer verschilt: eerst de Jaarlagen kiezen (zoals bij Doorzetten), dan de bestemmingsgroepen handmatig. Alleen via EDEXML, want de Jaarlaag komt daaruit.
_Avoid_: overgang, doorschuiven, promotie-herindeling

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
De mate waarin de voorkeuren van een leerling in de verdeling zijn ingewilligd, op een verzadigende schaal (iedereen krijgt voorkeur 1 voor wie dan ook voorkeur 2 krijgt). Positieve voorkeuren ("graag met") lopen van 0% (geen enkele ingewilligd) tot 100% (alle ingewilligd); een leerling zonder voorkeuren geldt als volledig tevreden (100%). Een geschonden vermij-voorkeur ("liever niet met") is echter erger dan een misgelopen graag-met-voorkeur: ze maakt een leerling actief ontevreden en drukt de tevredenheid onder 0%. Een leerling met uitsluitend vermij-voorkeuren is 100% tevreden zolang hij van iedereen wordt weggehouden, maar zakt bij de eerste schending onder 0% — tot −100% als alle vermij-voorkeuren geschonden zijn. De optimalisatie maximaliseert de tevredenheid lexicografisch over de minst tevreden leerlingen, en tilt zo een leerling met een geschonden vermij-voorkeur (negatief) vóór een leerling die enkel een graag-met-voorkeur misloopt (0%).
_Avoid_: score, geluk

**Extra zekerheid**:
Een ondergrens die de leerkracht per leerling kan eisen aan diens Tevredenheid, in drie betekenisvolle niveaus: *geen eis*, *minstens één voorkeur* (de belangrijkste of een willekeurige voorkeur wordt vervuld) of *belangrijkste voorkeur* (juist de zwaarst gewogen voorkeur wordt gegarandeerd). Anders dan Tevredenheid, die de verdeling maximaliseert maar mag schenden, is een gevraagde extra zekerheid een harde eis. Te veel of te hoge eisen kunnen de verdeling onmogelijk maken.
_Avoid_: minimale tevredenheid, garantie

**Niet-in-groep-uitsluiting**:
Een harde eis dat een specifieke leerling niet in een bepaalde bestemmingsgroep geplaatst mag worden — bedoeld voor gevallen waarin het écht niet kan, met name als er al een familielid (bijv. een oudere broer of zus) in die groep zit. Hard, dus altijd gerespecteerd. Onderscheidt zich van een negatieve Voorkeur naar een groep ("liever niet naar Blauw"), die zacht is en mag wijken.
_Avoid_: niet in, verbod

**Relaxatievloer**:
De minimale Tevredenheid die de verdeling voor iedere leerling probeert te halen voordat ze de balans-eisen per groep gaat versoepelen. De balans wordt alleen zover gebroken als nodig om zoveel mogelijk leerlingen boven die vloer te krijgen, en daarna zo strak mogelijk gehouden. De vloer is strikt positieve tevredenheid: een puur-positieve leerling haalt hem met minstens één ingewilligde voorkeur, een leerling met vermij-voorkeuren alleen als geen ervan geschonden wordt.
_Avoid_: ondergrens (dat is Extra zekerheid), minimale relaxatie

**Verdeling**:
Het resultaat van de optimalisatie: de toewijzing van alle leerlingen aan groepen.
_Avoid_: indeling, uitkomst

**Sociogram**:
Een visualisatie van de relaties tussen leerlingen, afgeleid uit hun voorkeuren. Het is een tweede analyse over dezelfde invoer als de Verdeling en staat daar los van: de optimalisatie maakt een Verdeling, het sociogram maakt een relatiegrafiek. Beide zijn "wat je met de ingelezen voorkeuren doet" — peers, geen onderdeel van elkaar.
_Avoid_: grafiek, netwerk, plaatje

**Tussenstand**:
De beste kandidaat-verdeling die de optimalisatie tot nu toe heeft gevonden, getoond tijdens het rekenen: de voorlopige indeling met namen plus samenvattende aantallen per groep. Nadrukkelijk voorlopig — zowel de toewijzing als de tevredenheid kan nog veranderen (ook verbeteren) tot de Verdeling definitief is.
_Avoid_: tussenresultaat, voorlopig resultaat

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
