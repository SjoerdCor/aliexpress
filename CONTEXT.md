# ALI Express

Webapplicatie waarmee basisscholen leerlingen verdelen over nieuwe groepen, op basis van voorkeuren van leerlingen en balans-eisen per groep.

## Language

**Leerling**:
Een kind dat wordt ingedeeld in een nieuwe groep.
_Avoid_: student, kind, pupil

**Groep**:
Een klas leerlingen. In een verdeling worden leerlingen vanuit hun huidige groep over bestemmingsgroepen verdeeld; "groep" is op zichzelf rolneutraal — of een groep herkomst of bestemming is, volgt uit de context, niet uit het woord.
_Avoid_: klas (ambigu tussen oud en nieuw)

**Bestemmingsgroep**:
De groep waarin een leerling door de verdeling wordt geplaatst. Een bestemmingsgroep kan al leerlingen bevatten die er blijven; de verdeling verdeelt de overige leerlingen over de bestemmingsgroepen, rekening houdend met die huidige aantallen jongens en meisjes.
_Avoid_: nieuwe groep, doelgroep

**Voorkeur**:
De wens van een leerling om wel of niet met een andere leerling in dezelfde groep te zitten.
_Avoid_: wens, keuze

**Verdeling**:
Het resultaat van de optimalisatie: de toewijzing van alle leerlingen aan groepen.
_Avoid_: indeling, uitkomst

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
