---
status: ready
adr: ../adr/0019-verklarende-conflictdiagnose-voor-harde-voorwaarden.md
---

# Implementatieplan: verklarende conflictdiagnose

## Doel

Wanneer de automatische indeling onmogelijk is door een combinatie van `Niet in`,
niet-samen en extra zekerheid, toont ALI Express één concreet, bewezen conflict. De
melding legt uit welke ingevoerde voorwaarden niet tegelijk uitvoerbaar zijn en schrijft
geen wijziging voor.

Slice 1 levert de conflictcore en een begrijpelijke weergave van de betrokken invoer.
Slice 2 kan later eenvoudige noodzakelijke gevolgen van extra zekerheid afleiden.

## Bestaande situatie

- `solver/modelbuilder.py` bouwt een apart feasibility-model waarin balans zacht blijft,
  `Niet in` altijd hard is en extra zekerheid/niet-samen per familie aan of uit kunnen.
- `solver/feasibility.py` voert twee of drie leave-one-out-solves uit en retourneert
  alleen `min_satisfaction`, `not_together`, `either`, `both` of `fundamental`.
- `solver/engine.py` verpakt dit als `FeasibilityError("infeasible_preferences")`.
- `web/validation_messages.py` zet de case om naar één Nederlandse tekstmelding.
- De foutmelding wordt als gewone tekst op de bestaande processingpagina getoond. Dat
  pad ondersteunt al regeleinden en hoeft niet te worden vervangen.
- De solver werkt met genormaliseerde matching keys. De displaynamen zijn pas in
  `main.py` beschikbaar via `PreferenceData` en `GroupCounts`.

## Slice 1 — bewezen conflict uit bestaande invoer

### 1. Introduceer een klein, web-onafhankelijk diagnosecontract

Voeg in `solver/conflicts.py` immutable dataclasses toe voor:

- `ForbiddenGroup(student, group)` — één ingevoerde `Niet in`-uitsluiting;
- `MinimumSatisfaction(student, floor, preferences=...)` — één instelling van extra
  zekerheid, met de gewone voorkeuren van die leerling als verklarende context;
- `NotTogetherRule(rule_index, students, max_together)` — één volledige gebruikersregel,
  niet één interne inequality per bestemmingsgroep;
- `Conflict(conditions)` — één bewezen irreducibele core;
- een serialiseerbare omzetting naar de context van `FeasibilityError`.

De gewone voorkeuren in `MinimumSatisfaction` zijn context en tellen niet als losse
corevoorwaarden. Maak dat onderscheid ook zichtbaar in veldnamen en tests.

### 2. Bouw een feasibility-model met assumptions

Breid `solver/modelbuilder.py` uit met een diagnostic builder die dezelfde structurele
modelonderdelen gebruikt als `build_feasibility_problem`:

- precies één bestemming per leerling en alle tevredenheidsvariabelen blijven gewone
  modelstructuur;
- balans blijft volledig zacht, gelijk aan het huidige diagnosepad;
- iedere `Niet in`-rij krijgt een eigen assumption literal;
- iedere extra-zekerheidsgrens krijgt een eigen assumption literal;
- iedere niet-samen-regel krijgt één assumption literal dat alle inequalities van die
  regel activeert;
- de builder retourneert model plus een mapping van literal-index naar
  `ConflictCondition`.

Refactor de drie bestaande private constraintbouwers alleen zover nodig om dezelfde
constraintdefinities conditioneel te kunnen activeren. Houd één implementatie van de
wiskundige regels; kopieer de inequalities niet naar `feasibility.py`.

Voeg een equivalentietest toe: met alle assumptions actief moet de diagnostic builder
dezelfde feasible/infeasible-uitkomst geven als het bestaande feasibility-model met alle
drie families hard.

### 3. Extraheer en verklein één core

Voeg in `solver/feasibility.py` een functie toe die:

1. het diagnostic model oplost met alle assumptions;
2. bij `INFEASIBLE` `sufficient_assumptions_for_infeasibility()` uitleest;
3. alleen binnen die gevonden core ieder literal één voor één weglaat;
4. een literal definitief verwijdert wanneer de rest nog steeds `INFEASIBLE` is;
5. uitsluitend na volledige afronding een `Conflict` teruggeeft.

Gebruik een stabiele condition-volgorde, vaste seed en één worker voor reproduceerbare
diagnoses. Test niet dat CP-SAT bij meerdere geldige cores altijd exact dezelfde kiest;
test dat de gekozen core infeasible en irreducibel is.

Behandel `FEASIBLE`, `UNKNOWN`, `MODEL_INVALID`, een lege/onbekende core en het verlopen
van het diagnosebudget expliciet. Geen van deze gevallen mag een gedeeltelijke core als
bewezen resultaat opleveren.

### 4. Begrens de extra diagnosetijd en behoud de bestaande fallback

Geef de hele detaildiagnose één wall-clockdeadline. Iedere vervolgsolve krijgt alleen de
resterende tijd als `max_time_in_seconds`; een verlopen budget stopt de verkleining.

Laat `engine.py` eerst de detaildiagnose proberen. Bij succes komt het conflict in de
`FeasibilityError`-context. Bij timeout of technische mislukking roept het enginepad de
bestaande familieniveaudetectie aan en blijft het huidige `case`-contract intact.

Gebruik voor slice 1 een standaarddeadline van 10 seconden. Houd de deadline als
benoemde functieparameter zodat unit tests geen echte klok hoeven af te wachten.

### 5. Vertaal matching keys op de bestaande laaggrens

Vang in `main.distribute_students_from_data` alleen de nieuwe gedetailleerde
`infeasible_preferences`-context op en vertaal:

- leerlingkeys via `PreferenceData.student_display`, met `unique_name` waar dat voor
  onderscheidende korte namen geschikt is;
- groepskeys via `GroupCounts.display`;
- targets in de voorkeurcontext als leerling of groep via dezelfde twee mappings.

Laat `solver/` onafhankelijk van presentatie- en webnamen. Houd de technische
exceptiontekst naamloos, zodat `tasks._handle_failure` via `logger.exception` geen PII
naar het app-brede log schrijft. Alleen de per-process gebruikersmelding mag namen
bevatten, conform ADR-0011.

### 6. Formatteer op twee detailniveaus in de bestaande tekstmelding

Breid `_format_infeasible_preferences` in `web/validation_messages.py` uit.

Voor een kleine core:

1. extra zekerheid per leerling;
2. direct daaronder diens gewone voorkeuren als ingesprongen context;
3. betrokken niet-samen-regels, herkenbaar aan regelnummers, maximum en leden;
4. `Niet in`-uitsluitingen, mechanisch per leerling samengevoegd;
5. de neutrale conclusie dat deze voorwaarden niet tegelijk uitvoerbaar zijn en dat er
   andere conflicten kunnen bestaan.

Voor een grote core toont de formatter een concrete inventaris van betrokken
regelnummers, extra-zekerheidsleerlingen en leerlingen met `Niet in`-uitsluitingen. Hij
leidt geen nieuw capaciteits- of oorzakelijk verhaal af. Houd de grens als benoemde
functieparameter met default acht; herijk dit met voorbeelden.

Gebruik alleen platte tekst en regeleinden. Voeg geen nieuw paneel, HTML in foutcontext,
databasekolom of directe navigatielink toe.

### 7. Dek gedrag en regressies af

Voeg gerichte tests toe:

- `tests/test_modelbuilder.py`: assumption-granulariteit en equivalentie met het harde
  feasibility-model;
- `tests/test_feasibility.py`: een Piet-achtig conflict, een conflict rond één
  niet-samen-regel, een combinatie van alle drie families, één keuze uit meerdere
  onafhankelijke conflicten en bewijs dat ieder element van de geretourneerde core
  noodzakelijk is;
- `tests/test_feasibility.py`: timeout/`UNKNOWN`/ongeldige core leveren geen detail en
  activeren de familieniveaufallback;
- `tests/test_main.py`: genormaliseerde leerling- en groepskeys worden naar ingevoerde
  displaynamen vertaald, voor zowel Excel- als formulierdata;
- `tests/test_validation_messages.py`: kleine-corevolgorde, ingesprongen ruwe
  voorkeurcontext, samenvoegen van `Niet in`, niet-samen-regelnummers, grote-core-
  inventaris en de bestaande vijf family cases;
- `tests/test_wizard_distribution.py`: een achtergrondfout bereikt ongewijzigd de
  bestaande processing-/flashflow met behoud van regeleinden;
- `tests/test_pii_invariant.py`: de technische logregel bevat geen namen uit de core.

Gebruik bij meerdere mogelijke cores invariant-gebaseerde assertions. Pin alleen exacte
tekst waar de input één unieke irreducibele core heeft.

### 8. Begrensde diagnosetijd

Een eenmalige lokale meting op 2 september 2026 (één run per scenario, één worker,
deadline 5 s) gaf de volgende waarden: klein Piet/Sam 0,0115 s en 4 solves; realistische
integratiedata met geïnjecteerd conflict 0,0292 s en 4 solves; grote niet-samen-core
0,0545 s en 11 solves; meerdere mogelijke conflicten 0,0231 s en 5 solves. De grootste
gemeten core had tien voorwaarden. Daarom is de standaarddeadline voor slice 1 vastgesteld
op 10 s: ruim boven deze metingen, maar begrensd voor zwaardere invoer. De diagnose
ontvangt deze waarde als functieparameter; worker count en seed blijven lokale
reproduceerbaarheidsinstellingen. De grens van acht voorwaarden blijft de defaultparameter
voor de kleine formatteringsvariant. De meetcode is geen onderdeel van de applicatie.

### 9. Verificatievolgorde

Voer na iedere verticale stap de smalle tests uit en aan het einde:

```text
uv run pytest tests/test_modelbuilder.py tests/test_feasibility.py
uv run pytest tests/test_main.py tests/test_validation_messages.py
uv run pytest tests/test_wizard_distribution.py tests/test_pii_invariant.py
uv run pytest tests --ignore=tests/browser
uv run pytest tests/browser
uv run black --check src tests benchmarks
uv run pylint src/aliexpress
git diff --check
```

## Slice 2 — korte afleiding uit extra zekerheid

Deze slice is afzonderlijk leverbaar en blokkeert slice 1 niet.

1. Extraheer of hergebruik één publieke pure tevredenheidsfunctie, zodat diagnose en
   solver exact dezelfde scoresemantiek gebruiken.
2. Enumereer voor een begrensd klein aantal voorkeuren de wel/niet-vervulde combinaties
   en bepaal welke de ingestelde extra zekerheid halen. Bij de gebruikelijke drie
   positieve en één negatieve voorkeur zijn dat maximaal zestien combinaties.
3. Formuleer alleen gevolgen die in iedere voldoende combinatie gelden, of een zeer
   korte bewezen disjunctie zoals “minstens één van Sam en Noor”.
4. Val terug op de ruwe voorkeurcontext zodra het aantal combinaties of de natuurlijke
   formulering de ingestelde grens overschrijdt.
5. Test gelijke en ongelijke gewichten, positieve en negatieve wensen, leerling- en
   groepstargets, meerdere alternatieven en de complexiteitsfallback.

Deze analyse gebruikt uitsluitend de extra zekerheid en de eigen voorkeuren van de
leerling. Zij gebruikt de overige conflictvoorwaarden niet, zodat de uitleg niet
circulair wordt.

## Expliciet buiten scope

- vertellen welke invoer de gebruiker moet versoepelen;
- alle conflicten of alle reparatiemogelijkheden opsporen;
- een core met het absoluut kleinste aantal voorwaarden garanderen;
- nieuwe semantische patronen zoals een gereconstrueerd capaciteits- of Hall-bewijs;
- rijke HTML, nieuwe foutopslag of deep-links naar leerlingen en regels;
- harde balansinstellingen verklaren via deze core; de nieuwe diagnose geldt voor het
  automatische pad waar balans zacht is. Bestaande balanscapdiagnose blijft apart.

## Acceptatiecriteria voor slice 1

- Bij een representatief klein conflict ziet de gebruiker namen, groepen, extra
  zekerheid, voorkeurcontext en eventuele niet-samen-regels in een logische volgorde.
- Iedere als harde voorwaarde getoonde regel behoort tot één bewezen irreducibele core.
- De melding bevat geen advies over wie of wat moet wijken.
- Een grote core blijft specifiek via een concrete inventaris en wordt niet misleidend
  afgekapt.
- Timeout of technische onzekerheid geeft exact de bestaande veilige familieniveautekst.
- De normale solve, balanscapdiagnose, foutweergave en naamloze logging regresseren niet.

## Belangrijkste risico’s en beheersing

- **Diagnostic model wijkt af van het echte model:** hergebruik constraintbouwers en
  voeg equivalentietests met alle assumptions actief toe.
- **Solvercore is niet minimaal:** verklein de gevonden core volledig; noem hem niet
  irreducibel als de deadline verloopt.
- **Meerdere geldige cores geven broze tests:** test geldigheid en irreducibiliteit in
  plaats van toevallige solverkeuze; gebruik één worker voor stabielere presentatie.
- **Diagnose verlengt een mislukte run:** één totaalbudget en familieniveaufallback.
- **Voorkeurcontext lijkt zelf hard:** inspringen en expliciet formuleren als basis van de
  tevredenheidsberekening, niet als aparte corevoorwaarde.
- **Namen lekken naar blijvende logs:** houd keys/displayvertaling buiten technische
  exceptiontekst en dek de ADR-0011-invariant af.
