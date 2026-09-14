# Plan: platformcompatibiliteit in CI

## Status

De implementatieslice is op 12 september 2026 lokaal uitgevoerd op
`feature/github-actions-ci` en blijft vóór commit voor review. De echte pull-requestworkflow
moet na expliciete review, commit en push nog worden uitgevoerd; matrixuitkomsten en runtimes
worden daarom pas daarna aan dit plan toegevoegd.

Dit plan vervangt voor de huidige CI-branch de verouderde matrixbeschrijving in Slice 9 van
`docs/plans/platformonafhankelijk-tdd.md`. De toen genoemde `macos-latest`-runner dekt nu
alleen ARM64 en is daarom niet voldoende voor het huidige supportcontract.

## Besluit in het kort

Behoud de bestaande volledige, resourcegesplitste suite op de primaire Pythonversie en voeg
één kleine, handmatig samengestelde compatibilitymatrix toe. Test OS en architectuur op
Python 3.13 en test de overige ondersteunde Pythonversies op Ubuntu. Maak geen cartesisch
product van ieder OS en iedere Pythonversie.

Het gewenste eindbeeld is:

```text
Bestaande jobs, tegelijk op eigen runners
├── Quality                                      Ubuntu latest / Python 3.13
├── Fast tests                                   Ubuntu latest / Python 3.13
├── Browser tests                                Ubuntu 26.04 / Python 3.13
├── Integration tests                            Ubuntu latest / Python 3.13
├── Slow acceptance test                         Ubuntu latest / Python 3.13
└── Coverage

Nieuwe compatibilitymatrix, tegelijk met bovenstaande jobs
├── Ubuntu 26.04 x64 / Python 3.13               fast + integration
├── Windows latest x64 / Python 3.13             fast + integration
├── macOS 15 Intel / Python 3.13                 fast + integration
├── macOS 15 ARM64 / Python 3.13                 fast + integration
├── Ubuntu latest x64 / Python 3.11              fast + integration
└── Ubuntu latest x64 / Python 3.12              fast + integration

Tests                                             aggregatie van alle vereiste jobs
```

Hiermee wordt iedere geclaimde OS-/architectuurvariant en iedere geclaimde Python-minor op
iedere pull request daadwerkelijk uitgevoerd, zonder de zes combinaties met elkaar of met
de bestaande jobs te serialiseren. Het bewijst bewust niet ieder mogelijk OS × Python-paar.

## Aanleiding en huidige feiten

De README noemt als eersteklas supportcontract:

- Windows x64;
- Ubuntu x64, inclusief Ubuntu 26.04;
- macOS x64 en ARM64;
- CPython 3.11, 3.12 en 3.13.

De huidige workflow draait alle jobs op `ubuntu-latest`. Volgens de actuele officiële
runnerdocumentatie is dat Ubuntu 24.04 x64. De expliciet geclaimde Ubuntu 26.04, Windows en
beide macOS-architecturen worden dus nog niet door CI bewezen.

De actuele standaardlabels die voor dit contract nodig zijn:

| Contract | Runnerlabel | Architectuur |
|---|---|---|
| Ubuntu 26.04 x64 | `ubuntu-26.04` | x64 |
| Windows x64 | `windows-latest` | x64 |
| macOS x64 | `macos-15-intel` | Intel x64 |
| macOS ARM64 | `macos-15` | Apple Silicon ARM64 |

`ubuntu-26.04` is tijdens het schrijven nog public preview. Omdat de README deze versie
expliciet als eersteklas ondersteund noemt, wordt de job wel vereist. Een structurele
runner-imagefout moet als externe CI-storing worden onderzocht en mag niet met
`continue-on-error` worden verborgen.

De huidige niet-browserselecties bestaan uit:

- 566 fast tests zonder echte solver; op niet-Windows wordt daarvan één Windows-specifieke
  test overgeslagen;
- 34 fast tests met echte solver;
- 22 niet-trage integration tests.

De stabiliteitsmeting uit `docs/plans/stabiele-parallelle-tests-ci.md` laat op de primaire
Ubuntu-runner ongeveer 17 seconden, 2 seconden en 18 seconden pytest-tijd voor deze drie
selecties zien. De bestaande slow acceptance test bepaalt met een mediaan van ruim elf
minuten de huidige kritieke doorlooptijd. De compatibilityjobs horen daar ruim onder te
blijven en draaien parallel.

Dit houdt de wandkloktijd naar verwachting vrijwel gelijk, maar niet het totale runnerwerk:
zes extra machines doen tegelijk checkout, installatie en ongeveer 37 seconden gemeten
pytest-werk. GitHub maximaliseert matrixparalleliteit afhankelijk van runnerbeschikbaarheid;
wachtrijtijd blijft dus een externe variabele. Voor publieke repositories zijn standaard
GitHub-hosted runners volgens GitHub gratis en onbeperkt, maar de extra compute en artifacts
blijven bewust zichtbaar in de meting. Bij een latere omzetting naar een private repository
moet de kostenafweging opnieuw worden gemaakt.

## Betekenis van `requires-python`

Laat in deze slice `requires-python = ">=3.11"` in `pyproject.toml` staan. Dit veld geeft aan
welke Pythonversies een installatietool als compatibel mag behandelen; het is niet de lijst
van versies waarop het project regressietests belooft. De README en de CI-matrix vormen hier
het geteste supportcontract.

Voeg ook geen bovengrens `<3.14` toe alleen om de metadata op de CI-matrix te laten lijken.
De Python Packaging User Guide raadt zulke bovengrenzen af. Python 3.14 blijft volgens de
README buiten het eersteklas supportcontract totdat een afzonderlijke wijziging dependencies
en tests daarvoor valideert.

## Niet-doelen en vaste grenzen

- Maak geen volledige OS × Python-matrix in de pull-requestworkflow.
- Installeer Playwright niet op Windows of macOS.
- Voeg compatibilitycoverage niet toe aan het bestaande gecombineerde coveragerapport.
- Draai echte CP-SAT-tests ook in de matrix nooit via xdist.
- Draai integration ook in de matrix sequentieel.
- Verander productie-`NUM_WORKERS = 8`, solvergedrag of testmarkers niet.
- Voeg geen retries, algemene timeoutverhogingen of toegestane failures toe.
- Verplaats of verwijder de slow acceptance test niet in deze wijziging.
- Voeg niet ook `actions/setup-python` toe: `astral-sh/setup-uv` ondersteunt een expliciete
  `python-version` in een matrix en de bestaande uv-cache is al per OS, architectuur en
  Pythonversie gescheiden.

## Implementatieslice — voeg de regressiepoort toe

### Bestanden

- `.github/workflows/ci.yml`
- `README.MD`
- `docs/plans/platformonafhankelijk-tdd.md`
- dit plan, uitsluitend om de uiteindelijke meetuitkomst en status vast te leggen

`pyproject.toml`, `uv.lock`, `AGENTS.md` en productiecode horen niet te wijzigen, tenzij een
echte matrixfailure een afzonderlijk afgestemde vervolgwijziging noodzakelijk maakt.

### Stap 1 — gebruik Ubuntu 26.04 voor de bestaande browserjob

Wijzig alleen `browser_tests.runs-on` van `ubuntu-latest` naar `ubuntu-26.04`. Behoud alle
bestaande testselecties, workergetallen, coverage en artifacts. Dit test Playwright 1.62 en
de volledige browsersuite op de Ubuntuversie die de README expliciet noemt, zonder een extra
browserjob of extra Chromiuminstallatie aan de kritieke route toe te voegen.

De overige bestaande testjobs blijven op `ubuntu-latest`. Daarmee blijven zowel het stabiele
Ubuntu-image als Ubuntu 26.04 vertegenwoordigd.

### Stap 2 — voeg één include-only matrixjob toe

Voeg een job met id `compatibility` en zichtbare naam
`Compatibility (${{ matrix.label }})` toe. Gebruik een include-only matrix om uitsluitend de
zes gekozen combinaties te maken:

```yaml
strategy:
  fail-fast: false
  matrix:
    include:
      - id: ubuntu-2604-py313
        label: Ubuntu 26.04 x64 / Python 3.13
        os: ubuntu-26.04
        python: "3.13"
      - id: windows-py313
        label: Windows x64 / Python 3.13
        os: windows-latest
        python: "3.13"
      - id: macos-intel-py313
        label: macOS 15 x64 / Python 3.13
        os: macos-15-intel
        python: "3.13"
      - id: macos-arm-py313
        label: macOS 15 ARM64 / Python 3.13
        os: macos-15
        python: "3.13"
      - id: ubuntu-py311
        label: Ubuntu latest x64 / Python 3.11
        os: ubuntu-latest
        python: "3.11"
      - id: ubuntu-py312
        label: Ubuntu latest x64 / Python 3.12
        os: ubuntu-latest
        python: "3.12"
```

Stel geen `max-parallel` in. GitHub mag de zes onafhankelijke jobs maximaal parallel
plannen. Gebruik `fail-fast: false`, zodat één platformfailure de diagnostiek van de andere
platformen niet annuleert. Geen combinatie krijgt `continue-on-error`: alle zes horen bij
het supportcontract.

### Stap 3 — installeer exact de gekozen Pythonversie

Gebruik na checkout de al aanwezige action, nu met de matrixversie:

```yaml
- name: Set up uv and Python
  uses: astral-sh/setup-uv@v10.1.0
  with:
    enable-cache: true
    python-version: ${{ matrix.python }}
```

Daarna blijft installatie shell-neutraal:

```yaml
- name: Install development dependencies
  run: uv sync --locked --extra dev
```

De expliciete `python-version` overschrijft alleen in deze job de `.python-version` van
3.13. `uv sync --locked` bewijst meteen dat de vastgezette dependencies voor de gekozen
runner, architectuur en Pythonversie installeerbaar zijn.

### Stap 4 — behoud de bewezen resourceprofielen

Draai in iedere matrixcombinatie de volledige fast- en integrationselecties, maar zonder
coverage. Gebruik voor de niet-solverselectie conservatief twee workers op ieder OS; deze
job is een compatibiliteitspoort, geen snelheidsbenchmark.

```yaml
- name: Run fast tests without real solver
  id: fast_non_solver
  continue-on-error: true
  run: >-
    uv run --locked pytest tests
    --ignore=tests/integration --ignore=tests/browser
    -q --no-cov -m "not slow and not real_solver" -n 2 --dist load
    --junitxml=junit-compatibility-fast-non-solver.xml

- name: Run fast tests with real solver
  id: fast_real_solver
  continue-on-error: true
  run: >-
    uv run --locked pytest tests
    --ignore=tests/integration --ignore=tests/browser
    -q --no-cov -m "not slow and real_solver" -n 0
    --junitxml=junit-compatibility-fast-real-solver.xml

- name: Run integration tests
  id: integration
  continue-on-error: true
  run: >-
    uv run --locked pytest tests/integration
    -q --no-cov -m "not slow" -n 0
    --junitxml=junit-compatibility-integration.xml
```

Gebruik `continue-on-error` alleen op deze drie tussenstappen, zodat na één failure de
overige selecties op hetzelfde platform nog diagnostiek leveren. Sluit daarna af met een
shell-neutrale resultaatcontrole. Kopieer niet het Bash-script uit de bestaande Ubuntujobs;
dat werkt niet onder de standaard PowerShell van een Windows-runner. Deze vorm werkt op alle
standaardshells omdat alleen `exit 1` aan de shell wordt doorgegeven:

```yaml
- name: Check compatibility test results
  if: >-
    always() &&
    (steps.fast_non_solver.outcome != 'success' ||
     steps.fast_real_solver.outcome != 'success' ||
     steps.integration.outcome != 'success')
  run: exit 1
```

Upload de JUnitbestanden met `if: always()` onder een unieke artifactnaam met
`${{ matrix.id }}`. Voeg geen coveragebestand en geen Playwrightartifact toe.

### Stap 5 — neem compatibility op in de stabiele aggregatiecheck

Voeg `compatibility` toe aan `needs` van de bestaande job `tests` en controleer
`${{ needs.compatibility.result }}` naast de huidige resultaten. Voor een matrixjob is dit
resultaat alleen `success` wanneer alle vereiste matrixcombinaties succesvol waren.

Behoud de zichtbare naam `Tests`. Branch protection hoeft daardoor niet naar zes vluchtige
matrixnamen te verwijzen en kan dezelfde stabiele eindcheck blijven vereisen.

### Stap 6 — werk de documentatie als onderdeel van dezelfde wijziging bij

Werk de README-sectie over testen kort bij met:

- CI voert de volledige suite uit op de primaire Pythonversie;
- de compatibilitymatrix valideert Windows x64, Ubuntu 26.04 x64, macOS x64 en macOS ARM64;
- CPython 3.11 en 3.12 worden aanvullend op Ubuntu gevalideerd;
- dit is doelbewust dimensionele dekking en geen volledig OS × Python-product.

Vervang Slice 9 in `docs/plans/platformonafhankelijk-tdd.md` door een korte statusnotitie en
een verwijzing naar dit plan. Laat daar niet de oude `macos-latest`-matrix of de afwijkende
slow-teststrategie als tweede actuele instructie staan.

Leg na de echte CI-run in dit plan vast:

- de commit en workflowattempt;
- testuitkomst per matrixcombinatie;
- totale jobduur per combinatie;
- welke test alleen op Windows uitvoerde;
- dat de CLI-subprocess-smoketest op alle vier OS-/architectuurvarianten slaagde;
- totale workflowduur tegenover de bestaande mediaan van 11m15.

## Verificatie

### Lokaal vóór review

Omdat één lokale machine de matrix niet kan bewijzen, is de lokale verificatie beperkt tot
syntax, selectie en regressies in de gewijzigde workflow:

```bash
uv run --locked pre-commit run --all-files

uv run --locked pytest tests --ignore=tests/integration --ignore=tests/browser \
  --collect-only -q --no-cov -m "not slow and not real_solver"
uv run --locked pytest tests --ignore=tests/integration --ignore=tests/browser \
  --collect-only -q --no-cov -m "not slow and real_solver"
uv run --locked pytest tests/integration \
  --collect-only -q --no-cov -m "not slow"
```

Verwachte verzameling op de huidige branch: 566 + 34 + 22 tests. Controleer daarnaast
`git diff --check` en het volledige diff. Een YAML-parser bewijst alleen syntax; hij vervangt
de echte Actions-run niet.

### Verplicht op GitHub Actions vóór de wijziging als af beschouwd wordt

Push pas na expliciete review en controleer vervolgens de pull-requestworkflow:

- alle zes matrixcombinaties bestaan en draaien zonder toegestane failures;
- de joblogs tonen de bedoelde Python-minor;
- `macos-15-intel` en `macos-15` rapporteren verschillende architecturen;
- de Windows-filelocktest draait op Windows en wordt daar niet overgeslagen;
- `test_serve_subprocess_smoke_and_clean_stop` slaagt op alle zes combinaties;
- echte solvertests en integration tonen `-n 0`;
- de browserjob draait de bestaande 127 niet-solver- en 10 real-solver-tests op
  `ubuntu-26.04`;
- JUnit-artifacts bestaan per matrixcombinatie;
- de aggregatiecheck `Tests` wordt rood als de compatibilitymatrix faalt;
- de totale workflowduur is niet betekenisvol hoger dan de bestaande slow-acceptancejob.

Voer bij voorkeur twee volledige workflowattempts op dezelfde commit uit. Gebruik geen
pytest-retry. Als een platform faalt, bewaar de eerste traceback en classificeer die als
applicatieregressie, ontbrekende wheel/dependency, runner-imageprobleem of timingprobleem
voordat iets wordt gewijzigd.

## Beslisregels bij onverwachte uitkomsten

- Als een compatibilityjob langer duurt dan de slow acceptance job, meet eerst installatie,
  fast non-solver, fast real-solver en integration apart. Verklein niet meteen de
  testdekking.
- Als alleen de niet-solverselectie onder resourcebelasting instabiel is, verlaag voor alle
  compatibilityjobs `-n 2` naar `-n 1` en herhaal de volledige matrix.
- Als een echte solvertest faalt, houd `-n 0` en onderzoek architectuur-, versie- of
  OR-Tools-gedrag; maak de test niet tolerant zonder afstemming.
- Als Ubuntu 26.04 uitsluitend door een aantoonbare preview-imagefout faalt, rapporteer de
  externe blokkade. Maak de job niet groen met `continue-on-error` zolang de README 26.04
  eersteklas noemt.
- Als macOS Intel en ARM verschillende optimale maar geldige solverrepresentanten produceren,
  behandel dat als solvergedragsvraag en lees ADR-0013 vóór een vervolgwijziging.
- Als de matrix een niet-ondersteunde dependencycombinatie vindt, pas eerst het supportcontract
  of de dependency aan in een afzonderlijke, complete slice; gebruik geen matrix-`exclude`
  om de claim stilzwijgend te omzeilen.

## Waarom nog geen volledige cartesische matrix

Een matrix van vier OS-/architectuurvarianten × drie Pythonversies zou twaalf keer dezelfde
622 niet-browsertests uitvoeren, naast de bestaande suite. Dat geeft sterkere zekerheid over
interacties tussen precies die combinaties, maar weinig extra signaal voor de eerste
regressiepoort en meer runnergebruik, wachtrijrisico en onderhoud.

De gekozen zes combinaties toetsen de twee onafhankelijke risicoassen rechtstreeks:

- OS-, filesystem-, subprocess- en architectuurgedrag op de primaire Pythonversie;
- syntax-, dependency- en runtimecompatibiliteit per ondersteunde Python-minor op Ubuntu.

Als later bewijs nodig is voor ieder OS × Python-paar, voeg dan in een afzonderlijk plan een
wekelijkse en handmatig startbare volledige matrix toe. Dat verlengt pull requests niet,
maar een scheduled workflow draait alleen vanaf de default branch en ontdekt een regressie
dus na de merge. Maak die afweging expliciet; presenteer de compacte PR-matrix niet als bewijs
voor ieder cartesisch paar.

## Definitie van klaar

- Iedere in de README genoemde OS-/architectuurvariant draait als vereiste CI-combinatie.
- CPython 3.11, 3.12 en 3.13 worden ieder door CI uitgevoerd.
- Ubuntu 26.04 installeert Chromium en draait de volledige browsersuite.
- Iedere compatibilityjob installeert vanuit `uv.lock` en verzamelt alle 622 niet-trage,
  niet-browsertests; alleen de expliciete Windows-filelocktest wordt buiten Windows
  overgeslagen.
- Echte solver- en integrationtests blijven per runner sequentieel.
- De bestaande gecombineerde coverage blijft ongewijzigd en komt alleen uit de primaire
  suite.
- `Tests` blijft de ene stabiele branch-protectioncheck en omvat compatibility.
- De gemeten totale pull-requestduur blijft in dezelfde orde als de bestaande
  slow-acceptancejob.
- README en beide oudere plannen geven geen tegenstrijdige actuele CI-instructies.
- Er zijn geen retries, algemene timeoutverhogingen, toegestane platformfailures of
  ongemotiveerde exclusions toegevoegd.

## Bronnen

- [GitHub-hosted runners](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
- [GitHub Actions matrix en paralleliteit](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#jobsjob_idstrategy)
- [uv in GitHub Actions](https://docs.astral.sh/uv/guides/integration/github/)
- [setup-uv caching](https://github.com/astral-sh/setup-uv/blob/main/docs/caching.md)
- [Python core metadata: Requires-Python](https://packaging.python.org/en/latest/specifications/core-metadata/#requires-python)
- [PyPA: vermijd bovengrenzen voor ondersteunde Pythonversies](https://packaging.python.org/en/latest/guides/dropping-older-python-versions/)

## Startprompt voor de uitvoeringssessie

```text
Lees AGENTS.md, docs/plans/stabiele-parallelle-tests-ci.md en
docs/plans/platformcompatibiliteit-in-ci.md volledig. Voer uitsluitend de implementatieslice
uit het laatste plan uit op feature/github-actions-ci. Behoud de bestaande resourcegesplitste
testlanes en coverage. Voeg de zesdelige include-only compatibilitymatrix toe, draai browser
op Ubuntu 26.04, werk de genoemde documentatie bij en verifieer de echte pull-requestworkflow.
Gebruik geen retries of toegestane failures. Raak het ongetrackte plan voor
excel-voorkeureninvoer niet aan. Rapporteer diff, testselecties, CI-uitkomsten en runtimes en
stop vóór commit voor review.
```
