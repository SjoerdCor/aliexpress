# Plan: stabiele parallelle tests in CI

## Status

Slices 1 tot en met 3 zijn geïmplementeerd op featurebranch `feature/github-actions-ci`.
De praktische stabiliteitsproef uit Slice 4 is op 12 september 2026 afgerond met vier
volledige groene CI-attempts en drie groene lokale herhalingen van zowel fast als browser.
De onderbouwing en de bewuste afwijking van de oorspronkelijk voorgestelde tien CI-runs
staan bij de uitkomst van Slice 4.

## Doel

Maak de CI sneller door veilige paralleliteit terug te brengen, zonder de huidige flakiness
te maskeren met retries of ruimere algemene timeouts.

Het gewenste eindbeeld is:

```text
GitHub Actions-jobs (ieder een eigen runner)
├── Quality
├── Fast tests
│   ├── tests zonder echte solver: parallel met xdist
│   └── tests met echte solver: sequentieel
├── Browser tests
│   ├── tests zonder echte solver: parallel met xdist
│   └── tests met echte solver: sequentieel
├── Integration tests: sequentieel
├── Slow acceptance test: sequentieel
└── Tests: aggregatie van alle vereiste testjobs
```

De jobs mogen tegelijk draaien omdat GitHub iedere job op een eigen runner uitvoert. Binnen
een runner mag nooit meer dan één echte CP-SAT-solve tegelijk actief zijn. Gewone tests, die
geen solve starten, mogen wel meerdere pytest-workers gebruiken.

## Probleem dat dit plan oplost

De parallelle configuratie is ooit gemeten op een ontwikkelmachine met 20 logische CPU's en
een kleinere suite. De oorspronkelijke meting telde 511 snelle tests en 91 browsertests. De
huidige suite telt ongeveer 600 snelle tests en 137 browsertests. De oude uitkomst staat nog
in de README:

```text
fast:    -n 6 --dist load
browser: -n 4 --dist load
```

Een standaard publieke `ubuntu-latest`-runner heeft momenteel 4 vCPU's (controleer dit bij
uitvoering opnieuw in de officiële
[GitHub-runnerdocumentatie](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)).
Iedere echte solve zet in productie `NUM_WORKERS = 8`. xdist ziet die interne
OR-Tools-threads niet. Daardoor kan de oude fast-configuratie in het slechtste geval
6 × 8 = 48 solverthreads tegelijk aanbieden en de browserconfiguratie
4 × 8 = 32 solverthreads naast vier Chromiumprocessen en vier Flask-servers.

Dit veroorzaakt geen inhoudelijk niet-deterministische oplossing: CP-SAT bewijst nog steeds
het optimum. De benodigde wandkloktijd is echter wel afhankelijk van OS-scheduling. Onder
zware oversubscriptie raken vooral Playwright- en subprocessdeadlines. Dezelfde test kan dan
de ene run binnen de deadline afronden en de andere run niet.

De eerder gevonden SQLite-race is al opgelost:

- root-`conftest.py` maakt een willekeurige SQLite-file per pytest-proces;
- iedere xdist-worker heeft een eigen Pythonproces en Flask-app;
- iedere browserworker bindt aan poort `0` en krijgt dus een eigen serverpoort;
- browseropslag en database-inhoud worden per test opnieuw opgebouwd.

Behoud deze isolatie. Bouw geen nieuwe gedeelde testdatabase en gebruik geen xdist-group als
vervanging voor ontbrekende database-isolatie.

Naast de resource-oversubscriptie bestaan twee concrete, onafhankelijke timingrisico's:

1. `test_serve_subprocess_smoke_and_clean_stop` vraagt het OS om een vrije poort, sluit de
   probesocket en start daarna pas het subprocess op die poort. Een ander proces kan de poort
   in de tussentijd innemen. De test heeft bovendien een vaste startupdeadline van tien
   seconden.
2. `test_home_gallery_automatically_advances` slaapt exact 4.500 ms en verwacht daarna een
   bepaalde slide. Een zwaar belaste browser hoeft de JavaScript-timer niet binnen dat
   venster verwerkt te hebben.

## Niet-doelen en vaste grenzen

- Verander productie-`NUM_WORKERS = 8` niet. ADR-0013 maakt dit onderdeel van de
  solverafspraak; een ander aantal kan een andere, even optimale representant kiezen.
- Voeg geen automatische pytest-retries toe.
- Voeg geen willekeurige sleeps toe en verhoog niet alle Playwright-timeouts.
- Draai `pytest tests -n <N>` niet als één gezamenlijke xdist-pool. Fast, browser en
  integration hebben verschillende resourceprofielen.
- Schrijf geen custom xdist-scheduler.
- Deel geen Playwright-page of browsercontext tussen tests.
- Maak een functionele productiewijziging alleen wanneer een test niet zonder zo'n wijziging
  betrouwbaar observeerbaar is. Motiveer dit dan vooraf bij de eigenaar.
- Behoud een volledige sequentiële opdracht als lokale eindcontrole en diagnosemogelijkheid.

## Algemene werkwijze voor alle uitvoeringssessies

Iedere Luna xhigh-sessie volgt deze regels:

1. Lees dit hele plan, `AGENTS.md` en de voor de slice genoemde bestanden.
2. Controleer `git status --short --branch`. Werk alleen op een featurebranch en raak
   wijzigingen uit andere sessies niet aan.
3. Begin vanaf de laatste door de eigenaar goedgekeurde commit. Als de vorige slice nog niet
   is goedgekeurd en gecommit, stop dan.
4. Voer alleen de eigen slice uit. Neem geen werk uit een volgende slice mee.
5. Gebruik `apply_patch` voor handmatige bestandswijzigingen.
6. Draai eerst de kleinste relevante testlane en daarna alleen de verificatie die in de slice
   staat.
7. Controleer `git diff --check` en het volledige eigen diff.
8. Rapporteer gewijzigde bestanden, testresultaten, risico's en het voorgestelde
   commitbericht.
9. Stop zonder commit. Commit uitsluitend na expliciete goedkeuring.

Als een test faalt:

- bewaar de eerste traceback en de xdist-worker waarop hij faalde;
- bepaal eerst of het een echte regressie, een timeout, een achtergebleven achtergrondtaak of
  een gedeelde-resource-race is;
- gebruik geen retry om de gate groen te krijgen;
- breid de slice niet ongemerkt uit wanneer de oplossing productiegedrag raakt.

## Slice 1 — markeer tests die de echte solver gebruiken

### Doel

Maak het resourceprofiel expliciet, zodat CI solverwerk sequentieel kan uitvoeren en alle
andere tests parallel kan blijven uitvoeren.

### Bestanden

- `pyproject.toml`
- `tests/test_feasibility.py`
- `tests/test_relaxation_floor.py`
- `tests/test_strategies.py`
- `tests/test_modelbuilder.py`
- `tests/test_balance_caps.py`
- `tests/test_sorted_weighted_slacks.py`
- `tests/browser/test_distribution_browser.py`
- `tests/browser/test_result_browser.py`
- `tests/browser/test_herindelen_browser.py`
- `tests/browser/test_roster_browser.py`
- eventueel één kleine verzamelingstest als die nodig blijkt om de selecties te bewaken

### Wijziging

Registreer in `pyproject.toml`:

```toml
markers = [
    "slow: long-running solver acceptance tests; skipped by default, pre-merge only",
    "real_solver: starts the real CP-SAT solver and must not overlap another solve on one runner",
]
```

Gebruik `real_solver`, niet het vagere `solver`: de marker onderscheidt tests die werkelijk
CP-SAT starten van tests die alleen solverdata, parsers of gemockte orchestration testen.

Markeer voor de snelle suite conservatief de drie bestanden die rechtstreeks echte solves
uitvoeren:

```python
pytestmark = pytest.mark.real_solver
```

- `tests/test_feasibility.py`
- `tests/test_relaxation_floor.py`
- `tests/test_strategies.py`

Het is acceptabel dat enkele lichte tests in die bestanden daardoor ook in de sequentiële
selectie vallen. Betrouwbare, begrijpelijke selectie gaat hier voor maximale fijnmazigheid.

Markeer daarnaast deze directe CP-SAT-controles individueel, omdat hun bestand ook tests
bevat die geen solve starten:

- `tests/test_modelbuilder.py`:
  `test_diagnostic_builder_has_one_assumption_per_user_condition`,
  `test_diagnostic_builder_matches_feasible_hard_feasibility_model`;
- `tests/test_balance_caps.py`:
  `test_build_soft_problem_caps_only_the_named_family`,
  `test_build_soft_problem_without_maxima_leaves_family_uncapped`;
- `tests/test_sorted_weighted_slacks.py`:
  `test_exact_sorted_weighted_slacks_keeps_every_valid_family_mapping`,
  `test_sorting_network_orders_values_descending`.

Markeer bij de browsertests alleen de testfuncties die een echte solve starten:

- `test_balance_limits_can_be_changed_unlimited_and_submitted`
- `test_processing_to_result_to_download`
- `test_completed_distribution_can_be_adjusted_and_run_again`
- `test_processing_stepper_completes`
- `test_result_group_cards_and_popover`
- `test_result_page_is_native_and_stays_inside_narrow_viewports`
- `test_result_adjustment_links_follow_saved_input_method_and_open_limits`
- `test_full_redistribute_flow_to_result`
- `test_redistribute_and_forward_flow_reaches_select_groups_then_next_step`
- `test_forward_route_steps_and_navigation_labels`

Controleer tijdens uitvoering opnieuw of sinds dit plan nieuwe browsertests zijn toegevoegd
die `/start_distribution` aanroepen of op een echt `/result` van een achtergrondsolve
wachten. Markeer die eveneens, maar verander geen tests die alleen `/status` mocken.

Integration blijft als volledige map sequentieel en hoeft voor de CI-selectie niet per test
gemarkeerd te worden. Voeg alleen een mapbrede marker toe als die elders aantoonbaar helpt;
voeg geen verborgen collection hook toe.

### Verificatie

Bewijs dat de twee selecties samen exact de oorspronkelijke suite vormen. Noteer de
verzamelde aantallen:

```bash
uv run pytest tests --ignore=tests/integration --ignore=tests/browser \
  --collect-only -q --no-cov -m "not slow and not real_solver"
uv run pytest tests --ignore=tests/integration --ignore=tests/browser \
  --collect-only -q --no-cov -m "not slow and real_solver"
uv run pytest tests/browser --collect-only -q --no-cov \
  -m "not slow and not real_solver"
uv run pytest tests/browser --collect-only -q --no-cov \
  -m "not slow and real_solver"
```

Draai daarna:

```bash
uv run pytest tests/test_feasibility.py tests/test_relaxation_floor.py \
  tests/test_strategies.py -q --no-cov -n 0
uv run pytest tests/browser -q --no-cov -m "real_solver" -n 0
```

Succescriteria:

- geen `PytestUnknownMarkWarning`;
- geen echte solve in de parallel bedoelde selectie;
- alle tien genoemde browsertests zitten in de `real_solver`-selectie;
- de som van beide selecties is gelijk aan de ongesplitste suite;
- solvergedrag en productiecode zijn niet gewijzigd.

### Review en commit

Voorgesteld commitbericht:

```text
test: classify real solver tests for safe scheduling
```

Stop na rapportage en wacht op goedkeuring.

## Slice 2 — verwijder intrinsieke timing- en poortraces

### Doel

Maak de twee bekende races betrouwbaar, onafhankelijk van de latere CI-indeling. Deze slice
mag geen ruimere algemene timeout of retry introduceren.

### Bestanden

- `tests/test_console_cli.py` (lichte CLI-unittests)
- `tests/integration/test_console_server_subprocess.py` (subprocess-smoketest)
- `tests/browser/test_home_browser.py`
- alleen indien een gerichte audit dezelfde foutvorm aantreft:
  `tests/browser/test_distribution_browser.py` en
  `tests/browser/test_sociogram_browser.py`

### CLI-subproces

Vervang het vooraf reserveren en weer vrijgeven van een poort door starten met `--port 0`.
Werk de test zo uit dat hij:

1. stdout van het subprocess continu leest zonder de hoofdtest onbeperkt te blokkeren;
2. wacht op de bestaande regel `Server gestart op http://127.0.0.1:<poort>`;
3. de werkelijk door Werkzeug gebonden poort uit die regel haalt;
4. precies die URL opvraagt en HTTP 200 verifieert;
5. bij een failure alle tot dan toe gelezen uitvoer toont;
6. het subprocess in `finally` altijd beëindigt;
7. na `SIGINT` nog steeds returncode 0 en `Server wordt gestopt` controleert.

Gebruik bijvoorbeeld een kleine reader-thread plus `queue.Queue`; gebruik geen vaste poort,
random poortbereik of retry na `address already in use`. Behoud een begrensde deadline om een
echt hangend subprocess te laten falen. Verhoog die deadline alleen wanneer metingen op een
solvervrije runner aantonen dat tien seconden onvoldoende is.

### Homepage-autoplay

Verwijder `page.wait_for_timeout(4500)`. Wacht rechtstreeks op het waarneembare gedrag:

```python
expect(status).to_have_text("2 van 4", timeout=6_000)
```

De test moet nog steeds bewijzen dat autoplay werkelijk plaatsvindt. Pauzeer de autoplay
niet, roep de JavaScript-implementatie niet rechtstreeks aan en maak de assertion niet
zwakker.

### Kleine audit van overige sleeps

Beoordeel de overige `wait_for_timeout`-calls. Vervang alleen sleeps die als
gereedheidsmechanisme dienen door een Playwright-assertion of `wait_for_function` op de
daadwerkelijke toestand. Behoud bewust gesimuleerde vertraging, zoals het trage gemockte
`/status`-antwoord dat sequentieel pollen test.

Waarschijnlijke kandidaten:

- native form validation na een klik: assert de validatietoestand en onveranderde URL;
- sociogramzoom: wacht tot `window.sociogramSnapshot().zoom` de verwachte grens bereikt.

Neem zo'n aanvullende vervanging alleen mee wanneer de test dezelfde timingfoutvorm heeft en
gericht verifieerbaar blijft. Maak van deze slice geen algemene browserrefactor.

### Verificatie

```bash
uv run pytest tests/test_console_cli.py -q --no-cov -n 0
uv run pytest tests/integration/test_console_server_subprocess.py -q --no-cov -n 0
uv run pytest tests/browser/test_home_browser.py -q --no-cov -n 2 --dist load
```

De subprocess-smoketest staat bewust onder `tests/integration`, zodat de fast-selectie met
`--ignore=tests/integration` uitsluitend de lichte CLI-unittests verzamelt.

Herhaal de drie gerichte opdrachten minimaal tien keer. Een shell-loop mag hiervoor in de
uitvoeringssessie worden gebruikt, maar hoort niet in de repository. Draai daarna eenmaal de
parallel bedoelde browserselectie:

```bash
uv run pytest tests/browser -q --no-cov \
  -m "not slow and not real_solver" -n 2 --dist load
```

Succescriteria:

- tien opeenvolgende groene gerichte runs;
- geen vaste-poortvenster tussen probe en subprocess-bind;
- geen slaap als vervanging voor de autoplayconditie;
- een failure toont bruikbare subprocessuitvoer;
- geen retryplugin en geen algemene timeoutverhoging.

### Review en commit

Voorgesteld commitbericht:

```text
test: remove timing races from browser and server checks
```

Stop na rapportage en wacht op goedkeuring.

## Slice 3 — splits CI op resourceprofiel

### Afhankelijkheden

Voer deze slice pas uit nadat slices 1 en 2 zijn goedgekeurd en gecommit.

### Doel

Laat onafhankelijke suites tegelijk op afzonderlijke GitHub-hosted runners draaien, terwijl
echte solves binnen een runner sequentieel blijven. Behoud één volledige gecombineerde
coverage-uitkomst en één stabiele vereiste eindcheck.

### Bestanden

- `.github/workflows/ci.yml`
- `README.MD`
- `AGENTS.md`
- `docs/plans/testsuite-versnellen.md`

Documentatie hoort in dezelfde slice als de CI-gedragswijziging; maak later geen aparte
documentatie-only commit.

### Gewenste jobs

Behoud `quality` en de bestaande sequentiële `slow-acceptance`-job. Vervang de ene
allesomvattende niet-trage testjob door drie jobs:

#### Fast tests

Installeert alleen development dependencies, geen Chromium.

Eerste opdracht, zonder echte solver:

```bash
uv run --locked pytest tests \
  --ignore=tests/integration --ignore=tests/browser \
  -q -m "not slow and not real_solver" -n 4 --dist load
```

Tweede opdracht, echte solver sequentieel:

```bash
uv run --locked pytest tests \
  --ignore=tests/integration --ignore=tests/browser \
  -q -m "not slow and real_solver" -n 0 --cov-append
```

#### Browser tests

Installeert development dependencies en Chromium met OS-dependencies.

Eerste opdracht, zonder echte solver:

```bash
uv run --locked pytest tests/browser \
  -q -m "not slow and not real_solver" -n 2 --dist load
```

Tweede opdracht, echte solver sequentieel:

```bash
uv run --locked pytest tests/browser \
  -q -m "not slow and real_solver" -n 0 --cov-append
```

#### Integration tests

Installeert alleen development dependencies, geen Chromium. Draait:

```bash
uv run --locked pytest tests/integration -q -m "not slow" -n 0
```

Geef ook deze job een uniek coverage-databestand. De huidige volledige CI-run neemt
integration mee in het gecombineerde rapport; de nieuwe indeling mag die informatie niet
stilzwijgend verliezen.

### Coverage

De bestaande volledige run maakt één gecombineerd rapport. Behoud die eigenschap voor fast,
browser en integration:

1. Geef iedere coverage producer een uniek `COVERAGE_FILE`, bijvoorbeeld `.coverage.fast`
   `.coverage.browser` en `.coverage.integration`.
2. Gebruik in fast en browser bij de tweede opdracht van dezelfde job `--cov-append`.
3. Upload het coverage-databestand als artifact.
4. Voeg een kleine `coverage`-job toe die na fast, browser en integration draait, de
   artifacts downloadt, `uv run coverage combine`, `uv run coverage report` en
   `uv run coverage xml` uitvoert.

Gebruik unieke artifactnamen. Maak de coverage-job afhankelijk van alle producers en laat
hem falen wanneer een verwacht artifact ontbreekt. Voeg geen coveragepercentage-drempel toe
als afzonderlijke, ongemotiveerde beleidswijziging.

### Diagnostische artifacts

Laat iedere testopdracht een uniek JUnit-bestand schrijven. Upload JUnit altijd, ook bij een
failure. Laat de browserjob bij failures bovendien Playwright-output uploaden. Activeer alleen
`retain-on-failure`-achtige diagnostiek; permanente video/tracing van alle groene tests is
niet nodig.

Artifacts zijn diagnosemiddelen, geen retrymechanisme.

### Aggregatiecheck

Voeg een lichte job met zichtbare naam `Tests` toe die `needs` gebruikt voor:

- fast tests;
- browser tests;
- integration tests;
- slow acceptance;
- coverage, als coverage een vereiste gate blijft.

Gebruik `if: always()` en controleer expliciet dat iedere benodigde job `success` was. Zo kan
branch protection één stabiele checknaam `Tests` blijven vereisen terwijl de onderliggende
jobs apart worden uitgevoerd. De aggregatiejob mag een mislukte of geannuleerde dependency
niet groen vertalen.

### Workerkeuze

Start met 4 fast-workers en 2 browserworkers, uitsluitend voor selecties zonder echte solver.
Deze aantallen passen bij een publieke `ubuntu-latest`-runner met 4 vCPU's en voorkomen dat
vier Chromiums om dezelfde CPU's strijden.

Verander deze aantallen alleen op basis van drie volledige groene metingen per kandidaat:

- fast: vergelijk 2 en 4 workers;
- browser: vergelijk 1 en 2 workers;
- kies minder workers bij minder dan 5% verschil in mediaan;
- behoud `--dist load` bij gelijkwaardige schedulers;
- test `worksteal` alleen als `load` aantoonbaar een grote staart door ongelijke testduur
  heeft.

Meet nooit workergetallen terwijl echte solves in dezelfde xdist-selectie zitten.

### Documentatie

Werk in dezelfde slice bij:

- `README.MD`: maak onderscheid tussen lokale snelle feedback, CI-profiel en volledige
  sequentiële eindcontrole;
- `AGENTS.md`: laat de baselinecommando's overeenkomen met de daadwerkelijk gekozen veilige
  selecties;
- `docs/plans/testsuite-versnellen.md`: voeg een gedateerde correctie toe dat de oude 6/4-
  meting alleen gold voor de toenmalige 20-CPU-machine en niet als universeel CI-profiel mag
  worden gebruikt.

Documenteer expliciet:

- echte solvertests draaien per machine sequentieel;
- fast en browser mogen onderling als CI-jobs tegelijk draaien;
- `uv run --locked pytest tests` blijft de volledige sequentiële niet-trage controle;
- `uv run pytest tests -n N` is geen ondersteunde gezamenlijke parallelle opdracht;
- workergetallen zijn hardware- en suiteafhankelijk.

### Lokale verificatie vóór push

```bash
uv run --locked pytest tests --ignore=tests/integration --ignore=tests/browser \
  -q --no-cov -m "not slow and not real_solver" -n 4 --dist load
uv run --locked pytest tests --ignore=tests/integration --ignore=tests/browser \
  -q --no-cov -m "not slow and real_solver" -n 0
uv run --locked pytest tests/browser -q --no-cov \
  -m "not slow and not real_solver" -n 2 --dist load
uv run --locked pytest tests/browser -q --no-cov \
  -m "not slow and real_solver" -n 0
uv run --locked pytest tests/integration -q --no-cov -m "not slow" -n 0
```

Controleer de workflowsyntaxis met de bestaande pre-commit/lintmiddelen. Push na goedgekeurde
commit de featurebranch en controleer de echte GitHub Actions-uitvoering.

### CI-verificatie

Voor deze slice is lokaal groen niet voldoende. Controleer in GitHub Actions:

- de vier testjobs starten werkelijk onafhankelijk;
- solverselecties tonen `-n 0` en overlappen binnen hun runner geen andere selectie;
- de browserjob start maximaal twee xdist-workers voor de niet-solverselectie;
- alle JUnit- en failure-artifacts worden gepubliceerd;
- de gecombineerde coverage-job vindt alle drie databestanden;
- de aggregatiecheck `Tests` faalt wanneer één dependency faalt;
- jobduur en totale wall-clockduur worden genoteerd.

Als GitHub Actions niet bereikbaar is, is de slice nog niet compleet. Rapporteer dan de
externe blokkade en commit de workflow niet als bewezen eindoplossing.

### Review en commit

Voorgesteld commitbericht:

```text
ci: run test suites with resource-aware parallelism
```

Stop na rapportage en wacht op goedkeuring.

## Slice 4 — stabiliteitsproef en alleen noodzakelijke correcties

### Afhankelijkheden

Voer deze slice pas uit nadat de CI-slice is goedgekeurd, gecommit en gepusht. Dit is primair
een verificatiesessie. Zij hoeft geen commit op te leveren.

### Doel

Bewijs dat de nieuwe indeling de flakiness werkelijk wegneemt en dat de snelheidswinst niet
alleen uit toevallige groene runs bestaat.

### Oorspronkelijk voorgestelde proef

Voorgesteld was:

1. tien opeenvolgende GitHub Actions-runs van de relevante featurebranch, zonder retries;
2. drie lokale runs van de fast niet-solverselectie;
3. drie lokale runs van de browser niet-solverselectie;
4. één lokale sequentiële volledige niet-trage run:

   ```bash
   uv run --locked pytest tests -n 0
   ```

5. één expliciete slow-acceptancerun volgens de bestaande opdracht;
6. controle van coverage.xml en de aggregatiecheck.

### Uitkomst — 12 september 2026

De proef is uitgevoerd op commit `fc20989` in PR #2. De eerste uitvoering en drie volledige
handmatige workflow-heruitvoeringen waren groen. Iedere attempt startte alle jobs opnieuw op
schone GitHub-hosted runners; pytest zelf gebruikte geen retries. Een eerste failure zou de
proef hebben gestopt.

De oorspronkelijk voorgestelde tien onafhankelijke runs bleken met de beschikbare triggers
niet uitvoerbaar zonder negen extra commits of negen close/reopen-cycli op de pull request.
Volledige workflow-heruitvoeringen blijven in GitHub bovendien attempts van dezelfde run.
Na vier groene volledige attempts is in overleg gestopt: tien groene waarnemingen zouden
nog steeds geen statistisch bewijs van afwezige flakiness vormen, terwijl iedere extra
attempt hoofdzakelijk 7 tot 14 minuten slow-acceptancetijd zou verbruiken.

Alle vier attempts selecteerden exact dezelfde tests:

| Selectie | Tests | Pytest-mediaan | Langzaamste pytest-run |
|---|---:|---:|---:|
| Fast zonder echte solver | 565 passed, 1 skipped | 17,14 s | 20,78 s |
| Fast met echte solver | 34 passed | 2,23 s | 2,25 s |
| Browser zonder echte solver | 127 passed | 1m15,34s | 1m23,18s |
| Browser met echte solver | 10 passed | 21,65 s | 24,08 s |
| Integration non-slow | 22 passed | 18,16 s | 19,55 s |
| Slow acceptance | 1 passed | 10m45,18s | 13m29,64s |

De totale jobtijden omvatten checkout, installatie, artifactverwerking en testuitvoering:

| CI-job | Mediaan | Langzaamste attempt |
|---|---:|---:|
| Quality | 48 s | 53 s |
| Fast tests | 44 s | 45 s |
| Browser tests | 2m20 | 2m37 |
| Integration tests | 42 s | 42 s |
| Slow acceptance test | 11m04 | 13m50 |
| Coverage | 21 s | 47 s |
| Tests | 3 s | 3 s |
| Volledige workflow | 11m15 | 14m25 |

De drie lokale fast-runs waren groen in 24,02 s, 21,24 s en 22,71 s: mediaan 22,71 s,
langzaamste run 24,02 s. De drie lokale browserruns waren groen in 8m15,46, 6m15,07 en
6m09,26: mediaan 6m15,07, langzaamste run 8m15,46. Er waren geen timeouts, workercrashes,
SQLite-locks of achtergebleven pytest-, browser- of serverprocessen.

De geplande volledige lokale sequentiële run is tijdens de kostenheroverweging afgebroken en
wordt daarom niet als geslaagd resultaat opgevoerd. Een extra lokale slow-run is niet gedaan:
dezelfde slow-test was al viermaal zonder retry groen op de doelomgeving. Samen selecteren de
CI-lanes 759 non-slow tests en één slow-test, zodat geen test door de splitsing verloren ging.

Alle verwachte JUnit- en coverage-artifacts zijn gepubliceerd. De coverage-job combineerde
de drie databestanden succesvol tot 95% dekking; de stabiele aggregatiecheck `Tests` was in
alle attempts groen. Playwright-failure-artifacts ontbraken zoals verwacht, omdat geen
browsertest faalde.

Twee niet-blokkerende onderhoudspunten zijn waargenomen maar vallen buiten deze stabiliteits-
slice: bestaande `ResourceWarning`-meldingen voor niet-gesloten SQLite-connecties in enkele
fast-tests, en de GitHub-waarschuwing dat `actions/checkout@v4` nog Node.js 20 target. Geen
van beide veroorzaakte in deze proef een failure of timingprobleem.

De rapportage hierboven legt per CI-job vast:

- aantal tests;
- pytest-runtime;
- totale jobruntime;
- mediaan over de uitgevoerde attempts;
- langzaamste run;
- eventuele timeout, workercrash of achtergebleven subprocessmelding.

### Beslisregels

- Eén failure in de meetreeks is geen acceptabele uitkomst zolang die niet als echte regressie
  is verklaard en opgelost.
- Een timeout wordt niet opgelost met retry of algemene timeoutverhoging.
- Als de niet-solverselectie flaky blijft, verlaag eerst alleen haar workergetal en herhaal de
  proef.
- Als een gemarkeerde solvertest flaky blijft terwijl hij sequentieel draait, onderzoek die
  test afzonderlijk; de oorzaak is dan niet meer xdist-oversubscriptie.
- Als na een browserfailure volgende tests database- of bestandsfouten geven, controleer of
  een achtergrondsolve na testteardown doorloopt. Voeg dan in een nieuwe, vooraf afgestemde
  slice expliciete test-side threadregistratie en cleanup toe. Verander niet stilzwijgend de
  productie-threadarchitectuur.
- Als `-n 2` en `-n 1` voor browser minder dan 5% verschillen, kies `-n 1`.
- Als fast `-n 4` en `-n 2` minder dan 5% verschillen, kies `-n 2`.

Wanneer alleen workergetallen of direct bijbehorende documentatie moeten worden gecorrigeerd,
mag deze sessie één kleine, complete vervolgwijziging voorstellen. Stop ook dan vóór commit.

Voorgesteld commitbericht indien aanpassing nodig blijkt:

```text
ci: tune test workers for hosted runners
```

## Definitie van klaar

- Gewone fast- en browsertests draaien aantoonbaar parallel.
- Binnen één runner is nooit meer dan één echte CP-SAT-solve tegelijk actief.
- Integration en slow draaien sequentieel.
- Vier volledige CI-attempts op dezelfde commit zijn zonder pytest-retry groen; de afwijking
  van de oorspronkelijk voorgestelde tien runs is hierboven gemotiveerd.
- SQLite, Flask-serverpoort, browsercontext en storage blijven geïsoleerd.
- De CLI-subprocespoort heeft geen close/rebind-race meer.
- De autoplaytest wacht op gedrag, niet op een vaste slaap.
- Een failure levert JUnit en bij browsertests relevante Playwright-diagnostiek op.
- De gecombineerde coverage-uitkomst blijft beschikbaar.
- Branch protection kan één stabiele `Tests`-aggregatiecheck gebruiken.
- README, AGENTS.md, CI en het oude versnellingsplan spreken elkaar niet meer tegen.
- Productie-`NUM_WORKERS`, solveruitkomsten en gebruikersgedrag zijn ongewijzigd.
- Er zijn geen automatische testretries, globale timeoutverhogingen of custom schedulers
  toegevoegd.

## Korte startprompts per sessie

Gebruik aan het begin van iedere afzonderlijke Luna xhigh-sessie de toepasselijke prompt.

### Sessie 1

```text
Lees AGENTS.md en docs/plans/stabiele-parallelle-tests-ci.md volledig. Voer uitsluitend
Slice 1 uit: markeer tests die de echte CP-SAT-solver starten en verifieer dat de gesplitste
selecties samen exact de bestaande suites vormen. Raak ander werk niet aan. Rapporteer het
diff en de testresultaten en stop vóór commit voor review.
```

### Sessie 2

```text
Lees AGENTS.md en docs/plans/stabiele-parallelle-tests-ci.md volledig. Controleer dat Slice 1
is goedgekeurd en gecommit. Voer uitsluitend Slice 2 uit: verwijder de CLI-poortrace en de
vaste autoplay-wacht, plus alleen aantoonbaar gelijksoortige kleine waits. Herhaal de
gerichte tests tienmaal. Rapporteer en stop vóór commit voor review.
```

### Sessie 3

```text
Lees AGENTS.md en docs/plans/stabiele-parallelle-tests-ci.md volledig. Controleer dat Slices
1 en 2 zijn goedgekeurd en gecommit. Voer uitsluitend Slice 3 uit: splits GitHub Actions op
resourceprofiel, behoud gecombineerde coverage en een stabiele Tests-aggregatiecheck, en werk
de bijbehorende documentatie in dezelfde slice bij. Verifieer ook de echte Actions-run.
Rapporteer en stop vóór commit voor review.
```

### Sessie 4

```text
Lees AGENTS.md en docs/plans/stabiele-parallelle-tests-ci.md volledig. Controleer dat Slice 3
is goedgekeurd, gecommit en gepusht. Voer uitsluitend de stabiliteitsproef uit Slice 4 uit.
Gebruik geen retries. Leg runtimes en iedere eerste failure vast. Wijzig alleen iets als het
plan daar expliciet een beslisregel voor geeft; rapporteer en stop altijd vóór commit.
```
