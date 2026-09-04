# Implementatieplan: snelle lokale en merge-testfeedback

## Doel

De testfeedback van ALI Express wordt aantoonbaar sneller zonder inhoudelijke dekking te
verliezen of testisolatie af te zwakken. Dit plan versnelt zowel de gewone Python/Flask-suite
als de Playwright-browsersuite. Integration en de realistische slow-solvertest blijven
inhoudelijk ongewijzigd, maar de merge-run mag ze slimmer gelijktijdig plannen.

Beoogde uitkomst:

- gerichte feedback tijdens ontwikkelen doorgaans binnen enkele seconden;
- minimaal 2x snellere volledige snelle Python/Flask-suite;
- minimaal 2x snellere volledige browsertestsuite;
- een reproduceerbaar gekozen worker-aantal en xdist-scheduler per suite;
- een optionele testmon-lus voor Pythonwijzigingen, nooit als enige kwaliteitsgate;
- als nice-to-have: de realistische slow-test begint bij een merge direct in een eigen
  proces/lane terwijl de overige tests parallel verdergaan;
- geen wijziging aan productiehashing, solvergedrag of productie-`NUM_WORKERS`.

## Suites in dit plan

```text
Snelle Python/Flask-suite:
  tests/ minus tests/integration minus tests/browser minus @pytest.mark.slow

Browsersuite:
  tests/browser

Integration:
  tests/integration minus @pytest.mark.slow

Slow:
  tests/integration/test_integration_herindelen.py::test_herindelen_realistic_scale
```

De suites houden afzonderlijke commando's. Eén gezamenlijke xdist-pool zou browsertests over
alle workers kunnen verspreiden, waardoor iedere betrokken worker een Chromiumproces start,
terwijl andere workers tegelijk OR-Tools met acht threads uitvoeren. Afzonderlijk tunen is
voorspelbaarder en voorkomt CPU- en geheugensurprises.

## Gemeten uitgangssituatie

Metingen op 4 september 2026, op de toenmalige worktree en zonder coverage:

| Meting | Resultaat |
|---|---:|
| Snelle Python/Flask-suite, sequentieel | 511 tests in 120,64 s |
| Alleen verzamelen van de snelle suite | 511 tests in 0,14 s |
| Browsersuite, sequentieel | 91 tests in 181,25 s |
| Volledige standaardselectie `-m "not slow"` | 623 van 624 tests |
| Integration | 21 van 22 tests; 1 slow-test gedeselecteerd |
| Logische CPU's op de ontwikkelmachine | 20 |
| OR-Tools-workers per gewone solve | 8 |
| Testfuncties met directe `client`-fixture | 165 |
| Standaard testhash genereren | circa 56,3 ms |
| Standaard testhash controleren | circa 58,4 ms |

De gewone `client`-fixture besteedt naar schatting
`165 * (56,3 + 58,4) ms = 18,9 s` aan productie-sterke hashing en verificatie.

De browsertestmeting toont een groter setup-probleem:

- de traagste setups kosten circa 1,36–1,54 s per test;
- `live_server` start en stopt voor iedere test opnieuw een echte Flask-server;
- dezelfde fixture dropt/maakt de database opnieuw, maakt een sterke hash en seedt de school;
- Playwright zelf doet het gunstiger: `browser` is session-scoped en `context`/`page` zijn
  function-scoped, zodat één Chromiumproces al wordt hergebruikt met een schoon browserprofiel
  per test;
- vijf processingtests wachten hardcoded 1.200 ms op de volgende poll; overige vaste waits
  zijn 100–300 ms.

Een verkennende xdist-run met vier workers is afgebroken vanwege databasefouten. Dit was geen
geldige performancebenchmark: de controller zette eenmaal `DATABASE_URL`, waarna alle
workers door `os.environ.setdefault(...)` dezelfde SQLite-file bleven gebruiken.

## Vaststaande keuzes

1. **Dependency en werkende isolatie vormen samen één slice.** Alleen xdist installeren
   levert geen bruikbaar resultaat en wordt niet apart gecommit.
2. **Eerst een geldige parallelle basis en eerste tuning, daarna fixture-optimalisaties.**
   Zo is zichtbaar welke winst van parallelisatie en welke van fixtures komt.
3. **Geen `-n auto`.** Een pytest-worker kan een OR-Tools-solve met acht eigen threads
   starten. Worker-aantallen worden gemeten.
4. **Geen globale `-n` of `--dist` in pytest-`addopts`.** Gericht debuggen met `-n 0` en het
   afzonderlijk draaien van browser-, integration- en slow-tests blijven voorspelbaar.
5. **Coverage niet in snelle ontwikkellussen.** Die commando's gebruiken `--no-cov`.
   Coverage blijft onderdeel van de volledige controle.
6. **Authgedrag blijft echt getest.** Expliciete auth- en login-browsertests behouden echte
   loginflows. Alleen gedeelde setup gebruikt een goedkope, geldige testhash.
7. **Playwright-isolatie blijft function-scoped.** `page` en `context` worden niet gedeeld.
   Alleen de Flask-server wordt eenmaal per worker gestart; database en storage worden voor
   iedere test opnieuw geïsoleerd.
8. **`load` en `worksteal` worden gemeten.** Worksteal is kansrijk bij ongelijke testduren,
   maar geen vooraf vastgelegde winnaar.
9. **Testmon is een opt-in versneller, geen gate.** Templates, CSS, JavaScript, JSON, XML en
   andere assets worden niet door testmon als codeafhankelijkheden gevolgd.
10. **De slow-test reserveren gebeurt niet via een impliciete xdist-truc.** De ingebouwde
    schedulers garanderen niet dat een benoemde test als eerste op een gereserveerde worker
    komt. Een aparte gelijktijdige lane/proces is eenvoudiger en deterministisch.

## Te wijzigen bestanden

Kern:

- `pyproject.toml` — xdist en, bij een geslaagde pilot, pytest-testmon toevoegen aan dev.
- `uv.lock` — bijgewerkt door `uv add`.
- `conftest.py` — unieke SQLite-database per worker/proces en veilige opruiming.
- `tests/conftest.py` — goedkopere algemene ingelogde client.
- `tests/browser/conftest.py` — server per worker, teststate per test en goedkope seedhash.
- `README.MD` — ontwikkel-, browser-, volledige en merge-testcommando's.
- `.gitignore` — `.testmondata` als testmon wordt behouden.

Alleen indien na meting nodig:

- `templates/processing.html` en de bijbehorende Flaskroute/config — uitsluitend om de
  pollinginterval onder tests configureerbaar te maken en vaste sleeps te vervangen door
  wachten op een echt `/status`-response.
- een klein PowerShell- of CI-runnerbestand voor twee gelijktijdige merge-lanes, zodra bekend
  is waar de merge-run werkelijk wordt uitgevoerd. Er staat nu geen CI-configuratie in de
  repository.

Solverproductiecode en productiehashinstellingen horen niet te veranderen.

## Vooraf — veilige werkstart en nieuwe nulmetingen

Begin de volgende sessie met `git status --short`. De worktree bevat mogelijk ander lokaal
werk; behoud dat en raak alleen de afgesproken bestanden aan. Controleer vooral of
`pyproject.toml`, `uv.lock`, beide `conftest.py`-bestanden of `README.MD` inmiddels veranderd
zijn.

Voer op dezelfde worktree één warme nulmeting per relevante suite uit:

```powershell
uv run pytest tests --ignore=tests/integration --ignore=tests/browser -q --no-cov
uv run pytest tests/browser -q --no-cov --durations=25 --durations-min=0.05
```

Noteer testenaantal, pytest-runtime en slagen/falen. De eerdere 120,64 s en 181,25 s zijn
context; de acceptatievergelijking gebruikt de metingen uit dezelfde implementatiesessie.

## Voortgang implementatiesessie

Deze sessie werkt op branch `speed-up-tests`. De warme nulmetingen volgens het plan zijn:

| Suite | Resultaat |
|---|---:|
| Snelle Python/Flask-suite | 511 tests in 98,35 s |
| Browsersuite | 93 tests in 105,59 s |

Het plan is als eerste commit op deze branch vastgelegd (`1fc6b72`). Slice 1 is
geïmplementeerd en geverifieerd; de implementatiewijzigingen blijven ongecommit voor review.

Slice-1-verificatie:

| Controle | Resultaat |
|---|---:|
| Gerichte suite, `-n 0` | 93 passed in 46,46 s |
| Gerichte suite, `-n 4 --dist load` | 93 passed in 8,64 s |
| Volledige snelle suite, `-n 4 --dist load` | 511 passed in 15,19 s |
| Vooraf ingestelde externe `DATABASE_URL` | genegeerd; externe database niet aangemaakt |
| Waargenomen databases tijdens `-n 4` | 5 unieke paden: `master`, `gw0`–`gw3` |
| Achtergebleven `aliexpress-pytest-*.db`-bestanden | 0 |

## Slice 1 — xdist plus veilige database-isolatie

### Wijziging

Voeg xdist toe aan de bestaande optionele dev-dependencies:

```powershell
uv add --optional dev pytest-xdist==3.8.0
```

Pas in dezelfde slice de root-`conftest.py` aan. Iedere pytest-procesimport overschrijft
bewust `DATABASE_URL` met een nieuw aangemaakte testdatabase. `setdefault` verdwijnt: een
extern ingestelde database-URL mag nooit gebruikt worden door tests die `db.drop_all()`
aanroepen.

De initialisatie blijft vóór iedere import van `app`, `aliexpress.web.appconfig` of
`tests/conftest.py`. Richtinggevende vorm:

```python
worker = os.environ.get("PYTEST_XDIST_WORKER", "master")
_db_fd, _db_path = tempfile.mkstemp(
    prefix=f"aliexpress-pytest-{worker}-",
    suffix=".db",
)
os.close(_db_fd)
os.environ["DATABASE_URL"] = f"sqlite:///{_db_path}"
```

Voeg testspecifieke opruiming toe die SQLAlchemy-sessies en de engine sluit voordat de
SQLite-file op Windows wordt verwijderd. Ruim ook de eventuele ongebruikte controllerfile
best effort op. Een opruimfout mag een eerdere testfailure niet maskeren; waarschuw in een
verder groene run wel met het concrete pad.

### Verificatie

```powershell
uv run pytest tests/test_auth.py tests/test_processes.py tests/test_results.py -q --no-cov -n 0
uv run pytest tests/test_auth.py tests/test_processes.py tests/test_results.py -q --no-cov -n 4 --dist load
uv run pytest tests --ignore=tests/integration --ignore=tests/browser -q --no-cov -n 4 --dist load
```

Controleer:

- geen `database is locked`, ontbrekende tabellen, dubbele sleutels of data uit andere tests;
- unieke databasepaden per worker;
- een vooraf in de shell ingestelde `DATABASE_URL` wordt genegeerd;
- geen nieuwe `aliexpress-pytest-*.db`-bestanden na een normale run;
- sequentieel `-n 0` blijft werken.

Bij parallel-only failures de gedeelde state oplossen. Geen sleeps, retries of ruimere
timeouts toevoegen om races te verhullen.

**Commit:** `test: add xdist with isolated worker databases`

## Slice 2 — worker-aantal en scheduler voor de snelle suite kiezen

Gebruik exact dezelfde snelle suite en `--no-cov`. Meet eerst eenmaal:

```powershell
uv run pytest tests --ignore=tests/integration --ignore=tests/browser -q --no-cov -n 2 --dist worksteal
uv run pytest tests --ignore=tests/integration --ignore=tests/browser -q --no-cov -n 4 --dist worksteal
uv run pytest tests --ignore=tests/integration --ignore=tests/browser -q --no-cov -n 6 --dist worksteal
```

Laat duidelijk tragere kandidaten vallen. Draai de beste twee nog tweemaal in afwisselende
volgorde. Vergelijk bij het winnende worker-aantal vervolgens minimaal twee runs met:

```text
--dist load
--dist worksteal
```

Worksteal geeft workers aanvankelijk een eigen queue. Een bijna lege worker neemt nog niet
gestarte tests over van een worker met veel resterend werk. Een reeds lopende solvertest kan
niet worden gesplitst of verplaatst. `load` stuurt pending tests naar iedere beschikbare
worker zonder gegarandeerde volgorde.

| Workers | Scheduler | Run 1 | Run 2 | Run 3 | Mediaan | Groen? |
|---:|---|---:|---:|---:|---:|---|
| 0 | geen | | | | | |
| 2 | worksteal | | | | | |
| 4 | worksteal | | | | | |
| 6 | worksteal | | | | | |
| winnaar | load | | | | | |

Beslisregel:

- kies de snelste configuratie die drie volledige runs groen blijft;
- bij minder dan 5% verschil: kies minder workers;
- bij gelijkwaardige schedulers: kies `load`, de xdist-default;
- verander bij oversubscriptie niet meteen productie-`NUM_WORKERS = 8`.

Dit is een meet-/beslisslice en hoeft geen eigen commit te hebben; leg de uitkomst vast in
dit plan of de sessienotities.

## Slice 3 — algemene client-fixture goedkoper maken

Maak in `tests/conftest.py` eenmaal per worker een goedkope, geldige hash buiten de
function-scoped `client`-fixture:

```python
_TEST_PASSWORD_HASH = generate_password_hash(
    "testpass",
    method="pbkdf2:sha256:1",
)
```

Gebruik die constante bij `test-school`. De fixture blijft eerst via `POST /login` inloggen;
de sessieopbouw blijft daarmee gelijk, terwijl per-test scrypt-generatie en -controle
verdwijnen. `tests/test_auth.py` en `tests/test_admin.py` blijven de standaardhash gebruiken.

Verifieer:

```powershell
uv run pytest tests/test_auth.py tests/test_admin.py tests/test_app.py -q --no-cov -n 0
uv run pytest tests/test_processes.py tests/test_results.py -q --no-cov -n 4 --dist load
```

Meet daarna de snelle suite eenmaal sequentieel en met de winnaar uit slice 2. Alleen als de
resterende login-POST aantoonbaar meer dan circa 2 s kost, mag de algemene `client` direct
een Flask-Login-sessie zetten. `unauthed_client`, `admin_client` en browserlogintests krijgen
geen sessie-injectie.

Omdat de werklast door deze wijziging verandert, voer één bevestigingsrun uit met de nummer
twee uit slice 2. Verander de gedocumenteerde winnaar alleen bij meer dan 5% verschil.

**Commit:** `test: avoid production-strength hashing in route fixtures`

## Slice 4 — browsertestserver en state van elkaar scheiden

### Waarom dit eerst komt

De browsersuite duurt momenteel 181,25 s. Playwright hergebruikt Chromium al per sessie en
maakt goedkope geïsoleerde browsercontexts per test. De eigen `live_server`-fixture doet
daarentegen voor alle 91 tests serverstart, serverstop, databaseopbouw, seedhash en loginsetup
opnieuw. De server en de teststate hebben verschillende gewenste scopes.

### Gewenste fixturevorm

Refactor `tests/browser/conftest.py` naar:

1. `live_server`, session-scoped per pytest-worker:
   - configureert de stabiele testinstellingen;
   - start `make_server("127.0.0.1", 0, app)` en één daemonthread;
   - yieldt de willekeurige lokale basis-URL;
   - stopt en joint pas na alle tests van die worker.
2. Een function-scoped, autouse `browser_test_state(tmp_path, live_server)`:
   - zet voor deze test `app.config["STORAGE_DIR"] = str(tmp_path)`;
   - verwijdert de SQLAlchemy-session vóór schemareset;
   - doet `drop_all()`/`create_all()` op de unieke workerdatabase;
   - seedt `browser-school` met een eenmaal per worker gemaakte goedkope geldige hash;
   - verwijdert de DB-session na de test.
3. Playwrights `page` en `context` blijven ongewijzigd function-scoped.
4. `login` en `open_groups_to` hangen expliciet van `browser_test_state` af, zodat
   fixturevolgorde geen impliciete aanname wordt.

Binnen één worker lopen tests sequentieel en mag `STORAGE_DIR` per test wijzigen. Met xdist
heeft iedere worker een eigen Pythonproces, Flask-app, SQLite-file, serverpoort en
Chromiumproces.

### Verificatie vóór parallelisatie

```powershell
uv run pytest tests/browser/test_login_browser.py -q --no-cov -n 0
uv run pytest tests/browser/test_preferences_form_browser.py -q --no-cov -n 0
uv run pytest tests/browser -q --no-cov -n 0 --durations=25 --durations-min=0.05
```

Controleer expliciet:

- iedere test start met lege database/storage en verse cookies;
- fout wachtwoord, login en logout blijven echte browserflows;
- een failure laat de session-scoped server netjes stoppen;
- setupduren zijn substantieel lager dan de eerdere 1,36–1,54 s;
- geen test blijkt afhankelijk van bestands-, database- of browserstate van zijn voorganger.

**Commit:** `test: reuse one Flask server per browser worker`

## Slice 5 — browsertests parallel tunen

Start behoudend: iedere worker start een Chromiumproces en enkele browsertests starten echte
OR-Tools-solves. Meet 2, 3 en 4 workers, niet `auto`:

```powershell
uv run pytest tests/browser -q --no-cov -n 2 --dist load
uv run pytest tests/browser -q --no-cov -n 3 --dist load
uv run pytest tests/browser -q --no-cov -n 4 --dist load
```

Vergelijk bij de twee beste aantallen ook `worksteal`. Draai de winnaar drie keer. Houd
Playwrights tracing, video en screenshots uit tijdens speedmetingen; hun defaults zijn al
uit. Artifacten `retain-on-failure` mogen apart worden gekozen voor merge-diagnostiek, maar
niet in de vergelijkende lokale benchmark.

| Workers | Scheduler | Run 1 | Run 2 | Run 3 | Mediaan | Groen? |
|---:|---|---:|---:|---:|---:|---|
| 0 | geen | | | | | |
| 2 | load | | | | | |
| 3 | load | | | | | |
| 4 | load | | | | | |
| winnaar | worksteal | | | | | |

Dezelfde beslisregels gelden: drie groene runs, minder workers bij minder dan 5% verschil,
en de eenvoudigste scheduler bij gelijkstand.

### Vaste waits alleen na opnieuw profileren

Na serverhergebruik en xdist opnieuw `--durations` draaien. Alleen wanneer de vaste waits nog
een materieel deel van de resterende runtime vormen:

- vervang 100–300 ms sleeps door Playwright-asserties of wachten op het relevante event;
- vervang de vijf waits van 1.200 ms door wachten op een echt volgend `/status`-response;
- maak zo nodig de productie-default van 1.000 ms via Flaskconfig testbaar en zet uitsluitend
  in `TESTING` een korte interval;
- behoud in productie exact 1.000 ms;
- gebruik geen algemene nul-timeout en geen willekeurige sleepverlaging die tests flaky maakt.

**Commit:** `test: run browser suite safely in parallel`

## Slice 6 — kleine pytest-testmon-pilot

Testmon is logisch als vierde, zeer korte ontwikkellus bovenop gerichte bestandsruns. Het
gebruikt coverage-informatie om per test bij te houden welke Python-code is uitgevoerd en
selecteert na wijzigingen alleen afhankelijke tests. De eerste run bouwt `.testmondata`; pas
latere runs kunnen selecteren.

Het is geen vervanger voor de volledige suite in deze repository:

- testmon volgt volgens de eigen documentatie geen statische bestanden of externe services;
- wijzigingen in `templates/`, `static/`, JSON/XML/XLSX-testdata of browsergedrag kunnen dus
  relevante tests hebben zonder dat testmon ze selecteert;
- het huidige globale `-m "not slow"` zorgt er volgens testmon voor dat gewone selectie kan
  terugvallen op `--testmon-noselect`; de pilot moet daarom expliciet controleren of
  `--testmon-forceselect` nodig is;
- de dependencydatabase is lokaal en hoort niet in git.

### Pilot

Voeg alleen voor de pilot de actuele gepinde dependency toe:

```powershell
uv add --optional dev pytest-testmon==2.2.0
```

Voeg `.testmondata` toe aan `.gitignore`. Bouw de database alleen over de snelle
Python/Flask-suite, eerst zonder parallelisatie om gedrag begrijpelijk te houden:

```powershell
uv run pytest tests --ignore=tests/integration --ignore=tests/browser -q --no-cov --testmon --testmon-forceselect -n 0
```

Gebruik vervolgens een echte kleine Pythonwijziging uit de sessie om te controleren:

- welke tests geselecteerd worden;
- runtime ten opzichte van een gericht testbestand en `--lf`;
- combinatie met het gekozen xdist-commando;
- of een aansluitende volledige snelle run iets vindt dat testmon miste.

### Beslisregel

Behoud testmon alleen als het na de initiële databaseopbouw merkbaar prettiger is dan zelf
het relevante testbestand kiezen en geen verrassende Pythonmissers laat zien. Documenteer:

```powershell
# Alleen Pythonwijzigingen; nooit de enige controle voor templates/static/testdata
uv run pytest tests --ignore=tests/integration --ignore=tests/browser -q --no-cov --testmon --testmon-forceselect -n <winnaar>
```

Niet opnemen in globale `addopts`, pre-push of merge. Als de pilot weinig oplevert, verwijder
de dependency en `.gitignore`-regel weer in dezelfde slice; dat is een geldige uitkomst.

**Optionele commit:** `test: add opt-in affected-test loop with testmon`

## Slice 7 — nice-to-have: slow-test meteen in een eigen merge-lane

### Gewenste planning

De realistische slow-test is potentieel langer dan de rest van de mergecontrole. De minimale
totale wandkloktijd ontstaat wanneer hij direct start naast de overige suites:

```text
t=0  lane slow:    [ realistic-scale slow-test ------------------------- ]
t=0  lane overige: [ fast ][ browser ][ integration-non-slow ][coverage]

totale duur ≈ max(duur slow, duur overige), niet de som
```

`@pytest.mark.xdist_group` met `--dist loadgroup` kan tests aan dezelfde worker koppelen, maar
reserveert geen specifieke worker en garandeert volgens xdist geen uitvoervolgorde. Ook
`load` biedt expliciet geen gegarandeerde volgorde. Bouw daarom geen fragiele oplossing die
toevallig vertrouwt op collection order of initiële chunks.

### Voorkeursimplementatie

1. Als de merge-run door CI wordt uitgevoerd: maak twee gelijktijdige jobs/lane-processen.
2. Als de merge-run lokaal op één Windows-machine wordt uitgevoerd: maak een kleine runner
   die twee verborgen childprocessen tegelijk start en beide exitcodes verzamelt.
3. Start de slow-lane als eerste statement, daarna onmiddellijk de overige lane.
4. Slow-lane:

   ```powershell
   uv run pytest tests/integration/test_integration_herindelen.py::test_herindelen_realistic_scale -q --no-cov -m slow -n 0
   ```

5. Overige lane: gebruik de gekozen commando's voor snelle, browser- en non-slow integration.
6. De merge-run slaagt alleen wanneer beide lanes exitcode 0 geven; bij één vroege failure
   mag de andere lane voor diagnose worden afgebroken als de runner dat betrouwbaar doet.

De slow-lane gebruikt `--no-cov`, zodat twee gelijktijdige pytest-cov-processen niet dezelfde
`.coverage`/`coverage.xml` wissen of overschrijven. De overige lane produceert het canonieke
coverage-rapport. Als dekking die uitsluitend door de slow-test ontstaat later verplicht
wordt, gebruik dan unieke coverage-datafiles en combineer die expliciet; voeg dat niet vooraf
toe.

Er staat momenteel geen CI-configuratie in deze repository. Deze slice blijft daarom
nice-to-have totdat in de implementatiesessie bekend is welk merge-systeem de tests start.
Een eigen xdist-scheduler schrijven is nadrukkelijk niet de fallback; dat is te veel
complexiteit voor één speciale test.

**Optionele commit:** `test: start merge slow test in a parallel lane`

## Slice 8 — documenteer de definitieve testlussen

Werk `README.MD` pas bij nadat beide suites getuned en testmon beoordeeld zijn. Gebruik overal
de gemeten winnaars, niet de voorbeeldwaarden.

### A. Gericht tijdens wijzigen

```powershell
uv run pytest tests/test_betreffend_bestand.py -q --no-cov -x
uv run pytest tests --ignore=tests/integration --ignore=tests/browser -q --no-cov --lf
```

### B. Optioneel impacted Python-tests

Alleen toevoegen als slice 6 slaagt, met waarschuwing dat template/static/testdatawijzigingen
niet worden gevolgd.

### C. Volledige snelle Python/Flask-suite

```powershell
uv run pytest tests --ignore=tests/integration --ignore=tests/browser -q --no-cov -n <winnaar> --dist <winnaar>
```

### D. Volledige browsersuite

```powershell
uv run pytest tests/browser -q --no-cov -n <browser-winnaar> --dist <browser-scheduler>
```

### E. Volledige controles

```powershell
uv run pytest tests/integration
uv run pytest tests
uv run pytest tests -m slow
```

Documenteer ook de merge-runner/lane als slice 7 werkelijk is geïmplementeerd. Maak duidelijk
dat de huidige `-m "not slow"` slechts de ene gemarkeerde slow-test uitsluit; browser en
overige integration worden daarmee niet uitgesloten.

**Commit:** `docs: document fast browser and merge test loops`

## Eindverificatie

1. Drie opeenvolgende groene runs met de gekozen snelle Python/Flask-configuratie.
2. Eén groene sequentiële snelle run met `-n 0`.
3. Drie opeenvolgende groene runs met de gekozen browserconfiguratie.
4. Eén groene sequentiële browserrun met `-n 0` om verborgen volgordeafhankelijkheid te
   signaleren.
5. Eén parallelle run met de bestaande coverageopties; controleer gecombineerd rapport en
   geldige `coverage.xml`.
6. De volledige non-slow suite en non-slow integration volgens de bestaande afspraak.
7. Slow uitsluitend in de merge-run of wanneer expliciet gevraagd; controleer bij slice 7
   dat hij werkelijk direct naast de overige lane begint.
8. Als testmon behouden is: impacted run gevolgd door een volledige snelle run; rapporteer
   eventuele gemiste tests.
9. `git diff --check` en controle dat productiehashing en solverinstellingen ongewijzigd zijn.

Vergelijk medianen met de nulmetingen uit dezelfde sessie:

```text
speed-up fast    = sequentiële fast-nulmeting / parallelle fast-mediaan
speed-up browser = sequentiële browsernulmeting / parallelle browsermediaan
```

## Definitie van klaar

- Iedere xdist-worker gebruikt een eigen tijdelijke SQLite-database.
- De snelle Python/Flask-suite is driemaal groen en minimaal 2x sneller.
- De browsersuite is driemaal groen en minimaal 2x sneller.
- Beide suites blijven afzonderlijk sequentieel uitvoerbaar met `-n 0`.
- Browsercontexts/pages en database/storage blijven per test geïsoleerd.
- Echte auth-, browserlogin-, logout- en wachtwoordwijzigingsflows blijven gedekt.
- Coverage, integration en slow blijven afzonderlijk uitvoerbaar en gedocumenteerd.
- Testmon is óf bewust als opt-in behouden óf na de pilot volledig verwijderd.
- Testmon wordt nooit gepresenteerd als veilig voor template-, static- of testdatawijzigingen.
- Als de merge-lane is gebouwd, bepaalt het falen van beide lanes correct de eindstatus en
  start de slow-test direct.
- Geen custom xdist-scheduler, retries, verbergende sleeps of globale parallel-defaults.
- Productiehashing en `NUM_WORKERS = 8` zijn ongewijzigd.

## Bewust buiten scope

- De inhoud of tijdslimiet van de realistische slow-test veranderen.
- Productie-`NUM_WORKERS` van OR-Tools aanpassen.
- Browsercontexts of pagina's tussen tests delen.
- Een complexe transaction/savepoint-fixture bouwen voordat `drop_all/create_all` na de
  overige wijzigingen opnieuw als bottleneck is gemeten.
- Testmon als CI/merge-gate gebruiken.
- Een custom xdist-scheduler onderhouden om één test aan een benoemde worker te pinnen.
- Alle suites in één xdist-pool dwingen.

Als een factor-2-doelstelling niet wordt gehaald, volgt opnieuw `--durations`- en
fixtureprofilering. Pas daarna wordt database-reset, solver-unitconfiguratie, polling of
imports als volgende optimalisatie gekozen.

## Bronnen

- [pytest-xdist: schedulers en workers](https://pytest-xdist.readthedocs.io/en/stable/distribution.html)
- [pytest-xdist: workers en unieke testrun-ID's](https://pytest-xdist.readthedocs.io/en/stable/how-to.html)
- [Playwright Python: fixture-scopes en parallelisatie](https://playwright.dev/python/docs/test-runners)
- [Playwright Python: geïsoleerde browsercontexts](https://playwright.dev/python/docs/next/browser-contexts)
- [pytest-testmon: selectie, opties en beperkingen](https://www.testmon.org/)
- [pytest-testmon: bepalen van afhankelijke tests](https://www.testmon.org/blog/determining-affected-tests/)
- [pytest-cov: gecombineerde coverage met xdist](https://pytest-cov.readthedocs.io/en/latest/xdist.html)
- [pytest: `--lf`, `--ff` en stepwise](https://docs.pytest.org/en/stable/how-to/cache.html)
