# Plan: platformonafhankelijke repository

## Doel

Maak de repository vanuit een verse clone bruikbaar en testbaar op Windows, Ubuntu en
macOS, zonder de Python/Flask/HTML/JavaScript-kern onnodig te herschrijven. Alle nieuwe
functionaliteit wordt test-first ontwikkeld in kleine slices. Iedere slice eindigt in
één afzonderlijke, inhoudelijk samenhangende commit.

De uitvoering vindt plaats op branch `chore/cross-platform-tdd`.

## Scope en beoogd supportcontract

Eerste klas ondersteund:

- Windows x64;
- Ubuntu x64, inclusief Ubuntu 26.04;
- macOS x64 en ARM64;
- CPython 3.11, 3.12 en 3.13;
- installatie en uitvoering vanuit een Git-clone met `uv`.

Voorlopig niet in scope:

- Windows ARM, omdat de vastgezette OR-Tools-versie geen Windows-ARM-wheel bevat;
- een zelfstandig installeerbare desktopapp, wheel of OS-installer;
- gegarandeerde ondersteuning van Python 3.14 zolang alle vastgezette binaire
  dependencies daarop niet aantoonbaar werken;
- een onzichtbaar draaiende achtergrondservice, tenzij dubbelklikbediening een harde
  gebruikerseis blijkt te zijn.

Het gewenste gemeenschappelijke bedieningsmodel is:

```text
uv run ali-express serve
uv run ali-express solve
uv run ali-express reset-local-data
```

De lokale server draait standaard in de voorgrond en stopt met `Ctrl+C`. OS-specifieke
shortcuts mogen eventueel als dunne adapters bestaan, maar bevatten geen eigen
proces-, configuratie- of verwijderlogica en geen absolute gebruikerspaden.

## Verplicht reviewprotocol

Voor iedere commit wordt de volgende cyclus gebruikt:

1. Begin vanaf de laatst goedgekeurde commit.
2. Schrijf eerst de kleinste relevante falende test en toon de verwachte failure.
3. Implementeer alleen wat nodig is om de test te laten slagen en refactor waar nodig.
4. Draai de tests die bij de slice horen en controleer het volledige staged/unstaged
   diff op onverwachte wijzigingen en PII.
5. Rapporteer aan de reviewer:
   - de gewijzigde bestanden;
   - een samenvatting van het diff;
   - de uitgevoerde tests en resultaten;
   - het voorgestelde commitbericht;
   - bekende risico's of open punten.
6. **Stop. Maak nog geen commit.**
7. Maak de commit uitsluitend na expliciete goedkeuring van de reviewer.
8. Toon daarna de commit-hash en begin pas vervolgens aan de volgende slice.

Geen `commit --amend`, squash, rebase, force-push of historie-rewrite zonder een
nieuwe expliciete goedkeuring. Een historie-rewrite is geen gewone commit en heeft de
afzonderlijke procedure verderop in dit document.

De commit van dit planbestand zelf volgt hetzelfde protocol en wordt dus pas na review
gemaakt.

## Huidige bevindingen

### Wat al platformonafhankelijk is

- De applicatiekern gebruikt Python, Flask, HTML en JavaScript.
- Opslag gebruikt hoofdzakelijk `os.path` en tijdelijke pytest-directories; deze API's
  zijn op zichzelf cross-platform.
- De progress writer bevat al expliciete afhandeling en een test voor Windows-filelocks.
- `uv.lock` is een universeel lockbestand met platform-specifieke wheels voor de
  belangrijkste dependencies op de beoogde architecturen.
- Op Ubuntu 26.04 met Python 3.13 slagen momenteel 511 snelle tests en 21 niet-trage
  integratietests. Eén test wordt daar terecht overgeslagen omdat die specifiek het
  Windows-filelockgedrag test.

### Concrete platformproblemen

1. `aliexpress.lnk` is een Windows-binary met een absoluut gebruikerspad en
   machine-identificerende metadata. Het bestand is niet overdraagbaar en bevat PII.
2. `aliexpress.bat` gebruikt PowerShell, Windows-venvpaden en `pythonw.exe`.
3. `kill-server.ps1` gebruikt WMI, Windows-procesnamen en Windows-netwerkcmdlets.
4. `wipe-testomgeving.ps1` gebruikt Windows-paden en WMI en noemt verouderde
   beheercommando's.
5. De twee `[project.scripts]`-entrypoints verwijzen naar `aliexpress.main:main`, maar
   die functie bestaat niet. Beide geïnstalleerde commando's geven momenteel een
   `ImportError`.
6. De lokale launcher heeft een vaste host en poort, opent de browser vóór aantoonbare
   server-readiness en heeft geen `--no-browser`-mogelijkheid voor headless Linux.
7. Lokale HTTP-uitvoering en gehoste HTTPS-productie worden onder dezelfde
   productieconfiguratie geschaard.
8. Schoolcodes en procesnamen worden directorynamen. Windows-reserved names,
   case-insensitive filesystems en Unicode-normalisatie kunnen daardoor per OS ander
   gedrag geven.
9. Er is geen CI die Windows, Ubuntu en macOS daadwerkelijk test.
10. Templates, static assets en persistente data worden vanaf de checkout-root gevonden.
    Dat is acceptabel voor de huidige `uv`-clone-scope, maar nog niet voor een later
    zelfstandig geïnstalleerd package.

## Correctie: Chromium op Ubuntu 26.04

De bestaande instructie is op deze repository daadwerkelijk kapot voor Ubuntu 26.04.
`pytest-playwright==0.8.0` heeft hier transitief Playwright 1.60.0 gelockt. Playwright
1.60 herkent `ubuntu26.04-x64` nog niet en stopt daarom vóór de download met:

```text
ERROR: Playwright does not support chromium on ubuntu26.04-x64
```

Officiële Ubuntu 26.04-ondersteuning is toegevoegd in Playwright 1.61. De actuele
stabiele Python-release tijdens het schrijven van dit plan is 1.62.0. De structurele
oplossing is daarom Playwright rechtstreeks als dev-dependency op 1.62.0 vastzetten,
het lockbestand vernieuwen en installatie plus browsertests op Ubuntu 26.04 uitvoeren.

`PLAYWRIGHT_HOST_PLATFORM_OVERRIDE=ubuntu24.04-x64` wordt niet de gedocumenteerde
oplossing: dat is een tijdelijke, onofficiële omweg die systeembibliotheekverschillen
kan maskeren.

Na de upgrade worden de commando's:

```bash
# Browserbinary; voldoende als de Linux-systeembibliotheken al aanwezig zijn.
uv run playwright install chromium

# Browserbinary plus OS-dependencies, bedoeld voor een verse Ubuntu-machine/CI.
uv run playwright install --with-deps chromium
```

Bronnen:

- [Playwright 1.61 release: Ubuntu 26.04 support](https://github.com/microsoft/playwright/releases/tag/v1.61.0)
- [Upstream Ubuntu 26.04 issue](https://github.com/microsoft/playwright/issues/40117)
- [Playwright browser- en dependency-installatie](https://playwright.dev/python/docs/browsers)
- [Playwright 1.62.0 op PyPI](https://pypi.org/project/playwright/1.62.0/)

## Uitvoeringsplan per slice

### Slice 1 — Playwright op Ubuntu 26.04 herstellen

Voorgestelde commit: `fix(dev): support Playwright on Ubuntu 26.04`

Probleem dat deze slice oplost: de browsertests kunnen op de doelmachine niet eens
worden geïnstalleerd.

Test-first/acceptatie:

1. Leg de huidige failure met Playwright 1.60 vast als reproduceerbare baseline.
2. Voeg `playwright==1.62.0` expliciet toe aan de dev-extra en vernieuw `uv.lock`.
3. Voer op Ubuntu 26.04 uit:
   - `uv sync --locked --extra dev`;
   - `uv run playwright install chromium`;
   - één kleine browsertest;
   - daarna de volledige browsersuite.
4. Pas de README pas aan nadat de commando's aantoonbaar werken.

Omdat browserinstallatie een omgevingsacceptatietest is, wordt hier geen kunstmatige
unit-test toegevoegd. De bestaande browsertests leveren de functionele regressietest.

### Slice 2 — Werkende canonieke CLI

Voorgestelde commit: `fix(cli): introduce a working ali-express command`

Probleem dat deze slice oplost: de geïnstalleerde entrypoints crashen direct.

Red:

- Voeg tests toe die de geregistreerde entrypoint laden.
- Laat `ali-express --help` en `ali-express solve` via een geïsoleerde CLI-runner lopen.
- De eerste test faalt op het ontbreken van `aliexpress.main.main`.

Green:

- Voeg een top-level Click-commandgroep toe.
- Laat `solve` de bestaande `distribute_students_once()` aanroepen.
- Maak de twee huidige aliases bewust gelijk of verwijder het overbodige alias.

Acceptatie:

- Het commando werkt vanuit de repo-root én vanuit een andere werkdirectory.
- De bestaande solvertests blijven groen.

### Slice 3 — Platformneutrale foreground-webserver

Voorgestelde commit: `feat(cli): add portable foreground web launcher`

Probleem dat deze slice oplost: starten is afhankelijk van batch, PowerShell en
`pythonw.exe`.

Red:

- CLI-tests voor `serve`, `--host`, `--port` en `--no-browser`.
- Tests dat de debugger en reloader niet actief zijn in lokale gebruikersmodus.
- Test dat een bezette poort een duidelijke fout geeft en geen onbekend proces doodt.
- Subprocess-smoketest: start op een vrije poort, vraag `/` op en stop netjes.

Green:

- Implementeer `uv run ali-express serve` op `127.0.0.1:5000`.
- Open de standaardbrowser pas nadat de server luistert.
- Ondersteun headless gebruik met `--no-browser`.
- Stop portabel met `Ctrl+C`.
- Laat `app.py` voor WSGI-imports bestaan, maar delegeer directe uitvoering aan dezelfde
  geteste launcher.

Acceptatie:

- Dezelfde opdracht werkt ongewijzigd in Bash, PowerShell en macOS Terminal.

### Slice 4 — Lokale HTTP-configuratie scheiden van productie-HTTPS

Voorgestelde commit: `fix(config): separate local and hosted environments`

Probleem dat deze slice oplost: `production` betekent nu zowel lokale HTTP-desktopmodus
als gehoste HTTPS-productie.

Red:

- Configuratietests voor local, development, testing en production.
- Een lokale login-smoketest die sessiegedrag over twee requests controleert.

Green:

- Voeg een expliciete lokale configuratie toe: debug/reloader uit en secure-cookie uit.
- Houd secure cookies verplicht in gehoste productie.
- Gebruik een projectspecifieke environmentnaam in plaats van Flask-intern gedrag te
  suggereren.
- Maak `.env`-resolutie onafhankelijk van de actuele werkdirectory.
- Voeg een `.env.example` zonder echte geheimen toe.

### Slice 5 — Veilige platformneutrale reset

Voorgestelde commit: `feat(cli): replace PowerShell reset with guarded command`

Probleem dat deze slice oplost: lokale testdata kan alleen met een Windows-script
worden gewist.

Red:

- Tests met een tijdelijke instance-directory voor bevestigen, annuleren, `--yes`, een
  lege omgeving en een actieve server.
- Test dat logs, configuratie en bestanden buiten de opgeloste doelen behouden blijven.
- Test dat een niet-lokale database-URL nooit als bestand wordt behandeld.

Green:

- Implementeer `ali-express reset-local-data`.
- Toon de volledig opgeloste doelpaden vóór bevestiging.
- Verwijder uitsluitend de lokale SQLite-database en storage-inhoud.
- Weiger bij een actieve instantie of onveilige/onverwachte configuratie.

Na acceptatie kan `wipe-testomgeving.ps1` verdwijnen.

### Slice 6 — Namen en opslagpaden op ieder OS hetzelfde behandelen

Voorgestelde commit: `fix(storage): enforce portable identifiers`

Probleem dat deze slice oplost: geldige Linux-directorynamen kunnen op Windows/macOS
ongeldig zijn of botsen.

Red:

- Tests voor `/`, `\\`, traversal en absolute paden.
- Tests voor Windows-reserved names zoals `CON`, `NUL`, `COM1` en `LPT1`.
- Tests voor `Klas` versus `klas`, Unicode-normalisatie en maximale lengte.
- Tests voor zowel procesnamen als via de beheer-CLI gemaakte schoolcodes.

Green:

- Centraliseer validatie en de genormaliseerde vergelijkingssleutel.
- Controleer ook de schooldirectory zelf op containment onder `STORAGE_DIR`.
- Valideer vóór database- of filesystemmutaties.
- Geef een begrijpelijke gebruikersfout bij afwijzing.

Een latere datamigratie naar opaque directory-ID's is robuuster, maar valt buiten deze
kleine slice tenzij bestaande namen aantoonbaar niet veilig gemigreerd kunnen worden.

### Slice 7 — Machinegebonden omlijsting verwijderen

Voorgestelde commit: `chore(platform): remove machine-specific launchers`

Probleem dat deze slice oplost: de checkout bevat een persoonlijke shortcut en
Windows-only operationele logica.

Wijzigingen:

- Verwijder `aliexpress.lnk` uit de huidige tree.
- Verwijder `kill-server.ps1`; de functie van `wipe-testomgeving.ps1` is in slice 5
  vervangen door `reset-local-data`.
- Verwijder `aliexpress.bat`, of reduceer deze na een expliciete UX-beslissing tot een
  dunne adapter die uitsluitend de canonieke Python-CLI start.
- Voeg `*.lnk` toe aan `.gitignore`.
- Voeg een pre-commitcontrole toe die Windows-shortcuts en absolute home/user-paden in
  nieuwe tracked bestanden blokkeert.
- Voeg zo nodig `.gitattributes` toe voor consistente line endings van resterende
  shell-adapters.

Verificatie:

- Een verse checkout bevat geen hardcoded gebruikers- of machinepad.
- Alle operationele handelingen zijn zonder de verwijderde scripts beschikbaar.

### Slice 8 — Documentatie voor Windows, Ubuntu en macOS

Voorgestelde commit: `docs(platform): document cross-platform workflows`

Probleem dat deze slice oplost: de README presenteert PowerShell en een Windows-shortcut
als de algemene workflow.

Wijzigingen:

- Documenteer installatie, start, stop, reset en tests met shell-neutrale commando's.
- Leg uit dat Playwright-Chromium alleen voor browsertests is; de app zelf gebruikt de
  standaardbrowser van het OS.
- Documenteer Ubuntu 26.04 en `--with-deps` expliciet.
- Corrigeer verouderde school/admincommando's.
- Leg het supportcontract en de grenzen ervan vast.
- Verplaats ongebruikte notebookdependencies uit de runtime-extra en controleer het
  lockbestand indien dat nog niet in een eerdere slice is gebeurd.

Verificatie:

- Laat de instructies uitvoeren vanuit verse Windows-, Ubuntu- en macOS-checkouts.

### Slice 9 — Cross-platform CI als regressiepoort

Voorgestelde commit: `ci: test supported operating systems and Python versions`

Probleem dat deze slice oplost: platformondersteuning is nu alleen een aanname.

Matrix:

- OS-lane op Python 3.13: `ubuntu-26.04`, `windows-latest`, `macos-latest`;
- Python-lane op Ubuntu: 3.11, 3.12 en 3.13;
- browserlane op `ubuntu-26.04` met Playwright Chromium;
- de realistische slow test alleen handmatig, gepland of vóór een release/merge.

Per normale OS-run:

```text
uv sync --locked --extra dev
uv run --locked pytest tests --ignore=tests/integration --ignore=tests/browser ...
uv run --locked pytest tests/integration -m "not slow" ...
```

Browserlane:

```text
uv run playwright install --with-deps chromium
uv run --locked pytest tests/browser ...
```

De platform-smoketest en de Windows-filelocktest moeten daadwerkelijk op hun eigen OS
draaien. De matrix wordt pas na een groene run als verplichte mergecheck ingesteld.

Ubuntu 26.04 is bij het schrijven van dit plan nog als preview runner beschikbaar;
queue- of image-instabiliteit wordt apart gehouden van applicatiefouten.

Bron:

- [GitHub-hosted Ubuntu 26.04 runner](https://github.com/actions/runner-images/issues/14226)

## PII uit de Git-geschiedenis verwijderen

### Voorlopige audit

De lokale audit van alle bereikbare refs toont:

- precies één historisch `aliexpress.lnk`-object;
- één commit waarin dit pad werd toegevoegd;
- bereikbaarheid via `origin/master`;
- geen `.env`-pad in de bereikbare objectlijst;
- één bewust gepubliceerd contactadres in de huidige README;
- normale Git-auteur-/committermetadata: twee unieke namen en één uniek e-mailadres.

De shortcut bevat ten minste een lokale gebruikersnaam, een absoluut Windows-pad en
machine-identificerende metadata. Exacte waarden worden niet in dit plan herhaald.

Deze audit is doelgericht en geen garantie dat elk vrij tekstveld of elk oud binair
bestand vrij van PII is. Voor de rewrite volgt daarom nog een bredere scan van blobs,
commitberichten, padnamen en metadata, waarbij rapportage waarden maskeert.

### Beslissingen die de eigenaar eerst moet nemen

1. Moet alleen de onbedoelde machine-PII uit `aliexpress.lnk` verdwijnen?
2. Moet het openbare contactadres uit de README ook uit alle historische versies worden
   verwijderd of is dit bewuste projectinformatie?
3. Moeten Git-auteur- en committergegevens worden herschreven? Standaardadvies: niet
   doen. Dit is normale broncode-attributie; herschrijven tast provenance en eventuele
   signatures aan.

### Wat wel haalbaar is

- Verwijder `aliexpress.lnk` uit iedere bereikbare commit met `git-filter-repo`.
- Vervang na expliciete keuze gerichte tekstwaarden in historische tekstblobs.
- Force-push de herschreven branches en tags naar GitHub.
- Verifieer in een verse clone dat het pad en de blob niet meer bereikbaar zijn.
- Laat actieve bijdragers opnieuw clonen of hun lokale geschiedenis zorgvuldig
  opschonen.
- Verwijder of herschrijf bestaande forks in overleg met hun eigenaren waar mogelijk.
- Voorkom herhaling met ignore- en pre-commitregels.

### Wat niet gegarandeerd kan worden

- Oude clones, backups, forks en downloads van anderen kunnen niet op afstand worden
  gewist.
- Oude commit-SHA's kunnen nog in caches, pull requests, logs of externe indexen staan.
- GitHub Support verwijdert gecachete data doorgaans alleen wanneer GitHub die als
  gevoelige data beschouwt; gewone persoons- of machine-informatie komt mogelijk niet
  voor support-purge in aanmerking.

### Veilige uitvoeringsprocedure

De historie-rewrite gebeurt **niet** als een gewone slice op de featurebranch. Zij wordt
pas uitgevoerd nadat de inhoudelijke branch is afgerond en bij voorkeur gemerged, zodat
niet halverwege alle commit-hashes veranderen.

1. Kondig een korte push-freeze aan.
2. Inventariseer branches, tags, forks, open PR's en branch protection.
3. Maak een afgeschermde backup/bundle; noteer dat deze backup de PII nog bevat.
4. Maak een verse mirror-clone, zoals `git-filter-repo` aanbeveelt.
5. Draai eerst alleen analyse en rapporteer gemaskeerde bevindingen.
6. Toon het exacte rewrite-commando en de geraakte refs aan de reviewer.
7. **Stop en vraag afzonderlijke expliciete toestemming voor de rewrite.**
8. Verwijder in de mirror ten minste het pad met conceptueel:

   ```text
   git filter-repo --sensitive-data-removal --invert-paths --path aliexpress.lnk
   ```

9. Verifieer objectlijst, refs, diffstat, tests en een verse lokale clone.
10. Toon het voorgenomen force-pushcommando en de remote refs.
11. **Stop opnieuw en vraag afzonderlijke expliciete toestemming voor de force-push.**
12. Force-push uitsluitend de vooraf gecontroleerde branches/tags.
13. Herstel branch protection, laat bijdragers reclonen en beoordeel forks/PR-caches.
14. Bewaar de PII-bevattende backup alleen zolang herstel noodzakelijk is en verwijder
    hem daarna gecontroleerd.

Gevolgen:

- iedere herschreven commit krijgt een nieuwe hash;
- open branches en PR's kunnen opnieuw gebaseerd moeten worden;
- commit signatures kunnen ongeldig worden;
- terugpushen van een oude clone kan de verwijderde objecten opnieuw publiceren.

Bronnen:

- [GitHub: Removing sensitive data from a repository](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)
- [`git-filter-repo` handleiding](https://github.com/newren/git-filter-repo/blob/master/Documentation/git-filter-repo.txt)

## Definitie van gereed

De repository is voor deze fase platformonafhankelijk wanneer:

- een verse clone op ieder ondersteund OS met de gedocumenteerde `uv`-commando's kan
  installeren;
- dezelfde CLI start, stopt en reset zonder OS-specifieke operationele logica;
- browserinstallatie en browsertests werken op Ubuntu 26.04;
- filesystemnamen en paden op ieder ondersteund OS hetzelfde worden gevalideerd;
- de volledige niet-trage CI-matrix groen is;
- er geen machinegebonden shortcut of absoluut gebruikerspad in de huidige tree staat;
- de goedgekeurde historie-rewrite is uitgevoerd en vanuit een verse clone is
  geverifieerd;
- iedere commit afzonderlijk door de eigenaar is gereviewd vóór hij werd gemaakt.
