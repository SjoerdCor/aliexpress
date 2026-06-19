# Routes opgesplitst in vijf Flask Blueprints

`app.py` was gegroeid tot ~977 regels met alle routes, helpers en achtergrondthreads door elkaar. We hebben de code verdeeld over vijf blueprints in `src/aliexpress/routes/`: `auth` (inloggen), `processes` (CRUD voor processen), `wizard` (de invoerfase stap voor stap), `results` (verwerking en uitvoer) en `admin` (beheerderslogin en impersonatie). Alle blueprints delen één Flask-instantie en één databaseverbinding — er is geen microservices-split, want de app draait op één machine per school.

De applicatiefactory `create_app()` leeft in `src/aliexpress/__init__.py` en registreert extensies, blueprints, error handlers en CLI-commando's. `app.py` in de projectroot is een dunne launcher van tien regels die de factory aanroept en de app beschikbaar stelt voor WSGI-servers en de desktop-snelkoppeling.

## Considered Options

**Losse modules zonder Blueprint**: routes in aparte `.py`-bestanden importeren in `app.py` via `app.add_url_rule`. Geeft geen URL-naamruimte; `url_for("upload_edexml")` werkt dan nog steeds globaal maar het wordt onduidelijk in welke module een endpoint leeft.

**Blueprints met URL-prefix per blueprint**: alle wizardroutes onder `/wizard/`, alle resultaatroutes onder `/results/`. Breekt bestaande bookmarks van leraren en past niet bij de lineaire user journey waarbij stappen op rootniveau zitten (`/groups_to`, `/result`).

**Factory in een aparte `app_factory.py`**: `create_app()` in een eigen module om `__init__.py` dun te houden. Voegt een extra bestand toe zonder een echte verantwoordelijkheidsgrens te trekken — `__init__.py` is de natuurlijke plek voor de package-factory.

## Consequences

Alleen `processes` krijgt een URL-prefix (`/processes`), omdat de routes daar van nature al onder dat pad leven. Auth, wizard, results en admin registreren op rootniveau; hun endpoints worden wel volledig gekwalificeerd (`url_for("wizard.groups_to_page")`), zodat de eigenaarschap van een route altijd duidelijk is in de code.

`create_app()` is idempotent bij herhaalde aanroepen in de testsuite: extensie-inits zijn veilig om meerdere keren te registreren, en de bestandslog-handler wordt overgeslagen wanneer `test_config` is meegegeven (de handler is niet idempotent).
