# Routes opgesplitst in vier Flask Blueprints

`app.py` was gegroeid tot ~977 regels met alle routes, helpers en achtergrondthreads door elkaar. We hebben de code verdeeld over vier blueprints in `src/aliexpress/routes/`: `auth` (inloggen), `processes` (CRUD voor processen), `wizard` (de invoerfase stap voor stap) en `results` (verwerking en uitvoer). Alle blueprints delen één Flask-instantie en één databaseverbinding — er is geen microservices-split, want de app draait op één machine per school.

## Considered Options

**Losse modules zonder Blueprint**: routes in aparte `.py`-bestanden importeren in `app.py` via `app.add_url_rule`. Geeft geen URL-naamruimte; `url_for("upload_edexml")` werkt dan nog steeds globaal maar het wordt onduidelijk in welke module een endpoint leeft.

**Blueprints met URL-prefix per blueprint**: alle wizardroutes onder `/wizard/`, alle resultaatroutes onder `/results/`. Breekt bestaande bookmarks van leraren en past niet bij de lineaire user journey waarbij stappen op rootniveau zitten (`/groups_to`, `/result`).

## Consequences

Alleen `processes` krijgt een URL-prefix (`/processes`), omdat de routes daar van nature al onder dat pad leven. Auth, wizard en results registreren op rootniveau; hun endpoints worden wel volledig gekwalificeerd (`url_for("wizard.groups_to_page")`), zodat de eigenaarschap van een route altijd duidelijk is in de code.
