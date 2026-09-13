# ALI Express

[![CI](https://github.com/SjoerdCor/aliexpress/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/SjoerdCor/aliexpress/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/SjoerdCor/aliexpress/branch/master/graph/badge.svg)](https://codecov.io/gh/SjoerdCor/aliexpress)

**Smart and Fair Student Group Allocation**

## Why use ALI Express?

ALI Express is designed to simplify and optimise the complex process of student group allocation. By taking into account a wide range of student preferences and group-balance criteria, it ensures that every student gets at least one fulfilled wish while keeping groups well-balanced — automatically and without manual tuning. This saves significant time and effort compared to doing it by hand, and produces a measurably fair outcome.

## Installation

ALI Express is managed with [uv](https://docs.astral.sh/uv/). Install `uv` and Git,
clone the repository, and run the commands below from the checkout root. The lockfile
supports CPython 3.11, 3.12 and 3.13.

```bash
git clone https://github.com/SjoerdCor/aliexpress.git aliexpress
cd aliexpress
uv sync --locked
```

The CP-SAT solver (`ortools`) is installed automatically. Run project commands through
`uv`; no manually activated virtual environment is required. Developers and contributors
should install the development tools as well:

```bash
uv sync --locked --extra dev
uv run --locked pre-commit install
uv run --locked pre-commit install --hook-type pre-push
```

Create `.env` in the checkout root before starting the application:

```
ALIEXPRESS_ENV=local
SECRET_KEY=<a-long-random-secret>
ADMIN_PASSWORD=<a-long-random-admin-password>
```

`local` is the default for the foreground `ali-express serve` command and is intended
for HTTP on the local machine. It deliberately disables debug mode, the reloader and
secure-only cookies. Direct WSGI factory use defaults safely to `production`; hosted
deployments should still set `ALIEXPRESS_ENV=production` explicitly and run behind
HTTPS. `SECRET_KEY` and `ADMIN_PASSWORD` are required in both environments; there is no
shared development fallback. `DATABASE_URL` is optional and defaults to the local
SQLite database in `instance/app.db`.

The `tests/browser/` suite uses Playwright to drive the real app and test its JavaScript.
Playwright Chromium is a test dependency only: the application itself opens the operating
system's default browser and does not require Chromium.

Install Chromium once for browser tests:

```bash
uv run --locked playwright install chromium
```

On a fresh Ubuntu 26.04 machine, install Chromium together with its OS dependencies:

```bash
uv run --locked playwright install --with-deps chromium
```

The `--with-deps` option is intended for Ubuntu/Linux. Windows and macOS normally use the
first command; the app itself works with the standard browser configured by the OS.

## Support contract

The first-class support target is a fresh Git checkout using `uv` on:

- Windows x64;
- Ubuntu x64, including Ubuntu 26.04;
- macOS x64 and ARM64;
- CPython 3.11, 3.12 and 3.13.

Windows ARM is not supported because the pinned OR-Tools version has no Windows ARM
wheel. Python 3.14, a standalone desktop installer/package and a background service are
outside the current support contract. The local server runs in the foreground and stops
with `Ctrl+C`.

## Testing

Run the smallest relevant test first. After a failure, rerun the last failures with
`--lf`:

```bash
uv run --locked pytest tests/test_betreffend_bestand.py -q --no-cov -x
uv run --locked pytest tests --ignore=tests/integration --ignore=tests/browser -q --no-cov --lf
```

The fast Python/Flask suite explicitly excludes browser and integration tests. Tests that
start the real CP-SAT solver are kept in a separate sequential lane; the remaining tests use
four pytest workers with the `load` scheduler:

```bash
uv run --locked pytest tests --ignore=tests/integration --ignore=tests/browser \
  -q --no-cov -m "not slow and not real_solver" -n 4 --dist load
uv run --locked pytest tests --ignore=tests/integration --ignore=tests/browser \
  -q --no-cov -m "not slow and real_solver" -n 0
```

Run the browser suite separately. Only its non-solver selection uses xdist; real-solver
browser tests run sequentially:

```bash
uv run --locked pytest tests/browser -q --no-cov \
  -m "not slow and not real_solver" -n 2 --dist load
uv run --locked pytest tests/browser -q --no-cov \
  -m "not slow and real_solver" -n 0
```

Run non-slow integration tests in their own sequential lane:

```bash
uv run --locked pytest tests/integration -q --no-cov -m "not slow" -n 0
```

For full checks, coverage and the default test selection are enabled:

```bash
uv run --locked pytest tests/integration
uv run --locked pytest tests
uv run --locked pytest tests -m slow
```

`uv run --locked pytest tests` is the complete non-slow suite, including browser and
integration tests, and is the full sequential control run. The global pytest configuration
uses `-m "not slow"` to exclude only the marked slow test; it does not exclude browser or
other integration tests. Use the explicit `--ignore` options and marker selections above
for resource-aware local feedback. Do not combine all suites in one `pytest tests -n N`
xdist pool: browser processes and real solver threads have different resource profiles.
The worker counts above are measured baselines for the current suite and CI hardware, not
universal defaults.

The pre-push hook runs this full control suite with the 90% coverage gate before a branch is
pushed. Install it with `uv run --locked pre-commit install --hook-type pre-push` as shown
above. The hook needs the same local browser and loopback-server prerequisites as the full
test suite.

CI requires at least 90% combined line coverage. Codecov also checks that project coverage
does not decrease by more than 0.5 percentage points compared with the pull request's base
commit.

In CI, Quality, Fast tests, Browser tests, Integration tests and the slow acceptance test
run as separate jobs on the primary Python version. The ordinary fast and browser selections
may run in parallel on their own runners, while each real-solver selection remains sequential
within its runner. A small compatibility matrix additionally validates Windows x64, Ubuntu
26.04 x64, macOS x64 and macOS ARM64 on Python 3.13, plus CPython 3.11 and 3.12 on Ubuntu.
This is intentional dimensional coverage, not a full OS-by-Python Cartesian product. CI
combines coverage from the fast, browser and integration jobs and publishes JUnit reports;
the `Tests` job is the stable aggregate gate for the required test jobs.

## CI and merge workflow

Develop and commit changes on a feature branch. Push that branch and open a pull request
to `master`; do not commit or push directly to `master`. GitHub Actions runs for pull requests
targeting `master` and for pushes to `master`; the workflow is defined in
[`.github/workflows/ci.yml`](.github/workflows/ci.yml).

Run the local checks described in the [Testing](#testing) section before opening a pull
request. Merge only after the required CI checks pass. Required-check and merge-policy
settings are configured separately in GitHub branch protection or rulesets.

`SECRET_KEY` signs the login session cookie. `ADMIN_PASSWORD` is the password for the
single administrator account, which is created automatically on startup — there is no
admin sign-up command. The app refuses to start when either required value is missing or
the admin password is too weak. To rotate the password, edit `.env` and restart; the
stored hash is synced to the new value. The administrator logs in at `/admin/login` with
this password only (no username).

## Logging

In production the app writes to `instance/logs/aliexpress.log`. The file rotates at
midnight and keeps 90 days of backups. Each line carries correlation fields so you can
filter on a single run or school:

```
2025-01-15 09:31:04,123 INFO [schoolX/Groep4/7 solve] aliexpress.main MainThread: All files read
2025-01-15 09:31:04,456 INFO [schoolX/Groep4/7 solve] aliexpress.solver.strategies MainThread: lexmaxmin level 0: Optimal in 1.23s, plateau=18
```

Fields in brackets: `school / process / run phase`. A dash (`-`) means no active context
(e.g. an unauthenticated request). No student names appear in any log file.

## Schools and login

Every data route requires logging in as a school, so student data is never reachable without
authentication. There is no self-registration: an administrator creates each school from the
command line. The command generates a temporary password and shows it once; the school must
change it at the first login.

```bash
uv run --locked flask schools add <schoolcode> --naam "Naam van de school"
```

Omit `--naam` to enter the display name interactively. `schoolcode` is the short, unique
login identifier the school types. It must be a portable single name: letters, numbers,
spaces, `-` and `_`, at most 64 characters, and not a Windows-reserved name such as `CON`,
`NUL`, `COM1` or `LPT1`. Passwords are stored as hashes, never in plain text.

Teachers log in at `/login` with that schoolcode and password. They obtain their credentials
from the administrator who set up ALI Express for their school. The session ends when the
browser is closed.

## Manual testing with synthetic scenarios

Start the app from the checkout root:

```bash
uv run --locked ali-express serve
```

The command listens on `127.0.0.1:5000`, opens the OS default browser after the server is
ready, and remains in the foreground. Stop it with `Ctrl+C`. For a headless machine, or
when you want to open the browser yourself, use:

```bash
uv run --locked ali-express serve --no-browser
```

Use `--host` and `--port` to select a different listening address or port. A busy port is
reported without stopping another process. The compatibility launcher `app.py` remains
available for WSGI imports, but the foreground CLI above is the supported local workflow.

If the database is empty, open a second terminal, create a school, and log in with schoolcode
`testschool`:

```bash
uv run --locked flask schools add testschool --naam "Testschool"
```

Then create the synthetic scenarios:

```bash
# carry-over, 35 students by default
uv run --locked python -m tests.testdata testschool doorzetten-2
# redistribute, 35 students
uv run --locked python -m tests.testdata testschool herindelen-2 herindelen 35
# redistribute with carry-over, 72 students
uv run --locked python -m tests.testdata testschool herdoor-2 herindelen_met_doorzetten 72
```

The optional final argument is the number of students: these examples use `35` and `72`; the
default is `35` when the argument is omitted. The processes already include the **Niet-samen**
constraints and can be run directly through **Opslaan & door naar indelen** and
**Start verdeling**. In all three modes, generated preferences mostly target classmates from
the same current group, with occasional cross-group and destination-group preferences.
Positive preferences are more likely to be reciprocal at higher intensities. All generated
weights use the fixed intensity levels available in the web form.

## Local commands

The supported operational commands are platform-neutral and work from Bash, PowerShell or
macOS Terminal:

```bash
# Open the local app and browser
uv run --locked ali-express serve

# Run without opening a browser; Ctrl+C still stops the foreground server
uv run --locked ali-express serve --no-browser

# Solve the default voorkeuren.xlsx and groepen.xlsx in the checkout
uv run --locked ali-express solve

# Show the resolved deletion targets and ask for confirmation
uv run --locked ali-express reset-local-data

# Explicit confirmation for an already reviewed local environment
uv run --locked ali-express reset-local-data --yes
```

Stop the server before resetting local data. `reset-local-data` only accepts the local
environment and a local SQLite file; it refuses an external database, unsafe paths or an
active instance. It deletes the SQLite database (including its sidecar files) and the
contents of `instance/storage`, while preserving the directories themselves. The command
prints the resolved targets before asking for confirmation.

The portable identifier constraints add normalized key columns to the database schema.
When updating an existing checkout from the earlier schema and its data does not need to
be preserved, run `reset-local-data` once before starting the updated application.

## How it works

After starting the local server, open the displayed URL if the browser did not open
automatically. Upload `voorkeuren.xlsx` and `groepen.xlsx`, optionally enter any "niet
samen" constraints, and click Run.

ALI Express then solves the problem in three steps:

1. **Ensure every student reaches strictly positive satisfaction.** The solver first finds the *minimal* relaxation of the group-balance constraints under which as many students as possible end up with satisfaction above zero — a pure-positive student clears this with at least one honored preference, while a student with avoid-preferences clears it only if none of them is violated. This gives a single relaxation budget R\*: as tight as the data allows, no tighter. (See [ADR-0015](docs/adr/0015-relaxatievloer-strikt-positieve-tevredenheid.md).)

2. **Set the group balance within that budget.** With R\* fixed, the solver distributes the allowed flexibility across all balance dimensions — group size, boy/girl ratio, and clique limits — as evenly as possible. With the defaults, no manual configuration is required; optional Balansgrenzen can cap how far this relaxation may go.

3. **Optimise student satisfaction.** Within the balance envelope from step 2, satisfaction is maximised using *plateaued lexicographic max-min*: the least satisfied student is raised as high as possible first, then the second least satisfied, and so on. The outcome is as fair as possible, not just good on average.

The result is shown in the browser as **group cards**: each destination group is a card with
its totals, split per jaarlaag into a Jongens and a Meisjes column of name chips. For the two
forward modes (Doorzetten and Herindelen met doorzetten) the jaarlaag shown on the cards, in the
klassenoverzicht and in the Excel is the *new* one — the year-layer the students move up to after
the transition — while the input asks for their current jaarlaag. Clicking a chip
opens a popover with the student's full name, origin class, satisfaction, honored/violated
preferences and any extra-certainty guarantee. Below the cards a compact **klassenoverzicht** shows
the counts and the balance (size difference and boy/girl imbalance) per group, followed by three
analysis tables (transition matrix, per-student satisfaction, fulfilled preferences). This
structured view is built in the reporting layer as a plain, JSON-serialisable view-model — see
[ADR-0016](docs/adr/0016-groepsindeling-gestructureerde-viewmodel.md). Download the full Excel
workbook for the school administration.

### While it solves

The processing page shown during a run always displays a calm core: a three-step stepper for
the stages above, the input overview (student/group counts), the sociogram link, and a rough
progress line ("expected to take another ~X, rough estimate"). The estimate starts as a static
sentence while step 2 (balance) is still running, then turns into a concrete number once the
solver has enough data: a first estimate after the balance step finishes, refined after every
completed satisfaction round to the longest round duration seen so far. It is deliberately an
overestimate rather than an underestimate.

Richer progress detail — plateau messages ("least satisfied student: now at X%"), the
tie-break notice, and a live preview of the interim group allocation — only appears once the
run looks like it will take a while: either the estimate predicts more than 45 seconds
remaining, or 45 seconds have actually elapsed. Most runs finish well within that window, so
this detail never appears and the page stays quiet; longer runs get the full picture as soon
as it is known to be worth showing.

Once revealed, the sociogram link and the live preview are grouped under a "Terwijl je wacht"
("While you wait") heading, appearing in that order as each becomes ready. The live preview
sits behind a "Voorlopige indeling" ("Tentative distribution") disclosure that starts collapsed
— with "wordt nog verbeterd..." ("still being improved...") as its subtext — so the tentative
distribution doesn't compete for attention with the solve stepper above it; the teacher can open
it to check progress, and it stays open across updates as the solver keeps refining the result.

The Sociogram is a browser-side view of the canonical preferences. Its Cytoscape.js runtime is
bundled locally, so opening the Sociogram works offline and sends no student data to a CDN or
other external service; the version and MIT licence are recorded in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Distribution modes

When creating a process, the teacher picks one of three modes. The choice shapes the wizard
and how the moving cohort and destination groups are determined, but all three run on the
*same* solver and balance engine — only the data preparation differs.

- **Doorzetten** (carry-over) — one incoming year layer is distributed over the combined
  groups of the next building phase; the older students who stay form the fixed occupancy of
  those destination groups.
- **Herindelen met dezelfde groepen** (redistribute in place) — the current students of the
  chosen groups are redistributed over those *same* groups (source and destination coincide),
  balanced per year layer.
- **Herindelen met doorzetten** (redistribute with progression) — the students of chosen year
  layers (e.g. 5-6-7) are distributed over separately chosen, empty-starting destination groups
  (the current 6-7-8 groups) as the cohorts move up a year at the school-year transition. Here
  the source groups differ from the destinations; the flow mirrors doorzetten (pick the year
  layers on upload, then "who takes part", then the destination groups).

The year layer comes from the EDEXML export for both redistribute modes.

## What ALI Express takes into account

### Student preferences

- **Positive wishes** ("Graag met") — who a student wants to be with
- **Negative wishes** ("Liever niet met") — who a student wants to avoid. Being kept away from everyone counts as full satisfaction (+100%); a violated avoid-wish drops the student into negative satisfaction, so 0% is not the floor.
- **Forbidden groups** ("Niet in") — groups a student cannot join (e.g. a sibling is already placed there)
- **Preference weights** — not all wishes carry equal importance; weighted preferences are supported
- **Minimum satisfaction** — a per-student guarantee that a student reaches at least a specified satisfaction level

### Group balance

- **Group size** — the spread between the largest and smallest group is minimised, both for the new cohort alone and for the total group (including students already in the group)
- **Boy/girl ratio** — balanced within the new cohort and in the total group
- **Clique limits** — at most N students from the same previous class end up together, preventing large social clusters; enforced separately for boys and girls

Optional **Balansgrenzen** are filled with generous, input-dependent defaults on the processing page. **Onbeperkt** removes the upper bound for that balance family; it does not disable balance optimisation. If selected bounds make a distribution infeasible, the page suggests a jointly feasible relaxation.

### Special constraints

- **"Niet samen" rules** — specified pairs or groups of students must not be placed together, or at most a given number of them may end up in the same group

## Known limitations

- **Retained students are counted only toward total-group balance, not cohort balance.** Marking a student as staying ("blijft in groep") adjusts the destination group's total boy/girl counts, but does not add them to the redistributed cohort. For a retained student in a year-layer *between* the cohort and the destination — e.g. a group-4 repeater in a 4/5/6 class, onto whom the distributed group-3 cohort is added — only the total-group effect is modelled, not the effect on the cohort's own balance. Forced placement of an individual student into a chosen group is likewise not supported.

## Copyright

Copyright (c) 2024–2026 Sjoerd Cornelissen. All rights reserved.

The source code is not licensed for use, copying, modification, or distribution.
Third-party components bundled with the application remain under their own licenses;
see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). In particular, the locally
vendored Cytoscape.js browser library is available under the MIT License.
Interested in using or building on ALI Express? I'd love to hear from you! Reach out at info@sjoerdcornelissen.com — I'm sure we can make it work.
