# ALI Express

**Smart and Fair Student Group Allocation**

## Why use ALI Express?

ALI Express is designed to simplify and optimise the complex process of student group allocation. By taking into account a wide range of student preferences and group-balance criteria, it ensures that every student gets at least one fulfilled wish while keeping groups well-balanced — automatically and without manual tuning. This saves significant time and effort compared to doing it by hand, and produces a measurably fair outcome.

## Installation

ALI Express is managed with [uv](https://docs.astral.sh/uv/). Install uv and git, clone this repository, and install the dependencies:

```bash
uv sync
```

Developers should also install the development tools and pre-commit hooks:

```bash
uv sync --extra dev
uv run pre-commit install
```

The CP-SAT solver (`ortools`) is installed automatically as a dependency. Run any command inside the environment with `uv run`, for example `uv run pytest tests/` to run the tests.

The `tests/browser/` suite uses Playwright to drive the real app and cover the page
JavaScript. It needs the browser installed once:

```bash
uv run playwright install chromium
```

The quick loop skips the slow suites with
`uv run pytest tests/ --ignore=tests/integration --ignore=tests/browser`.

Create a `.env` file containing at least:

```
FLASK_ENV=production
SECRET_KEY=<your-secret-key-here>
ADMIN_PASSWORD=<your-admin-password-here>
```

`SECRET_KEY` signs the login session cookie. In production the app refuses to start
without one; in development a fallback key is used so you can run it without a `.env`.

`ADMIN_PASSWORD` is the password for the single administrator account, which is created
automatically on startup — there is no admin sign-up or CLI command. The app refuses to
start if it is missing or too weak (in both development and production; there is no
fallback). To rotate it, edit `.env` and restart: the stored hash is synced to the new
value on every startup. The administrator logs in at `/admin/login` with this password
only (no username).

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
command line.

```bash
uv run flask schools add <schoolcode>
```

`schoolcode` is the short, unique login identifier the school types (a username). The command
then prompts for the school's full display name (shown in the interface) and a password,
entered twice and hidden. The password is stored as a hash, never in plain text.

Teachers log in at `/login` with that schoolcode and password. They obtain their credentials
from the administrator who set up ALI Express for their school. The session ends when the
browser is closed.

## How it works

Start the app by double-clicking the ALI Express shortcut — a browser window opens automatically. Upload `voorkeuren.xlsx` and `groepen.xlsx`, optionally enter any "niet samen" constraints, and click Run.

ALI Express then solves the problem in three steps:

1. **Ensure every student reaches strictly positive satisfaction.** The solver first finds the *minimal* relaxation of the group-balance constraints under which as many students as possible end up with satisfaction above zero — a pure-positive student clears this with at least one honored preference, while a student with avoid-preferences clears it only if none of them is violated. This gives a single relaxation budget R\*: as tight as the data allows, no tighter. (See [ADR-0015](docs/adr/0015-relaxatievloer-strikt-positieve-tevredenheid.md).)

2. **Set the group balance within that budget.** With R\* fixed, the solver distributes the allowed flexibility across all balance dimensions — group size, boy/girl ratio, and clique limits — as evenly as possible. No manual configuration required.

3. **Optimise student satisfaction.** Within the balance envelope from step 2, satisfaction is maximised using *plateaued lexicographic max-min*: the least satisfied student is raised as high as possible first, then the second least satisfied, and so on. The outcome is as fair as possible, not just good on average.

The result is shown in the browser; download the full Excel workbook for the school administration.

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

### Special constraints

- **"Niet samen" rules** — specified pairs or groups of students must not be placed together, or at most a given number of them may end up in the same group

## Known limitations

- **Retained students are counted only toward total-group balance, not cohort balance.** Marking a student as staying ("blijft in groep") adjusts the destination group's total boy/girl counts, but does not add them to the redistributed cohort. For a retained student in a year-layer *between* the cohort and the destination — e.g. a group-4 repeater in a 4/5/6 class, onto whom the distributed group-3 cohort is added — only the total-group effect is modelled, not the effect on the cohort's own balance. Forced placement of an individual student into a chosen group is likewise not supported.

## Copyright

Copyright (c) 2024–2026 Sjoerd Cornelissen. All rights reserved.

The source code is not licensed for use, copying, modification, or distribution.
Interested in using or building on ALI Express? I'd love to hear from you! Reach out at info@sjoerdcornelissen.com — I'm sure we can make it work.
