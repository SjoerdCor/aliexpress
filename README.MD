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

The HiGHS solver (`highspy`) is installed automatically as a dependency. Run any command inside the environment with `uv run`, for example `uv run pytest tests/` to run the tests.

Create a `.env` file containing at least:

```
FLASK_ENV=production
SECRET_KEY=<your-secret-key-here>
```

## How it works

Start the app by double-clicking the ALI Express shortcut — a browser window opens automatically. Upload `voorkeuren.xlsx` and `groepen.xlsx`, optionally enter any "niet samen" constraints, and click Run.

ALI Express then solves the problem in three steps:

1. **Ensure every student gets at least one wish.** The solver first finds the *minimal* relaxation of the group-balance constraints under which every student can still fulfil at least one positive preference. This gives a single relaxation budget R\*: as tight as the data allows, no tighter.

2. **Set the group balance within that budget.** With R\* fixed, the solver distributes the allowed flexibility across all balance dimensions — group size, boy/girl ratio, and clique limits — as evenly as possible. No manual configuration required.

3. **Optimise student satisfaction.** Within the balance envelope from step 2, satisfaction is maximised using *plateaued lexicographic max-min*: the least satisfied student is raised as high as possible first, then the second least satisfied, and so on. The outcome is as fair as possible, not just good on average.

The result is shown in the browser; download the full Excel workbook for the school administration.

## What ALI Express takes into account

### Student preferences

- **Positive wishes** ("Graag met") — who a student wants to be with
- **Negative wishes** ("Liever niet met") — who a student wants to avoid
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
