# ALI Express

**Smart and Fair Student Group Allocation**
## Why use ALI Express?
ALI Express is designed to simplify and optimize the complex process of student group allocation. By considering various criteria and preferences, it ensures that the groupings are balanced and that as many student wishes as possible are fulfilled. This tool saves time and effort, making the process efficient and effective.

## Installation
ALI Express is managed with [uv](https://docs.astral.sh/uv/). Install uv and git, clone this repository, and then install the dependencies:

```bash
uv sync
```

Developers should install the development tools (pytest, pylint, black, ...) and the pre-commit hooks as well:

```bash
uv sync --extra dev
uv run pre-commit install
```

The HiGHS solver (`highspy`) is installed automatically as a dependency. Run any command inside the environment with `uv run`, for example `uv run pytest tests/` to run the tests.

Alternatively, the project can still be installed with conda via `conda env create -f environment.yml` (or `environment_dev.yml` for developers).

Create a `.env` file that contains at least
```
FLASK_ENV=production
SECRET_KEY=<your-secret-key-here>
```

Following the How to use-section, a local web app launches which details a step-by-step plan to get to the allocation of the students.


## Detailed description
ALI Express automatically takes into account many criteria of both group balance and student satisfaction, including positiive and negative preferences for current peers, future groups, including weights, impossiblities and minimal satisfaction. For group balance, this includes group size for both total and new groups, balance between boys and girls and  maximum cliques from current groups. Additionally, it allows to specify which students should not be together, or not in too large groups.

## How to use

1. Run the shortcut to the bat-file in the home flder. It has the logo of ALI Express.
1. Now a page starts in the browser, follow the steps there.

More technically inclined users may run `leerlingindeling.ipynb` or `main.py` for more technical feedback. This requires manually oploading the following files:
    * `voorkeuren.xlsx` for the preferences of the students,
    * `groepen.xlsx` for the current groups and
    * `niet_samen.xlsx` for which students should not be together (possibly in large combination)