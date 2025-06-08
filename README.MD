# ALI Express

**Smart and Fair Student Group Allocation**
## Why use ALI Express?
ALI Express is designed to simplify and optimize the complex process of student group allocation. By considering various criteria and preferences, it ensures that the groupings are balanced and that as many student wishes as possible are fulfilled. This tool saves time and effort, making the process efficient and effective.

## Installation
```bash
conda env create -f environment.yml
```

Create a `.env` file that contains at least
```
FLASK_ENV=production
SECRET_KEY=<your-secret-key-here>
```

## Detailed description
ALI Express automatically takes into account many criteria of both group balance and student satisfaction, including positiive and negative preferences for current peers, future groups, including weights, impossiblities and minimal satisfaction. For group balance, this includes group size for both total and new groups, balance between boys and girls and  maximum cliques from current groups. Additionally, it allows to specify which students should not be together, or not in too large groups. 

## How to use
1. Fill out all the three input sheets: 
    * `voorkeuren.xlsx` for the preferences of the students,
    * `groepen.xlsx` for the current groups and
    * `niet_samen.xlsx` for which students should not be together (possibly in large combination)
1. Run the shortcut to the bat-file. It has the logo
1. Now a page starts in the browser, follow the steps there.

More technically inclined users may run `leerlingindeling.ipynb` or `main.py` for more technical feedback