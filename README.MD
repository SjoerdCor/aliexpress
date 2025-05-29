# ALI Express

**Smart and Fair Student Group Allocation**
## Why use ALI Express?
ALI Express is designed to simplify and optimize the complex process of student group allocation. By considering various criteria and preferences, it ensures that the groupings are balanced and that as many student wishes as possible are fulfilled. This tool saves time and effort, making the process efficient and effective.

## Installation

```bash
conda env create -f environment.yml
conda activate aliexpress
```

## Detailed description
ALI Express automatically takes into account many criteria of both group balance and student satisfaction, including positiive and negative preferences for current peers, future groups, including weights, impossiblities and minimal satisfaction. For group balance, this includes group size for both total and new groups, balance between boys and girls and  maximum cliques from current groups. Additionally, it allows to specify who should not be together, or not in too large groups. 

## How to use
1. Fill out all the three input sheets: 
  1. `voorkeuren.xlsx` for the preferences of the students,
  1. `groepen.xlsx` for the current groups and
  1. `niet_samen.xlsx` for which students should not be together (possibly in large combination)
1. Run ALI Express using `leerlingindeling.ipynb`
1. Look at the outgoing excel in the folder