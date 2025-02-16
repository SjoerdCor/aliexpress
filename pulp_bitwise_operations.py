"""Defines helper functions that implement the important bitwise operations in PuLP"""

import pulp


def xnor(
    prob: pulp.pulp.LpProblem, var1: pulp.pulp.LpVariable, var2: pulp.pulp.LpVariable
) -> pulp.pulp.LpVariable:
    """Applies an XNOR constraint in PuLP to var1 and 2

    Parameters
    ----------
    prob : pulp.LpProblem
        The pulp problem to which the constraint must be added
    var1 : pulp.LpVariable
        The first variable in the XNOR constraint (must be binary)
    var2 : pulp.LpVariable
        The second variable in the XNOR constraint (must be binary)

    Returns
    -------
    pulp.LpVariable
        The resulting variable that contains the constraint

    """
    result_var = pulp.LpVariable(f"xnor_{var1.name}_{var2.name}", cat="Binary")
    prob += result_var >= 1 - var1 - var2
    prob += result_var <= 1 + var1 - var2
    prob += result_var <= 1 - var1 + var2
    prob += result_var >= var1 + var2 - 1
    return result_var


def nand(
    prob: pulp.pulp.LpProblem, var1: pulp.pulp.LpVariable, var2: pulp.pulp.LpVariable
) -> pulp.pulp.LpVariable:
    """Applies a NAND constraint

    Parameters
    ----------
    prob : pulp.LpProblem
        The pulp problem to which the constraint must be added
    var1 : pulp.LpVariable
        The first variable in the XNOR constraint (must be binary)
    var2 : pulp.LpVariable
        The second variable in the XNOR constraint (must be binary)

    Returns
    -------
    pulp.LpVariable
        The resulting variable that contains the constraint
    """
    result_var = pulp.LpVariable(f"nand_{var1.name}_{var2.name}", cat="Binary")
    prob += result_var >= 1 - var1
    prob += result_var >= 1 - var2
    prob += result_var <= 2 - var1 - var2
    return result_var
