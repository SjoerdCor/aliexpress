"""Defines helper functions that implement the important bitwise operations in PuLP"""

import pulp


def xnor(prob, var1, var2):
    """Applies an XNOR constraint in PuLP and returns the result variable."""
    result_var = pulp.LpVariable(f"xnor_{var1.name}_{var2.name}", cat="Binary")
    prob += result_var >= 1 - var1 - var2
    prob += result_var <= 1 + var1 - var2
    prob += result_var <= 1 - var1 + var2
    prob += result_var >= var1 + var2 - 1
    return result_var
