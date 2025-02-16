"""Defines helper functions that implement the important bitwise operations in PuLP"""

import functools
import pulp


def prepare_logical_operations(func):
    """Decorator to validate that all *vars are binary variables and create result_var if needed."""

    @functools.wraps(func)
    def wrapper(prob, *pulp_vars, result_var=None):
        for var in pulp_vars:
            if not isinstance(var, pulp.LpVariable):
                raise ValueError(
                    f"Variable {var} is not a PuLP variable but {type(var)}"
                )
            if var.cat != pulp.LpBinary and not (
                var.cat == pulp.LpInteger and var.lowBound == 0 and var.upBound == 1
            ):
                print(var.cat, var.lowBound, var.upBound)
                raise ValueError(f"Variable {var} is not a binary, but {var.cat}")

        if result_var is None:
            result_var = pulp.LpVariable(
                f"{func.__name__}_" + "_".join([v.name for v in pulp_vars]),
                cat="binary",
            )

        return func(prob, *pulp_vars, result_var=result_var)

    return wrapper


# pylint: disable=invalid-name; capitals to distinguis from builtins
@prepare_logical_operations
def AND(prob, *pulp_vars, result_var=None):
    """Applies an AND constraint in PuLP for multiple variables and returns the result variable."""
    for var in pulp_vars:
        prob += result_var <= var
    prob += result_var >= pulp.lpSum(pulp_vars) - (len(pulp_vars) - 1)
    return result_var


def XNOR(
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


def NAND(
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
