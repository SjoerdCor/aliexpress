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
            if (var.cat != pulp.LpBinary and var.cat != "binary") and not (
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
    """Applies an AND constraint in PuLP for multiple variables and returns the result variable.

    Parameters
    ----------
    prob : pulp.LpProblem
        The pulp problem to which the constraint must be added
    vars : pulp.LpVariable
        The variables (>=2) for which the constraint must be added
    result_var : pulp.LpVariable (optional)
        The existing variable which specifies the constraint. A new one is added
        dynamically if not given

    Returns
    -------
    pulp.LpVariable
        The resulting variable that contains the constraint
    """
    for var in pulp_vars:
        prob += result_var <= var
    prob += result_var >= pulp.lpSum(pulp_vars) - (len(pulp_vars) - 1)
    return result_var


@prepare_logical_operations
def OR(prob, *pulp_vars, result_var=None):
    """Applies an OR constraint in PuLP for multiple variables and returns the result variable.

    Parameters
    ----------
    prob : pulp.LpProblem
        The pulp problem to which the constraint must be added
    vars : pulp.LpVariable
        The variables (>=2) for which the constraint must be added
    result_var : pulp.LpVariable (optional)
        The existing variable which specifies the constraint. A new one is added
        dynamically if not given

    Returns
    -------
    pulp.LpVariable
        The resulting variable that contains the constraint
    """
    for var in pulp_vars:
        prob += result_var >= var
    prob += result_var <= pulp.lpSum(pulp_vars)
    return result_var


@prepare_logical_operations
def XOR(prob, *pulp_vars, result_var=None):
    """Applies XOR constraint in PuLP for multiple variables

    Parameters
    ----------
    prob : pulp.LpProblem
        The pulp problem to which the constraint must be added
    vars : pulp.LpVariable
        The variables (>=2) for which the constraint must be added
    result_var : pulp.LpVariable (optional)
        The existing variable which specifies the constraint. A new one is added
        dynamically if not given

    Returns
    -------
    pulp.LpVariable
        The resulting variable that contains the constraint
    """

    def _xor(prob, var1, var2):
        prob += result_var <= var1 + var2
        prob += result_var >= var1 - var2
        prob += result_var >= var2 - var1
        prob += result_var <= 2 - var1 - var2
        return prob, result_var

    if len(pulp_vars) < 2:
        raise ValueError("Only 1 var given")
    prob, result_var = _xor(prob, pulp_vars[0], pulp_vars[1])
    for new_var in pulp_vars[2:]:
        prob, result_var = _xor(prob, result_var, new_var)
    return result_var


@prepare_logical_operations
def NOT(prob, *pulp_vars, result_var=None):
    """Apply the NOT constraint

    Parameters
    ----------
    prob : pulp.LpProblem
        The pulp problem to which the constraint must be added
    vars : pulp.LpVariable
        Exactly one variable must be given
    result_var : pulp.LpVariable (optional)
        The existing variable which specifies the constraint. A new one is added
        dynamically if not given

    Returns
    -------
    pulp.LpVariable
        The resulting variable that contains the constraint

    """
    if len(pulp_vars) > 1:
        raise ValueError("More than one variable given")
    prob += result_var == 1 - pulp_vars[0]
    return result_var


@prepare_logical_operations
def NAND(prob, *pulp_vars, result_var=None):
    """Applies NAND constraint in PuLP for multiple variables

    Parameters
    ----------
    prob : pulp.LpProblem
        The pulp problem to which the constraint must be added
    vars : pulp.LpVariable
        The variables (>=2) for which the constraint must be added
    result_var : pulp.LpVariable (optional)
        The existing variable which specifies the constraint. A new one is added
        dynamically if not given

    Returns
    -------
    pulp.LpVariable
        The resulting variable that contains the constraint
    """
    and_result = AND(prob, *pulp_vars)
    return NOT(prob, and_result, result_var=result_var)


@prepare_logical_operations
def NOR(prob, *pulp_vars, result_var=None):
    """Applies NOR constraint in PuLP for multiple variables

    Parameters
    ----------
    prob : pulp.LpProblem
        The pulp problem to which the constraint must be added
    vars : pulp.LpVariable
        The variables (>=2) for which the constraint must be added
    result_var : pulp.LpVariable (optional)
        The existing variable which specifies the constraint. A new one is added
        dynamically if not given

    Returns
    -------
    pulp.LpVariable
        The resulting variable that contains the constraint
    """
    or_result = OR(prob, *pulp_vars)
    return NOT(prob, or_result, result_var=result_var)


@prepare_logical_operations
def XNOR(prob, *pulp_vars, result_var=None):
    """Applies XNOR constraint in PuLP for multiple variables

    Parameters
    ----------
    prob : pulp.LpProblem
        The pulp problem to which the constraint must be added
    vars : pulp.LpVariable
        The variables (>=2) for which the constraint must be added
    result_var : pulp.LpVariable (optional)
        The existing variable which specifies the constraint. A new one is added
        dynamically if not given

    Returns
    -------
    pulp.LpVariable
        The resulting variable that contains the constraint

    """
    xor_result = XOR(prob, *pulp_vars)
    return NOT(prob, xor_result, result_var=result_var)
