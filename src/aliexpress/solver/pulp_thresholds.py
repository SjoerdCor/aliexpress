"""Generic big-M indicator constraints for PuLP: threshold → binary variable."""

import pulp


def apply_threshold_constraint(  # pylint: disable=too-many-arguments  # LP interface: each arg is a distinct mathematical parameter
    prob, value, threshold, threshold_var=None, sense=">=", *, M=1_000_000, eps=1e-6
):
    """
    Adds a threshold-based indicator constraint to a PuLP problem.

    If no binary variable is provided, one is created automatically.

    Parameters
    ----------
    prob : pulp.LpProblem
        The linear programming problem to which constraints are added.
    value : pulp.LpVariable
        The continuous decision variable being compared to the threshold.
    threshold : float
        The threshold value.
    threshold_var : pulp.LpVariable or None, optional
        A binary variable indicating whether the threshold is satisfied.
        If None, a new binary variable will be created and returned.
    sense : str, optional (">=" or "<=")
        Direction of the constraint:
        - ">=" : threshold_var = 1 if value >= threshold
        - "<=" : threshold_var = 1 if value <= threshold
    M : float, optional
        Big-M constant for constraints.
    eps : float, optional
        Small epsilon to enforce strict inequalities.

    Returns
    -------
    pulp.LpVariable
        The binary variable associated with the threshold.
    """
    if threshold_var is None:
        threshold_var = pulp.LpVariable(
            f"thr_{value.name}_{sense}_{threshold}", lowBound=0, upBound=1, cat="Binary"
        )

    if sense == ">=":
        prob += value >= threshold - M * (1 - threshold_var)
        prob += value <= threshold - eps + M * threshold_var
    elif sense == "<=":
        prob += value <= threshold + M * (1 - threshold_var)
        prob += value >= threshold + eps - M * threshold_var
    else:
        raise ValueError("sense must be either '>=' or '<='")

    return threshold_var


def apply_threshold_constraints(  # pylint: disable=too-many-arguments  # LP interface: each arg is a distinct mathematical parameter
    prob, value, thresholds, threshold_vars, *, M=1_000_000, eps=1e-6
):
    """
    Adds threshold-based indicator constraints to a PuLP problem.

    This function ensures that each binary variable in `threshold_vars` correctly
    tracks whether `value` has met or exceeded a given threshold. It enforces
    logical conditions using big-M constraints to approximate indicator behavior.

    Parameters
    ----------
    prob : pulp.LpProblem
        The linear programming problem to which constraints are added.
    value : pulp.LpVariable
        The continuous decision variable being compared to thresholds.
    thresholds : iterable of float
        The threshold values that determine activation of binary variables.
    threshold_vars : dict of {float: pulp.LpVariable}
        A dictionary mapping each threshold to a corresponding binary variable.
    """

    for threshold in thresholds:
        if threshold > 0:
            prob += value >= threshold - M * (1 - threshold_vars[threshold])
            prob += value <= threshold - eps + M * threshold_vars[threshold]
        else:
            prob += value <= threshold + M * (1 - threshold_vars[threshold])
            prob += value >= threshold + eps - M * threshold_vars[threshold]
