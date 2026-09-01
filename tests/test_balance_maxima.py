"""Tests for the BalanceMaxima dataclass (per-family upper bound on relaxation)."""

import pytest

from aliexpress.solver._balance import BalanceMaxima, default_balance_maxima


def _students(*specs):
    """Build a students dict from (stamgroep, sex, count) triples.

    ``sex`` is "Jongen" or "Meisje". Names are synthetic and unique.
    """
    students = {}
    index = 0
    for stamgroep, sex, count in specs:
        for _ in range(count):
            students[f"s{index}"] = {"Stamgroep": stamgroep, "Jongen/meisje": sex}
            index += 1
    return students


def _groups(*totals_or_pairs):
    """Build a groups_to dict.

    Each argument is either an int (a total split evenly, extra boy first) or a
    (Jongens, Meisjes) pair.
    """
    groups = {}
    for i, item in enumerate(totals_or_pairs):
        if isinstance(item, tuple):
            boys, girls = item
        else:
            boys = (item + 1) // 2
            girls = item // 2
        groups[f"g{i}"] = {"Jongens": boys, "Meisjes": girls}
    return groups


def test_default_construction_allows_all_none():
    """Default construction leaves every family uncapped (all None)."""
    maxima = BalanceMaxima()
    assert maxima.max_diff_n_students_year is None
    assert maxima.max_diff_n_students_total is None
    assert maxima.max_imbalance_boys_girls_year is None
    assert maxima.max_imbalance_boys_girls_total is None
    assert maxima.max_clique is None
    assert maxima.max_clique_sex is None


def test_fully_populated_valid_instance():
    """A fully populated BalanceMaxima keeps every provided value."""
    maxima = BalanceMaxima(
        max_diff_n_students_year=2,
        max_diff_n_students_total=3,
        max_imbalance_boys_girls_year=2,
        max_imbalance_boys_girls_total=3,
        max_clique=5,
        max_clique_sex=3,
    )
    assert maxima.max_diff_n_students_year == 2
    assert maxima.max_diff_n_students_total == 3
    assert maxima.max_imbalance_boys_girls_year == 2
    assert maxima.max_imbalance_boys_girls_total == 3
    assert maxima.max_clique == 5
    assert maxima.max_clique_sex == 3


@pytest.mark.parametrize(
    "field",
    [
        "max_diff_n_students_year",
        "max_diff_n_students_total",
        "max_imbalance_boys_girls_year",
        "max_imbalance_boys_girls_total",
        "max_clique",
        "max_clique_sex",
    ],
)
def test_none_is_accepted_per_field(field):
    """None is accepted for any single field (that family stays Onbeperkt)."""
    # All other fields set to a valid int, the field under test set to None.
    kwargs = {
        "max_diff_n_students_year": 2,
        "max_diff_n_students_total": 3,
        "max_imbalance_boys_girls_year": 2,
        "max_imbalance_boys_girls_total": 3,
        "max_clique": 5,
        "max_clique_sex": 3,
    }
    kwargs[field] = None
    maxima = BalanceMaxima(**kwargs)
    assert getattr(maxima, field) is None


def test_rejects_zero():
    """A cap of zero is rejected (caps must be at least 1)."""
    with pytest.raises(ValueError):
        BalanceMaxima(max_clique=0)


def test_rejects_negative_int():
    """A negative cap is rejected."""
    with pytest.raises(ValueError):
        BalanceMaxima(max_clique=-1)


def test_rejects_non_int_float():
    """A non-integer float cap is rejected."""
    with pytest.raises(TypeError):
        BalanceMaxima(max_clique=2.5)


def test_rejects_non_int_str():
    """A string cap is rejected."""
    with pytest.raises(TypeError):
        BalanceMaxima(max_clique="5")


def test_factory_herindelen_all_occupancy_zero():
    """Empty groups: whole-group defaults fall back to the floor, per-year fixed at 3."""
    # Herindelen: every target group starts empty. The whole-group defaults
    # fall back to their generous floor (4); the per-year defaults are fixed (3).
    students = _students(("A", "Jongen", 4), ("A", "Meisje", 4))
    groups_to = _groups(0, 0, 0, 0)

    maxima = default_balance_maxima(students, groups_to)

    assert maxima.max_diff_n_students_total == 4
    assert maxima.max_imbalance_boys_girls_total == 4
    assert maxima.max_diff_n_students_year == 3
    assert maxima.max_imbalance_boys_girls_year == 3


def test_factory_doorzetten_occupancy_driven_totals():
    """Existing occupancy wider than the floor drives the whole-group defaults."""
    # Doorzetten: existing occupancy already forces a spread wider than the
    # floor. Totals span 10..4 (spread 6) and one group is boy/girl-lopsided by
    # 5, so both whole-group defaults track the occupancy, not the floor.
    #
    # Note: |Jongens-Meisjes| has the same parity as the group total, so an
    # imbalance of 5 needs an odd total. We keep the max total 10 and min total
    # 4 (spread 6) and give a third group an odd total 9 = 7 boys + 2 girls
    # (imbalance 5), which is where gender_total == 5 comes from.
    students = _students(("A", "Jongen", 2), ("B", "Meisje", 2))
    groups_to = _groups((5, 5), (2, 2), (7, 2), (3, 3))  # totals 10, 4, 9, 6

    maxima = default_balance_maxima(students, groups_to)

    assert maxima.max_diff_n_students_total == 6  # max(4, 10 - 4)
    assert maxima.max_imbalance_boys_girls_total == 5  # max(4, 5)


def test_factory_clique_scales_with_largest_stamgroep():
    """max_clique is twice the even-split floor of the largest Stamgroep."""
    # Largest stamgroep has 9 students over 4 groups: an even split floors at
    # ceil(9/4) = 3 per group; the default doubles that to leave room to relax.
    students = _students(
        ("big", "Jongen", 5), ("big", "Meisje", 4), ("small", "Jongen", 1)
    )
    groups_to = _groups(0, 0, 0, 0)

    maxima = default_balance_maxima(students, groups_to)

    assert maxima.max_clique == 6  # 2 * ceil(9 / 4)


def test_factory_clique_sex_scales_with_largest_same_sex_part():
    """max_clique_sex scales with the largest same-sex Stamgroep part."""
    # Stamgroep "rood": 7 boys + 2 girls. The largest same-sex stamgroep part is
    # 7 boys; over 4 groups an even split floors at ceil(7/4) = 2, doubled to 4.
    students = _students(("rood", "Jongen", 7), ("rood", "Meisje", 2))
    groups_to = _groups(0, 0, 0, 0)

    maxima = default_balance_maxima(students, groups_to)

    assert maxima.max_clique_sex == 4  # 2 * ceil(7 / 4)


def test_constrains_anything_false_when_all_none():
    """constrains_anything is False when no family is capped."""
    assert not BalanceMaxima().constrains_anything()


def test_constrains_anything_true_when_any_field_set():
    """constrains_anything is True once any family is capped."""
    assert BalanceMaxima(max_clique=1).constrains_anything()
