"""Tests for the integer weight scale used by the CP-SAT model builder.

CP-SAT reasons over integers only; preference weights are user-entered floats.
``weight_scale`` returns the smallest factor that turns every weight into a whole
number, under the documented contract that weights are exact to 3 decimals.
"""

import pytest

from aliexpress.solver.scaling import weight_scale


def test_form_weight_set_scales_by_two():
    """The web form's weight set {0.5, 1, 2, 5} needs exactly a factor 2."""
    assert weight_scale([0.5, 1.0, 2.0, 5.0]) == 2


def test_whole_weights_need_no_scaling():
    """Weights that are already whole numbers keep scale 1."""
    assert weight_scale([1.0, 2.0, 3.0]) == 1


def test_arbitrary_decimals_get_a_data_driven_scale():
    """Excel allows any positive float; the scale adapts to the decimals present."""
    assert weight_scale([0.3, 1.0]) == 10
    assert weight_scale([0.125]) == 8
    assert weight_scale([0.25, 0.75]) == 4


def test_negative_weights_scale_on_magnitude():
    """Post-negation 'Liever niet met' weights are negative; the sign is irrelevant."""
    assert weight_scale([-3.0, 0.5]) == 2


def test_float_noise_within_the_contract_is_absorbed():
    """Weights are exact to 3 decimals: binary float noise must not inflate the scale."""
    assert weight_scale([0.1 + 0.2]) == weight_scale([0.3])


def test_no_weights_means_scale_one():
    """A problem without preferences needs no scaling at all."""
    assert weight_scale([]) == 1


def test_weight_rounding_to_zero_is_rejected():
    """A weight below the 3-decimal resolution would silently erase the wish."""
    with pytest.raises(ValueError):
        weight_scale([0.0004])
