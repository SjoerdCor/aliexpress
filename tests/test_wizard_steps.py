"""Tests for the route-dependent wizard step description."""

from aliexpress.web.wizard_steps import steps_for_mode
from tests.browser.wizard_route_helpers import WIZARD_STEPS_BY_MODE


def test_forward_route_contains_only_the_steps_it_visits():
    """Doorzetten skips group selection for existing groups."""
    assert [step.label for step in steps_for_mode("forward")] == list(
        WIZARD_STEPS_BY_MODE["forward"]
    )


def test_redistribute_route_puts_existing_group_choice_before_learners():
    """Herindelen shows its group-choice step and skips groups-to."""
    assert [step.label for step in steps_for_mode("redistribute")] == list(
        WIZARD_STEPS_BY_MODE["redistribute"]
    )


def test_redistribute_and_forward_checks_destinations_after_learners():
    """The combined mode uses select_groups for its destination-group step."""
    steps = steps_for_mode("redistribute_and_forward")
    assert [step.label for step in steps] == list(
        WIZARD_STEPS_BY_MODE["redistribute_and_forward"]
    )
    assert steps[2].endpoint == "wizard.select_groups"


def test_step_keys_follow_the_rendering_route_names():
    """Step keys stay aligned with the route/template stems used to render them."""
    for mode in WIZARD_STEPS_BY_MODE:
        for step in steps_for_mode(mode):
            route_name = step.endpoint.rsplit(".", maxsplit=1)[1]
            assert step.key == route_name.removesuffix("_page")
