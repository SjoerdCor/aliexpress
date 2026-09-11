"""The user-facing steps shared by the wizard routes and templates.

The three distribution modes do not have the same route. Keeping their steps here makes
the progress strip and the navigation buttons describe the same journey.
"""

from dataclasses import dataclass

from flask import url_for


@dataclass(frozen=True)
class WizardStep:
    """One visible wizard step and the endpoint that renders it."""

    key: str
    label: str
    endpoint: str


_UPLOAD_EDEXML = WizardStep("upload_edexml", "Schoolinformatie", "wizard.upload_edexml")
_SELECT_GROUPS = WizardStep("select_groups", "Groepen kiezen", "wizard.select_groups")
_ROSTER = WizardStep("roster", "Leerlingen controleren", "roster.roster_page")
_GROUPS_TO = WizardStep("groups_to", "Groepen controleren", "wizard.groups_to_page")
_GROUPS_TO_SELECT = WizardStep(
    "select_groups", "Groepen controleren", "wizard.select_groups"
)
_PREFERENCES_FORM = WizardStep(
    "preferences_form", "Voorkeuren invullen", "wizard.preferences_form"
)
_NOT_TOGETHER = WizardStep(
    "not_together", "Leerlingen spreiden", "wizard.not_together_page"
)
_PROCESSING = WizardStep("processing", "Groepsindeling berekenen", "results.processing")
_RESULT = WizardStep("result", "Resultaat bekijken", "results.result_page")
_DONE = WizardStep("done", "Klaar!", "results.done")


_STEPS_BY_MODE = {
    # Doorzetten: the groups in the new arrangement are checked after the population.
    "forward": (
        _UPLOAD_EDEXML,
        _ROSTER,
        _GROUPS_TO,
        _PREFERENCES_FORM,
        _NOT_TOGETHER,
        _PROCESSING,
        _RESULT,
        _DONE,
    ),
    # Herindelen: first choose the groups to redistribute, then check the participating
    # learners. The groups-to page is skipped by this mode.
    "redistribute": (
        _UPLOAD_EDEXML,
        _SELECT_GROUPS,
        _ROSTER,
        _PREFERENCES_FORM,
        _NOT_TOGETHER,
        _PROCESSING,
        _RESULT,
        _DONE,
    ),
    # Herindelen met doorzetten: groups are chosen after the roster, on the
    # select_groups endpoint. It is the same conceptual step as Groepen controleren.
    "redistribute_and_forward": (
        _UPLOAD_EDEXML,
        _ROSTER,
        _GROUPS_TO_SELECT,
        _PREFERENCES_FORM,
        _NOT_TOGETHER,
        _PROCESSING,
        _RESULT,
        _DONE,
    ),
}


def steps_for_mode(mode: str) -> tuple[WizardStep, ...]:
    """Return the actual visible route for ``mode``."""

    return _STEPS_BY_MODE.get(mode, _STEPS_BY_MODE["forward"])


def wizard_context(mode: str, current_key: str, *, previous_endpoint=None) -> dict:
    """Build progress and navigation context for one wizard page.

    ``previous_endpoint`` is used by the two preference input variants: both are the
    same conceptual step, but the back link must return to the selected input page.
    """

    steps = steps_for_mode(mode)
    current_index = next(
        index for index, step in enumerate(steps) if step.key == current_key
    )
    current = steps[current_index]
    previous = steps[current_index - 1] if current_index else None
    following = steps[current_index + 1] if current_index + 1 < len(steps) else None

    return {
        "wizard_steps": steps,
        "current_step": current_index + 1,
        "current_step_label": current.label,
        "previous_step": previous,
        "next_step": following,
        "prev_url": (
            url_for(previous_endpoint or previous.endpoint) if previous else None
        ),
        "next_url": url_for(following.endpoint) if following else None,
        "prev_label": f"← Terug naar {previous.label}" if previous else None,
        "next_label": f"Verder naar {following.label} →" if following else None,
    }
