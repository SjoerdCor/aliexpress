"""Shared expectations for the three browser-visible wizard routes."""

WIZARD_STEPS_BY_MODE = {
    "forward": (
        "Schoolinformatie",
        "Leerlingen controleren",
        "Groepen controleren",
        "Voorkeuren invullen",
        "Leerlingen spreiden",
        "Groepsindeling berekenen",
        "Resultaat bekijken",
        "Klaar!",
    ),
    "redistribute": (
        "Schoolinformatie",
        "Groepen kiezen",
        "Leerlingen controleren",
        "Voorkeuren invullen",
        "Leerlingen spreiden",
        "Groepsindeling berekenen",
        "Resultaat bekijken",
        "Klaar!",
    ),
    "redistribute_and_forward": (
        "Schoolinformatie",
        "Leerlingen controleren",
        "Groepen controleren",
        "Voorkeuren invullen",
        "Leerlingen spreiden",
        "Groepsindeling berekenen",
        "Resultaat bekijken",
        "Klaar!",
    ),
}


def assert_wizard_page(page, mode, active_label):
    """Check the route steps, heading, and central navigation labels on one page."""

    labels = WIZARD_STEPS_BY_MODE[mode]
    assert page.locator(".step").all_inner_texts() == list(labels)
    assert page.locator(".step.active").count() == 1
    assert page.locator(".step.active").inner_text().strip() == active_label
    assert page.locator("h1").inner_text().strip() == active_label

    current_index = labels.index(active_label)
    previous = page.locator("a.previous-step")
    if current_index:
        assert previous.count() == 1
        assert previous.inner_text().strip() == (
            f"← Terug naar {labels[current_index - 1]}"
        )
    else:
        assert previous.count() == 0

    if active_label == "Groepsindeling berekenen":
        # Starting the calculation is an action on this page, not navigation to the
        # result page. Its label still comes from the central current step name.
        assert all(
            active_label in button_text
            for button_text in page.locator(".next-step").all_inner_texts()
        )
    elif current_index + 1 < len(labels):
        next_label = labels[current_index + 1]
        next_buttons = page.locator(".next-step")
        assert next_buttons.count() >= 1
        assert all(
            next_label in button_text for button_text in next_buttons.all_inner_texts()
        )
