"""Browser acceptance checks for the homepage gallery and responsive layout."""

import pytest
from playwright.sync_api import expect


@pytest.mark.parametrize("width", [1280, 390])
def test_home_gallery_is_keyboard_and_touch_operable(live_server, page, width):
    """The three homepage examples can be changed without hiding content offscreen."""
    page.set_viewport_size({"width": width, "height": 720})
    page.goto(live_server)

    gallery = page.locator("[data-home-gallery]")
    status = gallery.locator("[data-home-gallery-status]")
    slides = gallery.locator("[data-home-slide]")

    expect(page.locator(".home-cta a")).to_have_attribute("href", "/processes")
    expect(gallery).to_be_visible()
    if width == 1280:
        assert page.locator(".container").bounding_box()["width"] <= 960
    expect(status).to_have_text("1 van 3")
    expect(slides.nth(0)).to_be_visible()
    expect(slides.nth(1)).to_be_hidden()
    expect(slides.nth(2)).to_be_hidden()

    gallery.locator("[data-home-gallery-next]").click()
    expect(status).to_have_text("2 van 3")
    expect(slides.nth(1)).to_be_visible()

    gallery.focus()
    page.keyboard.press("ArrowRight")
    expect(status).to_have_text("3 van 3")
    page.keyboard.press("ArrowLeft")
    expect(status).to_have_text("2 van 3")

    viewport = gallery.locator(".home-gallery-viewport")
    viewport.scroll_into_view_if_needed()
    box = viewport.bounding_box()
    assert box is not None
    start_x = box["x"] + box["width"] * 0.75
    end_x = box["x"] + box["width"] * 0.25
    y = box["y"] + box["height"] / 2
    page.mouse.move(start_x, y)
    page.mouse.down()
    page.mouse.move(end_x, y)
    page.mouse.up()
    expect(status).to_have_text("3 van 3")

    overflow = page.evaluate(
        """() => ({width: innerWidth, scroll: document.documentElement.scrollWidth})"""
    )
    assert overflow["scroll"] <= overflow["width"], overflow


@pytest.mark.usefixtures("login")
def test_home_renders_for_logged_in_school(live_server, page):
    """The school navigation does not hide the homepage CTA or gallery."""
    page.goto(live_server)
    expect(page.locator("h1")).to_have_text(
        "De groepsindeling: een complexe puzzel, in enkele minuten opgelost"
    )
    expect(page.locator(".home-cta a")).to_be_visible()
    expect(page.locator("[data-home-gallery]")).to_be_visible()
