from great_tables import GT, exibble, md, html, style, loc
from great_tables._utils_render_html import create_source_notes_component_h, create_body_component_h

small_exibble = exibble[["num", "char"]].head(3)


def assert_rendered_source_notes(snapshot, gt):
    built = gt._build_data("html")
    source_notes = create_source_notes_component_h(built)

    assert snapshot == source_notes


def assert_rendered_body(snapshot, gt):
    built = gt._build_data("html")
    body = create_body_component_h(built)

    assert snapshot == body


def test_source_notes_snap(snapshot):
    new_gt = (
        GT(exibble)
        .tab_source_note(md("An **important** note."))
        .tab_source_note(md("Another *important* note."))
        .tab_source_note("A plain note.")
        .tab_source_note(html("An <strong>HTML heavy</strong> note."))
    )

    assert_rendered_source_notes(snapshot, new_gt)


def test_styling_data_1(snapshot):
    new_gt = GT(small_exibble).tab_style(
        style=style.text(color="red"),
        locations=loc.body(),
    )

    assert_rendered_body(snapshot, new_gt)


def test_styling_data_2(snapshot):
    new_gt = GT(small_exibble).tab_style(
        style=style.text(color="red"),
        locations=loc.body(columns=["char"]),
    )

    assert_rendered_body(snapshot, new_gt)


def test_styling_data_3(snapshot):
    new_gt = GT(small_exibble).tab_style(
        style=style.text(color="red"),
        locations=loc.body(columns="char", rows=[0, 2]),
    )

    assert_rendered_body(snapshot, new_gt)


def test_styling_data_4(snapshot):
    new_gt = GT(small_exibble).tab_style(
        style=style.text(color="red"),
        locations=loc.body(columns=[], rows=[0, 2]),
    )

    assert_rendered_body(snapshot, new_gt)


def test_styling_data_5(snapshot):
    new_gt = GT(small_exibble).tab_style(
        style=style.text(color="red"),
        locations=loc.body(columns="char", rows=[]),
    )

    assert_rendered_body(snapshot, new_gt)


def test_styling_data_6(snapshot):
    new_gt = GT(small_exibble).tab_style(
        style=style.text(color="red"),
        locations=loc.body(columns=[], rows=[]),
    )

    assert_rendered_body(snapshot, new_gt)
