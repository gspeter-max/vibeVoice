import src.ui.hud as hud


def test_menu_bar_waveform_layout_centers_bars():
    layout = hud.compute_menu_bar_waveform_layout(
        status_width=52,
        status_height=22,
        num_bars=5,
        bar_width=2.0,
        bar_gap=3.0,
        bar_height=12.0,
    )

    assert len(layout) == 5

    xs = [item["x"] for item in layout]
    ys = [item["y"] for item in layout]
    heights = [item["height"] for item in layout]

    assert xs == sorted(xs)
    assert abs(((xs[0] + layout[0]["width"] + xs[-1]) / 2) - (52 / 2)) < 0.6
    assert all(abs(y - ((22 - 12.0) / 2)) < 0.6 for y in ys)
    assert all(height == 12.0 for height in heights)


def test_menu_bar_waveform_layout_fits_within_status_item():
    layout = hud.compute_menu_bar_waveform_layout(
        status_width=64,
        status_height=24,
        num_bars=7,
        bar_width=2.0,
        bar_gap=2.0,
        bar_height=14.0,
    )

    assert layout[0]["x"] >= 0
    assert layout[-1]["x"] + layout[-1]["width"] <= 64
    assert all(item["y"] >= 0 for item in layout)
    assert all(item["y"] + item["height"] <= 24 for item in layout)


def test_menu_bar_idle_alpha_is_visible():
    hud_widget = hud.MenuBarWaveformController.__new__(hud.MenuBarWaveformController)
    hud_widget._state = hud.HIDDEN
    hud_widget._voice_smooth = 0.0

    assert hud_widget._status_bar_alpha() >= 0.74


def test_menu_bar_bar_height_is_taller():
    layout = hud.compute_menu_bar_waveform_layout(
        status_width=hud.STATUS_ITEM_W,
        status_height=hud.STATUS_ITEM_H,
        num_bars=hud.NUM_BARS,
        bar_width=hud.BAR_W,
        bar_gap=hud.BAR_GAP,
        bar_height=hud.BAR_MAX_H,
    )

    assert all(item["height"] == hud.BAR_MAX_H for item in layout)
    assert hud.BAR_MAX_H > 12.0


def test_menu_bar_vertical_shift_is_centered():
    assert hud.STATUS_ITEM_VERTICAL_SHIFT == 0.0


def test_hud_ignores_transcript_payloads_and_keeps_status_only():
    hud_widget = hud.MenuBarWaveformController.__new__(hud.MenuBarWaveformController)
    hud_widget._state = hud.HIDDEN
    hud_widget._request_menu_bar_view_redraw = lambda: None
    hud_widget.show_listening = lambda: None
    hud_widget.show_done = lambda: None
    hud_widget.hide_hud = lambda: None
    hud_widget.show_thinking = lambda: None
    hud_widget.show_processing = lambda: None

    hud.MenuBarWaveformController._on_command(hud_widget, "draft:Hello World")
    hud.MenuBarWaveformController._on_command(hud_widget, "final:Hello World")

    assert hud_widget._state == hud.HIDDEN
    assert not hasattr(hud_widget, "_draft_text")
    assert not hasattr(hud_widget, "_final_text")
