from lognexus.utils import pretokenize_log_message


def test_pretokenize_adds_missing_terminal_punctuation():
    assert pretokenize_log_message("Battery Error") == ["Battery", "Error", "."]


def test_pretokenize_splits_punctuation_but_preserves_decimal():
    assert pretokenize_log_message("Altitude 12.5m; land now!") == [
        "Altitude",
        "12.5m",
        ";",
        "land",
        "now",
        "!",
    ]


def test_pretokenize_preserves_apostrophes_and_removes_wrapping_quotes():
    assert pretokenize_log_message('"Drone\'s battery low"') == [
        "Drone's",
        "battery",
        "low",
        ".",
    ]


def test_pretokenize_empty_message_returns_empty_list():
    assert pretokenize_log_message("   ") == []
