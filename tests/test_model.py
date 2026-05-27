from lognexus.model import LogNexusModel


def reconstruct(token_predictions):
    model = LogNexusModel.__new__(LogNexusModel)
    return model._reconstruct_sentences(token_predictions)


def test_reconstructs_bioes_sentences():
    predictions = [
        {"RC": "B-SENT"},
        {"signal": "I-SENT"},
        {"lost": "E-SENT"},
        {".": "O"},
        {"Returning": "B-SENT"},
        {"home": "E-SENT"},
    ]

    assert reconstruct(predictions) == ["RC signal lost", "Returning home"]


def test_reconstructs_single_token_sentence():
    assert reconstruct([{"Landing": "S-SENT"}]) == ["Landing"]


def test_reconstructs_end_tag_at_final_token():
    predictions = [{"Battery": "B-SENT"}, {"low": "E-SENT"}]

    assert reconstruct(predictions) == ["Battery low"]


def test_ignores_outside_tokens_and_semicolon_separators():
    predictions = [
        {"Noise": "O"},
        {";": "O"},
        {"Failsafe": "S-SENT"},
    ]

    assert reconstruct(predictions) == ["Failsafe"]
