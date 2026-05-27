from lognexus.decode import decode_bioes, flatten_group_segments, reconstruct_segments_group_aware


def test_decode_bioes_handles_single_and_multi_token_segments():
    tags = ["S-SEG", "O", "B-SEG", "E-SEG"]

    assert decode_bioes(tags) == [(0, 0), (2, 3)]


def test_reconstruct_segments_group_aware_splits_on_outside_tokens():
    groups = reconstruct_segments_group_aware(
        raw_message="Failsafe RTH; RC signal lost.",
        conll_tokens=["Failsafe", "RTH", ";", "RC", "signal", "lost", "."],
        tags=["B-SEG", "E-SEG", "O", "B-SEG", "I-SEG", "E-SEG", "O"],
        pretokenizer_name="white-space",
        tag_scheme="bioes",
    )

    assert flatten_group_segments(groups) == ["Failsafe RTH", "RC signal lost"]
    assert groups[0]["delimiter"] == ";"
