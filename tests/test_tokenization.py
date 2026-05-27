from lognexus.tokenization import WhiteSpaceTokenizer


def test_whitespace_tokenizer_preserves_offsets():
    tokens = WhiteSpaceTokenizer().tokenize("GEO: Fly safe.")

    assert [(token.text, token.start, token.end) for token in tokens] == [
        ("GEO:", 0, 4),
        ("Fly", 5, 8),
        ("safe.", 9, 14),
    ]
