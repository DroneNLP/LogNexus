import re
from dataclasses import dataclass
from typing import List


PRETOKENIZERS = [
    "white-space",
    "spacy",
    "nltk-punct",
    "nltk-tweet",
    "nltk-treebank",
]


@dataclass
class Token:
    text: str
    start: int
    end: int


class Tokenizer:
    def tokenize(self, text: str) -> List[Token]:
        raise NotImplementedError


class WhiteSpaceTokenizer(Tokenizer):
    def tokenize(self, text: str) -> List[Token]:
        return [
            Token(text=match.group(), start=match.start(), end=match.end())
            for match in re.finditer(r"\S+", text)
        ]


class SpacyTokenizer(Tokenizer):
    def __init__(self):
        try:
            import spacy
        except ImportError as exc:
            raise ImportError(
                "spacy is required for the spacy pretokenizer. "
                "Install it with `python -m pip install spacy`."
            ) from exc

        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger"])
        except OSError:
            self.nlp = spacy.blank("en")

    def tokenize(self, text: str) -> List[Token]:
        return [
            Token(text=token.text, start=token.idx, end=token.idx + len(token.text))
            for token in self.nlp(text)
        ]


class NLTKPunctTokenizer(Tokenizer):
    def __init__(self):
        try:
            import nltk
        except ImportError as exc:
            raise ImportError(
                "nltk is required for the nltk-punct pretokenizer. "
                "Install it with `python -m pip install nltk`."
            ) from exc

        for resource in ("punkt", "punkt_tab"):
            try:
                nltk.data.find(f"tokenizers/{resource}")
            except (LookupError, OSError):
                nltk.download(resource, quiet=True)

    def tokenize(self, text: str) -> List[Token]:
        from nltk.tokenize import word_tokenize

        tokens = []
        current_pos = 0
        for token_text in word_tokenize(text):
            idx = text.find(token_text, current_pos)
            if idx == -1 and token_text in ("``", "''"):
                idx = text.find('"', current_pos)
                if idx == -1:
                    idx = text.find("'", current_pos)
            if idx == -1:
                idx = current_pos
            tokens.append(Token(text=token_text, start=idx, end=idx + len(token_text)))
            current_pos = idx + len(token_text)
        return tokens


class NLTKTweetTokenizer(Tokenizer):
    def __init__(self):
        try:
            from nltk.tokenize import TweetTokenizer
        except ImportError as exc:
            raise ImportError(
                "nltk is required for the nltk-tweet pretokenizer. "
                "Install it with `python -m pip install nltk`."
            ) from exc
        self.tokenizer = TweetTokenizer(
            preserve_case=True,
            reduce_len=False,
            strip_handles=False,
        )

    def tokenize(self, text: str) -> List[Token]:
        tokens = []
        current_pos = 0
        for token_text in self.tokenizer.tokenize(text):
            idx = text.find(token_text, current_pos)
            if idx == -1:
                idx = current_pos
            tokens.append(Token(text=token_text, start=idx, end=idx + len(token_text)))
            current_pos = idx + len(token_text)
        return tokens


class NLTKTreebankTokenizer(Tokenizer):
    def __init__(self):
        try:
            from nltk.tokenize import TreebankWordTokenizer
        except ImportError as exc:
            raise ImportError(
                "nltk is required for the nltk-treebank pretokenizer. "
                "Install it with `python -m pip install nltk`."
            ) from exc
        self.tokenizer = TreebankWordTokenizer()

    def tokenize(self, text: str) -> List[Token]:
        return [
            Token(text=text[start:end], start=start, end=end)
            for start, end in self.tokenizer.span_tokenize(text)
        ]


def get_tokenizer(name: str) -> Tokenizer:
    tokenizers = {
        "white-space": WhiteSpaceTokenizer,
        "spacy": SpacyTokenizer,
        "nltk-punct": NLTKPunctTokenizer,
        "nltk-tweet": NLTKTweetTokenizer,
        "nltk-treebank": NLTKTreebankTokenizer,
    }
    if name not in tokenizers:
        raise ValueError(f"Unknown pretokenizer: {name}. Available: {list(tokenizers)}")
    return tokenizers[name]()
