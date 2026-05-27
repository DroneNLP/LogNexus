import re
from typing import Dict, List, Tuple

from lognexus.tokenization import Token, get_tokenizer


TAG_SCHEMES = ["iob2", "bioes"]

_TOKENIZER_CACHE: Dict[str, object] = {}


def _get_cached_tokenizer(name: str):
    if name not in _TOKENIZER_CACHE:
        _TOKENIZER_CACHE[name] = get_tokenizer(name)
    return _TOKENIZER_CACHE[name]


def decode_iob2(tags: List[str]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    start = None
    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            if start is not None:
                spans.append((start, i - 1))
            start = i
        elif tag.startswith("I-"):
            if start is None:
                start = i
        else:
            if start is not None:
                spans.append((start, i - 1))
                start = None
    if start is not None:
        spans.append((start, len(tags) - 1))
    return spans


def decode_bioes(tags: List[str]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    start = None
    for i, tag in enumerate(tags):
        if tag.startswith("S-"):
            if start is not None:
                spans.append((start, i - 1))
            spans.append((i, i))
            start = None
        elif tag.startswith("B-"):
            if start is not None:
                spans.append((start, i - 1))
            start = i
        elif tag.startswith("I-"):
            if start is None:
                start = i
        elif tag.startswith("E-"):
            if start is not None:
                spans.append((start, i))
                start = None
            else:
                spans.append((i, i))
        else:
            if start is not None:
                spans.append((start, i - 1))
                start = None
    if start is not None:
        spans.append((start, len(tags) - 1))
    return spans


def decode_tags_to_spans(tags: List[str], scheme: str) -> List[Tuple[int, int]]:
    if scheme == "iob2":
        return decode_iob2(tags)
    if scheme == "bioes":
        return decode_bioes(tags)
    raise ValueError(f"Unknown tag scheme: {scheme}")


def detokenize(tokens: List[str]) -> str:
    if not tokens:
        return ""
    text = " ".join(tokens)
    text = re.sub(r"\s+([.,!?:;])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    return re.sub(r"\s+([\)\]\}])", r"\1", text)


def _split_into_groups(
    tokens: List[str],
    tags: List[str],
) -> List[Tuple[List[str], List[str], List[int], str]]:
    groups: List[Tuple[List[str], List[str], List[int], str]] = []
    cur_toks: List[str] = []
    cur_tags: List[str] = []
    cur_idxs: List[int] = []

    for i, (token, tag) in enumerate(zip(tokens, tags)):
        if tag == "O":
            if cur_toks:
                groups.append((cur_toks, cur_tags, cur_idxs, ""))
                cur_toks, cur_tags, cur_idxs = [], [], []
            if groups:
                g_toks, g_tags, g_idxs, delimiter = groups[-1]
                groups[-1] = (g_toks, g_tags, g_idxs, delimiter + token)
        else:
            cur_toks.append(token)
            cur_tags.append(tag)
            cur_idxs.append(i)

    if cur_toks:
        groups.append((cur_toks, cur_tags, cur_idxs, ""))
    if not groups:
        groups.append(([], [], [], ""))
    return groups


def reconstruct_segments_group_aware(
    raw_message: str,
    conll_tokens: List[str],
    tags: List[str],
    pretokenizer_name: str,
    tag_scheme: str,
) -> List[Dict]:
    tokenizer = _get_cached_tokenizer(pretokenizer_name)
    offset_tokens: List[Token] = tokenizer.tokenize(raw_message)
    use_offsets = len(offset_tokens) == len(conll_tokens)

    groups = _split_into_groups(conll_tokens, tags)
    result: List[Dict] = []

    for group_id, (group_tokens, group_tags, original_indices, delimiter) in enumerate(groups, start=1):
        span_indices = decode_tags_to_spans(group_tags, tag_scheme)
        segments: List[str] = []

        if use_offsets and original_indices:
            group_start = offset_tokens[original_indices[0]].start
            group_end = offset_tokens[original_indices[-1]].end
            group_text = raw_message[group_start:group_end]
        elif group_tokens:
            group_text = detokenize(group_tokens)
        else:
            group_text = ""

        for start, end in span_indices:
            if use_offsets and original_indices:
                original_start = original_indices[start]
                original_end = original_indices[end]
                char_start = offset_tokens[original_start].start
                char_end = offset_tokens[original_end].end
                segments.append(raw_message[char_start:char_end])
            else:
                segments.append(detokenize(group_tokens[start:end + 1]))

        result.append(
            {
                "group_id": group_id,
                "text": group_text,
                "segments": segments,
                "delimiter": delimiter,
            }
        )

    return result


def flatten_group_segments(groups: List[Dict]) -> List[str]:
    segments: List[str] = []
    for group in groups:
        segments.extend(group.get("segments", []))
    return segments
