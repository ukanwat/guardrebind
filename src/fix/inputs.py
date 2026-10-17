"""Input builders for the fix: the stock format and the bounded format.

Stock is what the model card does: the chat template renders the whole user
message as one string and the tokenizer parses it, special tokens included, so
literal "[/INST]" in the document becomes the real control token.

Bounded is the boundary half of the fix. The content is tokenized on its own
with special-token parsing off, so nothing in it can become a control token, and
it is wrapped in two reserved ids the tokenizer can never produce from text.
Everything outside the content is rendered by the stock template, unchanged.
"""

from __future__ import annotations

from ..prompt import build_messages

# Unused reserved ids in the Tekken vocab (<SPECIAL_36>, <SPECIAL_37>). Their
# text form does not tokenize back to these ids, so content cannot forge them.
DOC_OPEN_ID = 36
DOC_CLOSE_ID = 37

_PLACEHOLDER = "DOC"


def _render(tokenizer, instruct: str, query: str) -> tuple[str, str]:
    text = tokenizer.apply_chat_template(
        build_messages(instruct, query, _PLACEHOLDER),
        add_generation_prompt=True,
        tokenize=False,
    )
    prefix, suffix = text.split(_PLACEHOLDER)
    return prefix, suffix


def _content(prefix: str, document: str) -> tuple[str, str]:
    """The template puts a space before the document and the tokenizer merges
    it into the document's first token. Move it across so plain documents
    tokenize exactly as in the stock path."""
    stripped = prefix.rstrip(" ")
    return stripped, prefix[len(stripped):] + document


def _ids(tokenizer, text: str, parse_special: bool) -> list[int]:
    return tokenizer(
        text, add_special_tokens=False, split_special_tokens=not parse_special
    )["input_ids"]


def stock_ids(tokenizer, instruct: str, query: str, document: str) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        build_messages(instruct, query, document), add_generation_prompt=True
    )
    return list(encoded["input_ids"] if hasattr(encoded, "keys") else encoded)


def bounded_ids(tokenizer, instruct: str, query: str, document: str) -> list[int]:
    prefix, suffix = _render(tokenizer, instruct, query)
    prefix, document = _content(prefix, document)
    return (
        _ids(tokenizer, prefix, parse_special=True)
        + [DOC_OPEN_ID]
        + _ids(tokenizer, document, parse_special=False)
        + [DOC_CLOSE_ID]
        + _ids(tokenizer, suffix, parse_special=True)
    )


def split_ids(tokenizer, instruct: str, query: str, document: str) -> list[int]:
    """Stock layout, but content tokenized as plain text. Isolates the
    control-token channel from the boundary tokens."""
    prefix, suffix = _render(tokenizer, instruct, query)
    prefix, document = _content(prefix, document)
    return (
        _ids(tokenizer, prefix, parse_special=True)
        + _ids(tokenizer, document, parse_special=False)
        + _ids(tokenizer, suffix, parse_special=True)
    )
