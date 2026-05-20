from __future__ import annotations

import re
from typing import Iterable


def sanitize_video_prompt(
    prompt: str,
    character_identifiers: Iterable[str],
) -> str:
    """Remove character names from prompts sent to video generators.

    Image generation can still use named portrait references. This keeps video
    prompts focused on roles so person-safety filters are less likely to treat a
    cameo as a request for a prominent named person.
    """
    if not prompt:
        return prompt

    term_map = _build_term_replacements(character_identifiers)
    sanitized = prompt
    for term, replacement in sorted(term_map.items(), key=lambda item: len(item[0]), reverse=True):
        sanitized = _replace_term(sanitized, term, replacement)
    return sanitized


def _build_term_replacements(character_identifiers: Iterable[str]) -> dict[str, str]:
    term_to_replacements: dict[str, set[str]] = {}
    for index, identifier in enumerate(character_identifiers or []):
        if _is_non_person_or_role_identifier(identifier):
            continue

        replacement = "the presenter" if index == 0 else f"the character {index + 1}"
        for term in _identifier_terms(identifier):
            term_to_replacements.setdefault(term.casefold(), set()).add(replacement)

    return {
        term: next(iter(replacements))
        for term, replacements in term_to_replacements.items()
        if len(replacements) == 1
    }


def _identifier_terms(identifier: str) -> list[str]:
    clean = re.sub(r"[<>]", " ", identifier or "")
    clean = re.sub(r"[_-]+", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    if not clean:
        return []

    terms = [clean]
    name_tokens = re.findall(r"[A-Za-z][A-Za-z'’-]*", clean)
    if len(name_tokens) == 1 and clean == name_tokens[0] and clean.isupper():
        return []

    if len(name_tokens) >= 2:
        terms.extend(name_tokens)
    return list(dict.fromkeys(terms))


def _is_non_person_or_role_identifier(identifier: str) -> bool:
    clean = re.sub(r"\s+", " ", identifier or "").strip().casefold()
    if not clean:
        return True

    non_person_keywords = {
        "ai",
        "app",
        "assistant",
        "avatar",
        "bot",
        "brand",
        "company",
        "girlfriend",
        "logo",
        "mascot",
        "narrator",
        "product",
        "voice",
        "vo",
    }
    tokens = set(re.findall(r"[a-z]+", clean))
    return bool(tokens & non_person_keywords)


def _replace_term(text: str, term: str, replacement: str) -> str:
    term_body = r"\s+".join(re.escape(part) for part in term.split())
    patterns = [
        re.compile(rf"<\s*{term_body}\s*>(?P<poss>['’]s)?", flags=re.IGNORECASE),
        re.compile(rf"(?<![A-Za-z0-9_]){term_body}(?P<poss>['’]s)?(?![A-Za-z0-9_])", flags=re.IGNORECASE),
    ]

    def repl(match: re.Match[str]) -> str:
        if match.groupdict().get("poss"):
            return f"{replacement}'s"
        return replacement

    for pattern in patterns:
        text = pattern.sub(repl, text)
    return text
