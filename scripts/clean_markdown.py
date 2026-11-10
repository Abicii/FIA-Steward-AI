"""Strip repeating headers, footers, and page numbers from parsed markdown."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

ALWAYS_DROP = [
    re.compile(r"^©"),
    re.compile(r"^issue\s+\d+", re.I),
    re.compile(r"^\d{1,2}\s+[A-Za-z]+\s+\d{4}$"),
    re.compile(r"^page\s+\d+(?:\s*(?:/|of)\s*\d+)?$", re.I),
    re.compile(r"^-\s*\d+\s*-$"),
    re.compile(r"^\d+\s*/\s*\d+$"),
    re.compile(r"^[A-Z]\d{1,3}$"),
    re.compile(r"^#{1,6}\s*[A-Z]\s*$"),
    re.compile(r"^[-*_]+$"),
]

HEADING_PREFIX = re.compile(r"^#{1,6}\s*")
EMPHASIS = re.compile(r"[*_]+")
RUNNING_LINE_MAX_LEN = 120
RUNNING_LINE_MIN_COUNT = 3
PROLOGUE_LINES = 20


def _normalize(line: str) -> str:
    text = line.strip()
    text = HEADING_PREFIX.sub("", text)
    text = EMPHASIS.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_always_drop(normalized: str) -> bool:
    if not normalized:
        return False
    return any(pattern.match(normalized) for pattern in ALWAYS_DROP)


def _running_headers(lines: list[str]) -> set[str]:
    counts = Counter()
    for line in lines:
        key = _normalize(line)
        if not key or len(key) > RUNNING_LINE_MAX_LEN:
            continue
        if _is_always_drop(key):
            continue
        counts[key] += 1
    return {key for key, count in counts.items() if count >= RUNNING_LINE_MIN_COUNT}


def _in_prologue(running: set[str], lines: list[str]) -> set[str]:
    prologue = {_normalize(line) for line in lines[:PROLOGUE_LINES]}
    return {key for key in running if key in prologue}


def clean_markdown(text: str) -> str:
    lines = text.splitlines()
    running = _running_headers(lines)
    keep_once = _in_prologue(running, lines)
    seen: set[str] = set()
    cleaned: list[str] = []

    for line in lines:
        key = _normalize(line)
        if not key:
            cleaned.append("")
            continue
        if _is_always_drop(key):
            continue
        if key in running:
            if key in keep_once and key not in seen:
                seen.add(key)
                cleaned.append(line.rstrip())
            continue
        cleaned.append(line.rstrip())

    collapsed: list[str] = []
    blank_run = 0
    for line in cleaned:
        if line.strip():
            collapsed.append(line)
            blank_run = 0
            continue
        blank_run += 1
        if blank_run <= 2:
            collapsed.append("")

    return "\n".join(collapsed).strip() + "\n"


def clean_file(path: Path) -> None:
    original = path.read_text(encoding="utf-8")
    path.write_text(clean_markdown(original), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Strip headers, footers, and page numbers from markdown.")
    parser.add_argument(
        "path",
        nargs="?",
        default="parsed data",
        help="Markdown file or folder (default: parsed data)",
    )
    args = parser.parse_args()
    target = Path(args.path)

    if target.is_file():
        clean_file(target)
        return

    for markdown_file in sorted(target.glob("*.md")):
        clean_file(markdown_file)


if __name__ == "__main__":
    main()
