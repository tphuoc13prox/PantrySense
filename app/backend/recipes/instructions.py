"""Cleaning and segmentation of raw recipe instructions into ordered steps."""
from __future__ import annotations

import json
import re


def clean_and_segment_instructions(raw_instructions: list[str] | str | None) -> list[str]:
    """Cleans, de-fragments, and normalizes cooking instructions into structured steps."""
    if not raw_instructions:
        return []

    items: list[str] = []
    if isinstance(raw_instructions, str):
        raw_str = raw_instructions.strip()
        if raw_str.startswith("[") and raw_str.endswith("]"):
            try:
                parsed = json.loads(raw_str)
                if isinstance(parsed, list):
                    items = [str(x) for x in parsed]
                else:
                    items = [raw_str]
            except Exception:
                items = [line.strip() for line in raw_str.split("\n") if line.strip()]
        elif raw_str.startswith("c(") and raw_str.endswith(")"):
            items = re.findall(r'"([^"]*)"', raw_str) or re.findall(r"'([^']*)'", raw_str)
        else:
            items = [line.strip() for line in raw_str.split("\n") if line.strip()]
    elif isinstance(raw_instructions, list):
        items = [str(x) for x in raw_instructions if x is not None]
    else:
        return []

    # Step 1: Merge broken sentence fragments (lines ending with backslash, comma, or starting with lowercase continuation)
    merged: list[str] = []
    for raw_item in items:
        text = raw_item.strip()
        if not text:
            continue
        # Remove standalone punctuation junk
        if text in (",", ".", ";", ":", '"', "'", "\\", "/", "-"):
            continue

        # Clean escaped quotes
        text = text.replace('\\"', '"').replace("\\'", "'")

        # Check if previous item had a trailing backslash or comma or was incomplete
        if merged and (
            merged[-1].endswith("\\")
            or merged[-1].endswith(",")
            or (len(text) > 0 and text[0].islower() and not merged[-1].endswith("."))
        ):
            prev = re.sub(r'[\s\\,]+$', '', merged[-1])
            # If previous ended with dimension like 1/8\ -> 1/8"
            if re.search(r'\d+/\d+$', prev):
                prev = prev + '"'
            merged[-1] = f"{prev} {text}".strip()
        else:
            merged.append(text)

    # Step 2: Split multi-step blocks (e.g. "1. Preheat... 2. Mix...") if present
    segmented: list[str] = []
    for item in merged:
        numbered_matches = list(re.finditer(r'(?:^|\s)(?:Step\s*)?(\d+)[.)]\s+', item, re.I))
        if len(numbered_matches) >= 2:
            for i in range(len(numbered_matches)):
                start = numbered_matches[i].end()
                end = numbered_matches[i + 1].start() if i + 1 < len(numbered_matches) else len(item)
                step_text = item[start:end].strip()
                if step_text:
                    segmented.append(step_text)
        else:
            sub_lines = [line.strip() for line in item.split("\n") if line.strip()]
            if len(sub_lines) > 1:
                segmented.extend(sub_lines)
            else:
                segmented.append(item)

    # Step 3: Final sanitization
    final_steps: list[str] = []
    for s in segmented:
        s = re.sub(r'\\+', '', s)
        s = re.sub(r'[\s,;]+$', '', s)
        s = re.sub(r'\s+', ' ', s).strip()
        s = re.sub(r'^(?:Step\s*\d+[:.]?\s*|\d+[.)]\s*)', '', s, flags=re.I).strip()

        if len(s) >= 4 and re.search(r'[a-zA-Z0-9]', s):
            s = s[0].upper() + s[1:]
            if not s.endswith(('.', '!', '?')):
                s += '.'
            final_steps.append(s)

    return final_steps
