"""Check that every number in the agent's answer came from a tool.

The LLM is told to copy numbers from tool results only. This module verifies
that afterwards: it pulls every number out of the answer text and looks for a
matching value (allowing for rounding) in the tool outputs, the dataset info,
the limits, or the user's question. Anything unmatched is flagged in the UI
and in the report so an engineer knows not to trust it.
"""

from __future__ import annotations

import re

import numpy as np

# A number not glued to a letter/underscore/digit on its left (so "p95" or "v2" are skipped).
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.])[-+−]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
LIST_MARKER_RE = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)


def collect_numbers(obj) -> list[float]:
    """Recursively gather every number inside nested dicts/lists/strings."""
    found: list[float] = []
    if isinstance(obj, bool) or obj is None:
        return found
    if isinstance(obj, (int, float)):
        found.append(float(obj))
    elif isinstance(obj, str):
        found.extend(float(m.replace("−", "-")) for m in NUMBER_RE.findall(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            found.extend(collect_numbers(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            found.extend(collect_numbers(v))
    return found


def find_unverified_numbers(text: str, reference) -> list[str]:
    """Return the number tokens in `text` that match nothing in `reference`.

    A token matches if, after rounding to the precision it was written with,
    it equals some reference value (sign ignored, so '1.4 V dip' matches -1.4).
    Values x1000 and /1000 are also accepted so 's' <-> 'ms' and 'kW' <-> 'W'
    conversions of real tool values are not flagged.
    """
    refs = np.abs(np.array(collect_numbers(reference), dtype=float))
    refs = np.concatenate([refs, refs * 1000.0, refs / 1000.0]) if refs.size else refs
    cleaned = LIST_MARKER_RE.sub("", text or "")
    unverified: list[str] = []
    for token in NUMBER_RE.findall(cleaned):
        value = abs(float(token.replace("−", "-")))
        mantissa = token.lower().split("e")[0]
        decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
        tolerance = 0.5 * 10 ** (-decimals) + 1e-9
        if refs.size == 0 or np.min(np.abs(refs - value)) > tolerance:
            if token not in unverified:
                unverified.append(token)
    return unverified
