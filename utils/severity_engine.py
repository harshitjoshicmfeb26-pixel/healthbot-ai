"""
utils/severity_engine.py
─────────────────────────
Severity-aware triage using the *official* DDXPlus pathology severity
rating, as a second, data-driven safety signal alongside the existing
keyword-based `utils.red_flag_rules`.

Why this exists
────────────────
`release_conditions.json` ships a "severity" field per pathology (DDXPlus
rates this on a 1-5 scale). Before this module, that field was loaded by
`ddxplus_decoder.condition_metadata()` but never read for its value —
the project's only safety layer was a hand-typed English/Hinglish/Marathi
keyword list. This module closes that gap by computing a severity-weighted
score over the model's *current* top-k differential, mirroring the idea
behind the DSHM ("rule-in / rule-out of severe pathologies") metric used in
the DDXPlus research literature.

IMPORTANT — verify the severity polarity before trusting this in any
real scenario
────────────────────────────────────────────────────────────────────────
DDXPlus's public documentation states each pathology has a 1-5 severity
rating, but does not make the polarity (does 1 or 5 mean "most severe")
obvious from the field name alone, and this project's bundled dataset copy
was not included with the code under review, so it cannot be empirically
confirmed here. **Call `describe_severity_scale()` once you have your real
`data/release_conditions.json` in place** — it prints the severity values
for a handful of conditions that are unambiguously high-acuity (Anaphylaxis,
Possible NSTEMI / STEMI, Pulmonary embolism, Spontaneous pneumothorax)
next to ones that are unambiguously low-acuity (Allergic sinusitis, Acute
otitis media, URTI), so you can visually confirm which end of the scale is
"severe" in your copy and set `SEVERE_END` below accordingly.

This module is advisory and additive. It never overrides or replaces the
keyword-based red-flag rules in `utils.red_flag_rules` — both run, and the
caller (see `chatbot/bot.py`) shows whichever signal fired.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from utils.ddxplus_decoder import condition_severity

# Which end of the official 1-5 scale represents the MORE severe pathology.
# "low" means severity 1 is most severe (1 = worst, 5 = mildest).
# "high" means severity 5 is most severe (5 = worst, 1 = mildest).
# Default is "low" because it matches the one citably-confirmed data point
# available at the time this module was written (Myasthenia gravis = 3, a
# mid-scale chronic neuromuscular disease, which is consistent with a scale
# where 1 and 2 are reserved for acute, high-mortality presentations).
# CONFIRM THIS against your own data before relying on it — see module
# docstring and `describe_severity_scale()`.
SEVERE_END: Literal["low", "high"] = "low"

# A pathology counts as "rule-in severe" once its severity crosses this
# point on the official scale (inclusive), in the direction set by
# SEVERE_END. With SEVERE_END="low" and SEVERITY_CUTOFF=2, severities 1-2
# are treated as high-acuity.
SEVERITY_CUTOFF = 2

_CALIBRATION_SEVERE_EXAMPLES = (
    "Anaphylaxis",
    "Possible NSTEMI / STEMI",
    "Pulmonary embolism",
    "Spontaneous pneumothorax",
)
_CALIBRATION_MILD_EXAMPLES = (
    "Allergic sinusitis",
    "Acute otitis media",
    "URTI",
    "Viral pharyngitis",
)


@dataclass
class SeverityFinding:
    pathology: str
    display_name: str
    confidence: float
    severity: int | None
    is_high_acuity: bool


@dataclass
class SeverityTriageResult:
    findings: list[SeverityFinding]
    weighted_severity_score: float | None  # lower-is-worse on a 0-1 scale if SEVERE_END="low"-normalized
    any_high_acuity_candidate: bool
    message: str


def _is_high_acuity(severity: int | None) -> bool:
    if severity is None:
        return False
    if SEVERE_END == "low":
        return severity <= SEVERITY_CUTOFF
    return severity >= (6 - SEVERITY_CUTOFF)


def _normalized_severity(severity: int) -> float:
    """Map the official 1-5 scale to a 0-1 scale where 1.0 = most severe."""
    if SEVERE_END == "low":
        return (6 - severity) / 5.0
    return severity / 5.0


def evaluate_differential_severity(predictions: list[dict]) -> SeverityTriageResult:
    """
    Compute a severity-weighted safety read on the model's current top-k
    differential.

    `predictions` is the same list of dicts produced by
    `model.predictor.predict_case` / `predict` — each item is expected to
    have `pathology` (or `disease`) and `confidence`.
    """
    findings: list[SeverityFinding] = []
    weighted_sum = 0.0
    weight_total = 0.0

    for item in predictions or []:
        pathology = str(item.get("pathology") or item.get("disease") or "")
        if not pathology:
            continue
        confidence = float(item.get("confidence", 0.0))
        severity = condition_severity(pathology)
        is_high = _is_high_acuity(severity)
        findings.append(SeverityFinding(
            pathology=pathology,
            display_name=str(item.get("disease", pathology)),
            confidence=confidence,
            severity=severity,
            is_high_acuity=is_high,
        ))
        if severity is not None:
            weighted_sum += confidence * _normalized_severity(severity)
            weight_total += confidence

    weighted_score = (weighted_sum / weight_total) if weight_total > 0 else None
    any_high = any(f.is_high_acuity for f in findings)

    if not any(f.severity is not None for f in findings):
        message = (
            "Severity engine has no usable severity data for the current "
            "candidates (release_conditions.json may be the bundled demo "
            "subset rather than the official file)."
        )
    elif any_high:
        high_names = ", ".join(f.display_name for f in findings if f.is_high_acuity)
        message = (
            f"At least one high-acuity candidate is present in the current differential "
            f"({high_names}), based on the official DDXPlus severity rating, regardless of "
            f"its probability rank. Treat this as a prompt to seek care promptly, not as a diagnosis."
        )
    else:
        message = "No high-acuity pathology, by official severity rating, is in the current differential."

    return SeverityTriageResult(
        findings=findings,
        weighted_severity_score=weighted_score,
        any_high_acuity_candidate=any_high,
        message=message,
    )


def describe_severity_scale() -> str:
    """
    Print/return the official severity values for a handful of obviously
    high- and low-acuity conditions, so you can visually confirm which end
    of the scale means "more severe" in your own release_conditions.json.

    Run this once after installing the real DDXPlus metadata, before
    trusting SEVERE_END's default value.
    """
    lines = ["Severity values from your current release_conditions.json:", ""]
    lines.append("Expected HIGH acuity (severe):")
    for name in _CALIBRATION_SEVERE_EXAMPLES:
        lines.append(f"  {name}: severity={condition_severity(name)}")
    lines.append("")
    lines.append("Expected LOW acuity (mild):")
    for name in _CALIBRATION_MILD_EXAMPLES:
        lines.append(f"  {name}: severity={condition_severity(name)}")
    lines.append("")
    lines.append(
        "If the 'expected HIGH acuity' group consistently shows LOWER numbers "
        "than the 'expected LOW acuity' group, SEVERE_END='low' (the current "
        "default) is correct. If it's the other way around, set SEVERE_END='high' "
        "at the top of utils/severity_engine.py."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe_severity_scale())
