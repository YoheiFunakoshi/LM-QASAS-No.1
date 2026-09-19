"""Select a requested number of distinct CDR-H3 amino-acid sequences.

This module consumes already scored Peak clone records. It does not calculate
LM-QASAS scores, run AbLang2, or decide which biological candidates qualify.
Each record requires ``cdr3``, ``score``, ``source_row``, and ``timepoint``;
``timepoint`` must be exactly ``"Peak"``. Callers must explicitly map their
input schema and timepoint labels before selection. All other fields, including
optional nested ``metadata``, are retained in the selected sequence's records.

Sequence strings must already contain only uppercase canonical amino acids.
There is deliberately no silent stripping, case conversion, or sequence repair.
Original inputs are never modified and returned clone records are deep copies.
"""

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any


TOP_N_PRESETS = (300, 500, 1000)
CANONICAL_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class UniqueCandidate:
    """One distinct CDR-H3 and every contributing Peak clone record."""

    cdr3: str
    score: float
    clone_records: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class BoundaryTie:
    """A score shared by candidates on both sides of the requested cutoff."""

    score: float
    selected_count: int
    excluded_count: int


@dataclass(frozen=True)
class SelectionResult:
    requested: int
    available: int
    candidates: tuple[UniqueCandidate, ...]
    boundary_tie: BoundaryTie | None

    @property
    def returned(self) -> int:
        return len(self.candidates)

    @property
    def shortfall(self) -> int:
        return self.requested - self.returned


def validate_requested_count(requested: int) -> int:
    """Accept any positive integer, including values outside the presets."""
    if isinstance(requested, bool) or not isinstance(requested, int):
        raise TypeError("requested must be a positive integer, not bool")
    if requested <= 0:
        raise ValueError("requested must be greater than zero")
    return requested


def _validate_record(record: Mapping[str, Any], index: int) -> tuple[str, float]:
    """Validate without putting sequence content or metadata in exceptions."""
    prefix = f"record {index}"
    if not isinstance(record, Mapping):
        raise TypeError(f"{prefix} must be a mapping")
    required = ("cdr3", "score", "source_row", "timepoint")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"{prefix} is missing fields: {', '.join(missing)}")
    if record["timepoint"] != "Peak":
        raise ValueError(f"{prefix} timepoint must be exactly 'Peak'")
    cdr3 = record["cdr3"]
    if not isinstance(cdr3, str):
        raise TypeError(f"{prefix} cdr3 must be a string")
    if not cdr3 or not set(cdr3).issubset(CANONICAL_AMINO_ACIDS):
        raise ValueError(f"{prefix} cdr3 must contain uppercase canonical amino acids only")
    score = record["score"]
    if isinstance(score, bool) or not isinstance(score, Real):
        raise TypeError(f"{prefix} score must be a finite real number, not bool")
    try:
        numeric_score = float(score)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{prefix} score must be representable as a finite float") from exc
    if not isfinite(numeric_score):
        raise ValueError(f"{prefix} score must be finite")
    return cdr3, numeric_score


def select_unique_cdr3(
    records: Iterable[Mapping[str, Any]], requested: int
) -> SelectionResult:
    """Return up to ``requested`` distinct sequences from scored Peak records.

    Provisional implementation decision: the representative score of a sequence
    is the maximum score among its contributing clones. This aggregation is not
    claimed to reproduce the paper's duplicate-handling method. All contributing
    clone records, including lower scoring records, are retained in input order.

    Candidates are ranked by descending representative score, then ascending
    CDR3 string. Ties use exact float equality, without rounding or a tolerance.
    A tie spanning the cutoff is reported but does not expand the requested N.
    Insufficient input returns all available unique sequences and a shortfall;
    duplicates are never added to fill the request. Empty input is permitted.

    ``source_row`` is an opaque source identifier whose presence is required;
    its interpretation and uniqueness are the caller's responsibility. Distinct
    clone records may legitimately have the same CDR3. Use metadata to retain
    source file, V/J/isotype, and any other provenance needed by the application.
    Every input record is validated, even if its score is below the cutoff.
    """
    requested = validate_requested_count(requested)
    grouped_records: dict[str, list[dict[str, Any]]] = {}
    scores: dict[str, float] = {}
    for index, record in enumerate(records):
        cdr3, score = _validate_record(record, index)
        grouped_records.setdefault(cdr3, []).append(deepcopy(dict(record)))
        if cdr3 not in scores or score > scores[cdr3]:
            scores[cdr3] = score

    ranked_sequences = sorted(scores, key=lambda cdr3: (-scores[cdr3], cdr3))
    selected_sequences = ranked_sequences[:requested]
    boundary_tie = None
    if 0 < requested < len(ranked_sequences):
        cutoff_score = scores[ranked_sequences[requested - 1]]
        if cutoff_score == scores[ranked_sequences[requested]]:
            selected_count = sum(scores[cdr3] == cutoff_score for cdr3 in selected_sequences)
            excluded_count = sum(
                scores[cdr3] == cutoff_score for cdr3 in ranked_sequences[requested:]
            )
            boundary_tie = BoundaryTie(cutoff_score, selected_count, excluded_count)

    candidates = tuple(
        UniqueCandidate(cdr3, scores[cdr3], tuple(grouped_records[cdr3]))
        for cdr3 in selected_sequences
    )
    return SelectionResult(requested, len(ranked_sequences), candidates, boundary_tie)
