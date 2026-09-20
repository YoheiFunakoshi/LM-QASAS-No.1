"""Read-only CPM CSV / Takara-RG XLSX ingestion for three timepoints.

The accepted Type label is a provisional input gate, not a claim that the
whole VDJ sequence is functional. Cseg determines subclass; file names do not.
Counts and Frequency(%) are retained only as source metadata, never weights.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TIMEPOINTS = ("Pre", "Peak", "Post")
CSV_COLUMNS = (
    "Vseg", "Jseg", "CDR3", "AAlength", "NTlength", "Type", "Cseg",
    "Counts", "Frequency(%)",
)
ACCEPTED_TYPES = frozenset({"WithConserved_NoStop"})
_CANONICAL_AA = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]+\Z")
_INTEGER = re.compile(r"[0-9]+\Z")
_ISOTYPES = {
    "IGHG1": "IGHG1", "IGHG2": "IGHG2", "IGHG3": "IGHG3", "IGHG4": "IGHG4",
    "IGHA1": "IGHA1", "IGHA2": "IGHA2", "IGHM": "IGHM", "IGHD": "IGHD", "IGHE": "IGHE",
    "IgG1": "IGHG1", "IgG2": "IGHG2", "IgG3": "IGHG3", "IgG4": "IGHG4",
    "IgA1": "IGHA1", "IgA2": "IGHA2", "IgM": "IGHM", "IgD": "IGHD", "IgE": "IGHE",
}


class InputValidationError(ValueError):
    """An input failure whose message excludes source paths and row values."""


@dataclass(frozen=True)
class InputBundle:
    """Local-only records and audit; source_file is an absolute resolved path."""

    clones: list[dict[str, Any]]
    audit: dict[str, Any]
    input_hashes: dict[str, str]


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _gene_set(raw: str, prefix: str) -> tuple[str, ...] | None:
    """Retain an unresolved annotation set rather than choosing its first item."""
    genes = set()
    pattern = re.compile(rf"{prefix}[A-Z0-9][A-Z0-9/-]*(?:\*[A-Z0-9]+)?\Z")
    for annotation in raw.split("//"):
        annotation = annotation.strip()
        if not pattern.fullmatch(annotation):
            return None
        genes.add(annotation.split("*", 1)[0])
    return tuple(sorted(genes)) if genes else None


def _isotype(raw: str) -> str | None:
    annotation = raw.strip()
    if "*" in annotation:
        parts = annotation.split("*")
        if len(parts) != 2 or not re.fullmatch(r"[A-Z0-9]+", parts[1]):
            return None
        annotation = parts[0]
    return _ISOTYPES.get(annotation)


def _nonnegative_number(raw: str, maximum: float | None = None) -> bool:
    try:
        value = float(raw)
    except (ValueError, OverflowError):
        return False
    return math.isfinite(value) and value >= 0 and (maximum is None or value <= maximum)


def _validate_row(raw: dict[str, str]) -> tuple[list[str], tuple[str, ...] | None, tuple[str, ...] | None, str | None]:
    reasons = []
    cdr3 = raw["CDR3"]
    if not _CANONICAL_AA.fullmatch(cdr3):
        reasons.append("cdr3_noncanonical")
    if len(cdr3) < 5:
        reasons.append("cdr3_too_short")
    aa = raw["AAlength"]
    nt = raw["NTlength"]
    if not _INTEGER.fullmatch(aa):
        reasons.append("aa_length_invalid")
    elif int(aa) != len(cdr3):
        reasons.append("aa_length_mismatch")
    if not _INTEGER.fullmatch(nt):
        reasons.append("nt_length_invalid")
    elif int(nt) != 3 * len(cdr3):
        reasons.append("nt_length_mismatch")
    if raw["Type"] not in ACCEPTED_TYPES:
        reasons.append("type_not_accepted")
    v_genes = _gene_set(raw["Vseg"], "IGHV")
    j_genes = _gene_set(raw["Jseg"], "IGHJ")
    if v_genes is None:
        reasons.append("v_annotation_invalid")
    if j_genes is None:
        reasons.append("j_annotation_invalid")
    isotype = _isotype(raw["Cseg"])
    if isotype is None:
        reasons.append("cseg_unmapped")
    if not _nonnegative_number(raw["Counts"]):
        reasons.append("counts_invalid")
    if not _nonnegative_number(raw["Frequency(%)"], maximum=100):
        reasons.append("frequency_invalid")
    return reasons, v_genes, j_genes, isotype


def _parse_csv(data: bytes, path: Path, timepoint: str, subject: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeError:
        raise InputValidationError(f"{timepoint}: CSV must use UTF-8 encoding.") from None
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    exclusions = []
    reason_counts: Counter[str] = Counter()
    total = accepted = ambiguous_v = ambiguous_j = 0
    try:
        header = next(reader, None)
        if header is None or len(header) != len(CSV_COLUMNS) or set(header) != set(CSV_COLUMNS):
            raise InputValidationError(f"{timepoint}: CSV must have the exact required column names, once each.")
        previous_line = reader.line_num
        for row in reader:
            source_row = previous_line + 1
            previous_line = reader.line_num
            if len(row) != len(header):
                raise InputValidationError(f"{timepoint}: malformed CSV row width at line {source_row}.")
            total += 1
            raw = dict(zip(header, row, strict=True))
            reasons, v_genes, j_genes, isotype = _validate_row(raw)
            if reasons:
                exclusions.append({"source_row": source_row, "reasons": reasons})
                reason_counts.update(reasons)
                continue
            accepted += 1
            ambiguous_v += len(v_genes) > 1
            ambiguous_j += len(j_genes) > 1
            key = (v_genes, j_genes, raw["CDR3"], isotype)
            if key not in grouped:
                grouped[key] = {
                    "cdr3": raw["CDR3"],
                    "v_gene": "//".join(v_genes),
                    "j_gene": "//".join(j_genes),
                    "isotype": isotype,
                    "timepoint": timepoint,
                    "subject": subject,
                    "source_file": str(path),
                    "source_row": source_row,
                    "source_rows": [],
                    "raw_records": [],
                }
            grouped[key]["source_rows"].append(source_row)
            grouped[key]["raw_records"].append(raw)
    except csv.Error:
        raise InputValidationError(f"{timepoint}: malformed CSV syntax.") from None
    if not grouped:
        raise InputValidationError(f"{timepoint}: no rows remain after input validation.")
    audit = {
        "source_file": str(path),
        "total_rows": total,
        "accepted_rows": accepted,
        "excluded_rows": len(exclusions),
        "clone_count": len(grouped),
        "merged_rows": accepted - len(grouped),
        "reason_counts": dict(sorted(reason_counts.items())),
        "exclusions": exclusions,
        "rows_with_multiple_v_genes": ambiguous_v,
        "rows_with_multiple_j_genes": ambiguous_j,
        "full_vdj_functionality_verified": False,
    }
    return list(grouped.values()), audit


def load_three_inputs(paths: Mapping[str, Path], subject: str) -> InputBundle:
    """Validate three same-format inputs and group within-timepoint clone keys.

    The header may be reordered, but it must contain exactly CSV_COLUMNS.
    source_row/source_rows refer to physical starting line numbers, header = 1.
    Raw field strings are preserved without edits; annotation normalization only
    affects clone keys. InputBundle and audit contain private input provenance.
    """
    if not isinstance(paths, Mapping) or set(paths) != set(TIMEPOINTS):
        raise InputValidationError("Exactly Pre, Peak and Post input paths are required.")
    if not isinstance(subject, str) or not subject.strip() or any(ord(c) < 32 for c in subject):
        raise InputValidationError("Subject must be a nonempty text identifier without control characters.")
    resolved: dict[str, Path] = {}
    for timepoint in TIMEPOINTS:
        try:
            path = Path(paths[timepoint]).resolve(strict=True)
            if not path.is_file():
                raise OSError()
            if any(path.samefile(other) for other in resolved.values()):
                raise InputValidationError("Timepoints must use three different physical files.")
            resolved[timepoint] = path
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            if isinstance(exc, InputValidationError):
                raise
            raise InputValidationError(f"{timepoint}: input file is missing or inaccessible.") from None
    clones = []
    samples = {}
    hashes = {}
    suffixes = {path.suffix.lower() for path in resolved.values()}
    if len(suffixes) != 1 or not suffixes <= {'.csv', '.xlsx'}:
        raise InputValidationError('Use three files of the same supported format: CPM CSV or Takara/RG XLSX.')
    is_excel = suffixes == {'.xlsx'}
    for timepoint, path in resolved.items():
        try:
            data = path.read_bytes()
            before = hashlib.sha256(data).hexdigest()
            try:
                if is_excel:
                    from .takara import parse_workbook
                    sample_clones, sample_audit = parse_workbook(data, path, timepoint, subject)
                else:
                    sample_clones, sample_audit = _parse_csv(data, path, timepoint, subject)
                    sample_audit.update(input_format='cpm_csv', input_policy='cpm-csv-input-v1',
                                        isotype_granularity='subclass',
                                        cdr3_definition='as_reported_no_boundary_repair')
            finally:
                if before != _sha256_path(path):
                    raise InputValidationError(f"{timepoint}: input changed during reading; results discarded.")
            clones.extend(sample_clones)
            sample_audit["source_unchanged_after_read"] = True
            samples[timepoint] = sample_audit
            hashes[timepoint] = before
        except OSError:
            raise InputValidationError(f"{timepoint}: input could not be read or rechecked.") from None
    return InputBundle(
        clones=clones,
        input_hashes=hashes,
        audit={
            "policy_version": "input-v2",
            "samples": samples,
            "policies": {
                "input_format": 'takara_rg_xlsx' if is_excel else 'cpm_csv',
                "accepted_types": None if is_excel else sorted(ACCEPTED_TYPES),
                "minimum_cdr3_length": 5,
                "nt_length_rule": 'not_available_not_synthesized' if is_excel else "NTlength equals 3 times the observed CDR3 amino-acid length",
                "gene_normalization": ('strip annotation whitespace and alleles; sorted distinct comma alternatives'
                                       if is_excel else 'strip surrounding annotation whitespace and alleles; sorted distinct // alternatives'),
                "isotype": 'unambiguous C annotation after allele removal; subclass retained' if is_excel else "explicit Cseg mapping preserving subclass; filename ignored",
                "clone_key": ["v_gene_annotation_set", "j_gene_annotation_set", "cdr3", "isotype"],
                "counts_used_as_weights": False,
                "d_annotation_used_for_eligibility": False,
                "d_function_used_for_eligibility": False,
                "d_used_in_clone_key": False,
                "full_vdj_functionality_verified": False,
            },
            "limitations": ([
                'Excel source frame=in-frame and V/J labels=F are required, but not independently reannotated. D calls and D function labels do not control eligibility.',
                'No source NT sequence or independent full-VDJ productivity result is available.',
                'CDR3 is preserved as reported; vendor boundary equivalence with CPM is unconfirmed.',
                'Multiple constant calls are accepted only when every allele maps to one subclass.',
                'Counts do not weight clone observations; the top-50 ranking is not the repertoire input.',
            ] if is_excel else [
                "Type is an input quality label, not an isotype or independent proof of full VDJ functionality.",
                "D-gene, VDJ frame, functional-gene and ORF/pseudogene status cannot be independently verified from these columns.",
                "V/J alternatives remain unresolved annotation sets; different unresolved sets are different clone keys.",
                "Cseg subclass granularity and allele removal are provisional implementation choices.",
                "Counts units are unconfirmed; Counts and Frequency(%) do not weight clone observations.",
            ]),
        },
    )
