"""Utilities for Foldy-managed Boltz multiple-sequence alignments."""

import csv
import io

from werkzeug.exceptions import BadRequest


def rewrite_msa_query_sequence(msa_bytes: bytes, sequence: str) -> bytes:
    """Replace only the first query sequence in a Boltz CSV MSA."""
    source = io.StringIO(msa_bytes.decode("utf-8"))
    rows = list(csv.reader(source))
    if len(rows) < 2 or len(rows[0]) < 2 or rows[0][1] != "sequence":
        raise BadRequest("Source MSA is not a Boltz CSV MSA with a sequence column.")
    if len(rows[1]) < 2:
        raise BadRequest("Source MSA query row is malformed.")
    rows[1][1] = sequence
    output = io.StringIO()
    csv.writer(output, lineterminator="\n").writerows(rows)
    return output.getvalue().encode("utf-8")
