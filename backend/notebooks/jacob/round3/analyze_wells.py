#!/usr/bin/env python3

import os
import sys

sys.path.append("/Users/jacobroberts/git/foldy/backend/src/notebooks/jacob/round3")

from dna_build_parser import DNABuildParser

# Initialize parser
parser = DNABuildParser("/Users/jacobroberts/git/foldy/backend/src/notebooks/jacob/round3")

print("=== Detailed Well Analysis ===")

# Get full sequences for comparison
wells = ["E02", "C23", "E17"]
well_data = {}

for well in wells:
    for seq, (plate, w) in parser.oligo_sequence_map.items():
        if w == well:
            well_data[well] = seq
            break

# Print full sequences
for well in wells:
    if well in well_data:
        print(f"\n{well} full sequence:")
        print(well_data[well])
    else:
        print(f"\n{well}: NOT FOUND")

# Compare C23 and E17 closely - user said reverse is in one of these
print("\n=== Comparing C23 and E17 ===")
if "C23" in well_data and "E17" in well_data:
    c23_seq = well_data["C23"]
    e17_seq = well_data["E17"]

    print(f"C23 length: {len(c23_seq)}")
    print(f"E17 length: {len(e17_seq)}")

    # Look for differences
    if len(c23_seq) == len(e17_seq):
        differences = []
        for i, (c1, c2) in enumerate(zip(c23_seq, e17_seq)):
            if c1 != c2:
                differences.append(f"Position {i}: C23='{c1}' E17='{c2}'")

        if differences:
            print("Differences found:")
            for diff in differences:
                print(f"  {diff}")
        else:
            print("Sequences are identical!")
    else:
        print("Different lengths - showing first 100 chars:")
        print(f"C23: {c23_seq[:100]}")
        print(f"E17: {e17_seq[:100]}")

# Check E02 in the CSV
print(f"\n=== Checking for E02 in CSV ===")
try:
    with open(
        "/Users/jacobroberts/git/foldy/backend/src/notebooks/jacob/round3/r3_oligos_combined.csv",
        "r",
    ) as f:
        lines = f.readlines()
        e02_found = False
        for i, line in enumerate(lines):
            if "E02" in line or "E2" in line:
                print(f"Line {i+1}: {line.strip()}")
                e02_found = True

        if not e02_found:
            print("E02 not found in CSV - checking nearby wells:")
            for i, line in enumerate(lines):
                if any(well in line for well in ["E01", "E03", "D02", "F02"]):
                    print(f"Line {i+1}: {line.strip()}")

except Exception as e:
    print(f"Error reading CSV: {e}")

# Maybe the Teselagen sequences are wrong? Let's see if there's a partial match
print(f"\n=== Checking for partial matches with Teselagen sequences ===")
teselagen_forward = "ACTCTTCAAAGCGAACCTGAAAGAAACAAAGATAATGTCGA"
teselagen_reverse = "GCCATGGGGTAAAGTTGAAACTGCTGAATAGACTCGCATAT"

for well, seq in well_data.items():
    # Check both directions for forward
    fwd_in_well = teselagen_forward.upper() in seq.upper()
    well_in_fwd = seq.upper() in teselagen_forward.upper()

    # Check both directions for reverse
    rev_in_well = teselagen_reverse.upper() in seq.upper()
    well_in_rev = seq.upper() in teselagen_reverse.upper()

    if fwd_in_well or well_in_fwd or rev_in_well or well_in_rev:
        print(f"{well}: Found partial match!")
        if fwd_in_well:
            print(f"  Forward sequence found in {well}")
        if well_in_fwd:
            print(f"  {well} sequence found in forward")
        if rev_in_well:
            print(f"  Reverse sequence found in {well}")
        if well_in_rev:
            print(f"  {well} sequence found in reverse")
