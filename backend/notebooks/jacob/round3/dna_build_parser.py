"""
DNA Build Parser for J5 ELN Data

This module provides classes to parse and access DNA build information from J5 ELN JSON files
and associated plate mapping CSV files.
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional

import pandas as pd


class Aliquot(NamedTuple):
    """Represents an aliquot with source plate, well, volume, and expected sequence information."""

    source_plate: str
    source_well: str
    volume: float
    expected_sequence: Optional[str] = None


@dataclass
class PCRComponents:
    """Components needed for a PCR reaction."""

    left_primer: Aliquot
    right_primer: Aliquot
    template: Aliquot


class DNABuildParser:
    """Parser for J5 ELN DNA build data."""

    def __init__(self, round3_dir: str):
        """
        Initialize parser with path to round3 directory.

        Args:
            round3_dir: Path to directory containing build folders and CSV files
        """
        self.round3_dir = round3_dir
        assert os.path.exists(round3_dir), f"Round3 directory does not exist: {round3_dir}"

        # Load CSV plate mappings
        self._load_plate_mappings()

        # Cache for loaded JSON data
        self._json_cache: Dict[str, Dict] = {}

    def _load_plate_mappings(self) -> None:
        """Load oligo and template plate mappings from CSV files."""
        # Load IDT specsheet as primary source
        idt_csv = os.path.join(self.round3_dir, "idt_specsheet_r3_oligos.csv")

        # Load second oligo CSV for R2_random oligos
        random_oligo_csv2 = os.path.join(
            self.round3_dir, "teselagen_plate_folde_r2_oligos_dbat_oplr_random.csv"
        )

        assert os.path.exists(idt_csv), f"IDT specsheet CSV not found: {idt_csv}"
        assert os.path.exists(random_oligo_csv2), f"R2 oligo CSV not found: {random_oligo_csv2}"

        # Load IDT specsheet
        idt_map = pd.read_csv(idt_csv)
        assert "Well Position" in idt_map.columns, "Missing 'Well Position' column in IDT CSV"
        assert "Sequence Name" in idt_map.columns, "Missing 'Sequence Name' column in IDT CSV"
        assert "Sequence" in idt_map.columns, "Missing 'Sequence' column in IDT CSV"

        # Load R2 oligo CSV (for R2_random oligos)
        random_oligo_plate_details = pd.read_csv(random_oligo_csv2)
        assert (
            "Well Location" in random_oligo_plate_details.columns
        ), "Missing 'Well Location' column in R2 oligo CSV"
        assert (
            "Sample Name" in random_oligo_plate_details.columns
        ), "Missing 'Sample Name' column in R2 oligo CSV"

        # Create sequence-to-well mapping from IDT data
        self.oligo_sequence_map = {}  # normalized_sequence -> (plate_name, well)

        for _, row in idt_map.iterrows():
            sequence = row["Sequence"]
            well = row["Well Position"]
            plate_name = row.get("Plate Name", "oligo_plate")

            # Normalize sequence: remove spaces, uppercase
            normalized_seq = sequence.replace(" ", "").upper()
            self.oligo_sequence_map[normalized_seq] = (plate_name, well)

        # Also create old-style name mapping from R2 CSV for R2_random oligos
        self.oligo_plate_map = {}  # name -> well (for R2_random oligos only)
        for _, row in random_oligo_plate_details.iterrows():
            sample_name = row["Sample Name"]
            well = row["Well Location"]
            sequence = row["Sequence"]
            normalized_seq = sequence.replace(" ", "").upper()
            # Use a consistent plate name for R2 oligos
            self.oligo_sequence_map[normalized_seq] = ("random_oligo_plate", well)
            # self.oligo_plate_map[sample_name] = well

        print(f"Loaded {len(self.oligo_sequence_map)} oligos from IDT specsheet")
        print(f"Loaded {len(self.oligo_plate_map)} R2_random oligos from secondary CSV")

        # Load template mappings
        template_csv = os.path.join(self.round3_dir, "r3_template_plate_take4.csv")
        assert os.path.exists(template_csv), f"Template CSV not found: {template_csv}"

        self.template_map = pd.read_csv(template_csv)
        assert (
            "Well Location" in self.template_map.columns
        ), "Missing 'Well Location' column in template CSV"
        assert (
            "Sample Name" in self.template_map.columns
        ), "Missing 'Sample Name' column in template CSV"

        # Create lookup dict: Sample Name -> Well Location
        self.template_plate_map = dict(
            zip(self.template_map["Sample Name"], self.template_map["Well Location"])
        )

    def _load_build_json(self, build_name: str) -> Dict:
        """Load and cache JSON data for a build."""
        if build_name not in self._json_cache:
            result_file = os.path.join(self.round3_dir, build_name, "result.json")
            assert os.path.exists(result_file), f"Result JSON not found: {result_file}"

            with open(result_file, "r") as f:
                self._json_cache[build_name] = json.load(f)

        return self._json_cache[build_name]

    def getBuildNames(self) -> List[str]:
        """Return a list of the names of available builds."""
        build_names = []
        for item in os.listdir(self.round3_dir):
            item_path = os.path.join(self.round3_dir, item)
            if os.path.isdir(item_path):
                result_json = os.path.join(item_path, "result.json")
                if os.path.exists(result_json):
                    build_names.append(item)

        assert len(build_names) > 0, "No build directories with result.json found"
        return sorted(build_names)

    def getAssemblyNamesForBuild(self, build_name: str) -> List[str]:
        """Return a list of assembly products for a build (should be plasmid names)."""
        assert build_name in self.getBuildNames(), f"Build not found: {build_name}"

        data = self._load_build_json(build_name)

        # Assembly products are in j5RunConstruct
        constructs = data["output"]["j5RunConstruct"]
        assembly_names = [construct["name"] for construct in constructs.values()]

        assert len(assembly_names) > 0, f"No assembly products found for build: {build_name}"
        return sorted(assembly_names)

    def getAssemblyPCRs(self, assembly_name: str) -> List[str]:
        """Return the list of PCRs necessary for this assembly reaction."""
        # Find the build that contains this assembly
        build_name = None
        for candidate_build in self.getBuildNames():
            if assembly_name in self.getAssemblyNamesForBuild(candidate_build):
                build_name = candidate_build
                break

        assert build_name is not None, f"Assembly not found in any build: {assembly_name}"

        data = self._load_build_json(build_name)

        # Find the construct ID for this assembly
        constructs = data["output"]["j5RunConstruct"]
        target_construct_id = None
        for construct_id, construct_data in constructs.items():
            if construct_data["name"] == assembly_name:
                target_construct_id = construct_id
                break

        assert (
            target_construct_id is not None
        ), f"Construct ID not found for assembly: {assembly_name}"

        # Use J5 relationship data to find assembly pieces for this construct
        construct_pieces = data["output"]["j5ConstructAssemblyPiece"]
        assembly_pieces = data["output"]["j5AssemblyPiece"]

        # Find assembly pieces linked to this construct
        assembly_piece_ids = []
        for link in construct_pieces.values():
            if link["j5RunConstructId"] == target_construct_id:
                assembly_piece_ids.append(link["assemblyPieceId"])

        assert (
            len(assembly_piece_ids) > 0
        ), f"No assembly pieces found for construct: {assembly_name}"

        # Get PCR names from assembly pieces
        pcr_names = []
        for piece_id in assembly_piece_ids:
            if piece_id in assembly_pieces:
                piece = assembly_pieces[piece_id]
                if piece["type"] == "PCR":
                    # Assembly piece name is like AP_AP_OplR_CH_R3_0001
                    # Need to map to PCR name like PCR_AP_OplR_CH_R3_0001
                    piece_name = piece["name"]
                    if piece_name.startswith("AP_"):
                        pcr_name = "PCR_" + piece_name[3:]  # Remove 'AP_' and add 'PCR_'
                        pcr_names.append(pcr_name)

        assert len(pcr_names) > 0, f"No PCRs found for assembly: {assembly_name}"
        return sorted(pcr_names)

    def getPCRComponents(
        self, pcr_name: str, primer_volume: float, template_volume: float
    ) -> PCRComponents:
        """Return the components necessary for that PCR."""
        # Find the build that contains this PCR
        build_name = None
        pcr_data = None

        for candidate_build in self.getBuildNames():
            data = self._load_build_json(candidate_build)
            pcr_reactions = data["output"]["j5PcrReaction"]

            for pcr in pcr_reactions.values():
                if pcr["name"] == pcr_name:
                    build_name = candidate_build
                    pcr_data = pcr
                    break
            if pcr_data:
                break
        # print(f'FOUND PCR DATA FOR {pcr_name}: {pcr_data}')

        assert pcr_data is not None, f"PCR not found: {pcr_name}"
        assert build_name is not None, f"Build not found for PCR: {pcr_name}"

        data = self._load_build_json(build_name)

        # Get primer and template information
        forward_primer_id = pcr_data["forwardPrimerId"]
        reverse_primer_id = pcr_data["reversePrimerId"]
        template_id = pcr_data["primaryTemplateId"]

        # Look up primers in j5Oligo
        j5_oligo = data["output"]["j5Oligo"]
        sequences = data["output"]["sequence"]

        forward_oligo_entry = None
        reverse_oligo_entry = None

        for oligo in j5_oligo.values():
            if oligo["id"] == forward_primer_id:
                forward_oligo_entry = oligo
            elif oligo["id"] == reverse_primer_id:
                reverse_oligo_entry = oligo

        assert (
            forward_oligo_entry is not None
        ), f"Forward primer not found for ID: {forward_primer_id}"
        assert (
            reverse_oligo_entry is not None
        ), f"Reverse primer not found for ID: {reverse_primer_id}"

        # Get sequence data from the sequence collection
        forward_sequence_id = forward_oligo_entry["sequenceId"]
        reverse_sequence_id = reverse_oligo_entry["sequenceId"]

        assert (
            forward_sequence_id in sequences
        ), f"Forward primer sequence not found for ID: {forward_sequence_id}"
        assert (
            reverse_sequence_id in sequences
        ), f"Reverse primer sequence not found for ID: {reverse_sequence_id}"

        forward_sequence_entry = sequences[forward_sequence_id]
        reverse_sequence_entry = sequences[reverse_sequence_id]

        # print(f'FOUND FORWARD OLIGO FOR {pcr_name}: {forward_sequence_entry}')
        # print(f'FOUND REVERSE OLIGO FOR {pcr_name}: {reverse_sequence_entry}')

        forward_primer_name = forward_sequence_entry["name"]
        reverse_primer_name = reverse_sequence_entry["name"]

        # Look up template in sequences
        sequences = data["output"]["sequence"]
        assert template_id in sequences, f"Template sequence not found for ID: {template_id}"
        template_name = sequences[template_id]["name"]

        # Get actual sequences from the sequence entries
        forward_oligo_seq = forward_sequence_entry["sequence"]
        reverse_oligo_seq = reverse_sequence_entry["sequence"]

        # Normalize sequences for lookup
        forward_seq_normalized = forward_oligo_seq.replace(" ", "").upper()
        reverse_seq_normalized = reverse_oligo_seq.replace(" ", "").upper()

        # Look up by sequence in IDT data first, fall back to name-based lookup for R2_random
        def find_oligo_well(primer_name: str, sequence: str) -> tuple[str, str]:
            # Try exact sequence match first (IDT data)
            if sequence in self.oligo_sequence_map:
                plate_name, well = self.oligo_sequence_map[sequence]
                return (plate_name, well)

            # Fall back to name-based lookup for R2_random oligos
            if primer_name in self.oligo_plate_map:
                well = self.oligo_plate_map[primer_name]
                return ("r2_oligo_plate", well)

            # If not found, provide helpful error
            raise AssertionError(
                f"Primer not found - Name: {primer_name}, Sequence: {sequence}, examples: {[(k, v) for k, v in self.oligo_sequence_map.items() if v[1] == 'A12']}"
            )

        forward_plate, forward_well = find_oligo_well(forward_primer_name, forward_seq_normalized)
        reverse_plate, reverse_well = find_oligo_well(reverse_primer_name, reverse_seq_normalized)

        # Get template location
        assert (
            template_name in self.template_plate_map
        ), f"Template not found in plate map: {template_name}"
        template_well = self.template_plate_map[template_name]

        # Create aliquot objects
        left_primer = Aliquot(forward_plate, forward_well, primer_volume, forward_seq_normalized)
        right_primer = Aliquot(reverse_plate, reverse_well, primer_volume, reverse_seq_normalized)
        template = Aliquot("template_plate", template_well, template_volume)

        return PCRComponents(left_primer, right_primer, template)


def _well_to_row_col(well: str) -> tuple[int, int]:
    """Convert well name (e.g., 'A1') to (row, col) indices (0-based)."""
    assert len(well) >= 2, f"Invalid well format: {well}"
    row = ord(well[0].upper()) - ord("A")
    col = int(well[1:]) - 1
    assert 0 <= row < 8, f"Invalid row in well: {well} (must be A-H)"
    assert 0 <= col < 12, f"Invalid column in well: {well} (must be 1-12)"
    return row, col


def _row_col_to_well(row: int, col: int) -> str:
    """Convert (row, col) indices to well name."""
    assert 0 <= row < 8, f"Invalid row: {row} (must be 0-7)"
    assert 0 <= col < 12, f"Invalid column: {col} (must be 0-11)"
    return f"{chr(ord('A') + row)}{col + 1}"


def _increment_well(well: str) -> str:
    """Increment well by 1 position (e.g., A1 -> A2, A12 -> B1)."""
    row, col = _well_to_row_col(well)
    col += 1
    if col >= 12:
        col = 0
        row += 1
    assert row < 8, f"Well overflow beyond H12 from {well}"
    return _row_col_to_well(row, col)


def create_pcr_plate_assignments(
    build_pcr_blocks: Dict[str, tuple[str, str, str]], parser: DNABuildParser
) -> List[tuple[str, str, str]]:
    """
    Create PCR plate assignments for builds.

    Args:
        build_pcr_blocks: Dict mapping build_name -> (plate_name, pcr1_start_well, pcr2_start_well)
        parser: DNABuildParser instance

    Returns:
        List of (plate_name, well, pcr_name) tuples for all PCRs
    """
    assignments = []

    for build_name, (plate_name, pcr1_start, pcr2_start) in build_pcr_blocks.items():
        assert build_name in parser.getBuildNames(), f"Build not found: {build_name}"

        # Get assemblies for this build, sorted by name
        assemblies = sorted(parser.getAssemblyNamesForBuild(build_name))

        # Track current wells for PCR 1 and PCR 2
        current_pcr1_well = None
        current_pcr2_well = None

        # Keep track of all assigned wells to detect collisions
        assigned_wells = set()

        for assembly in assemblies:
            # Get PCRs for this assembly
            pcrs = sorted(parser.getAssemblyPCRs(assembly))
            assert (
                len(pcrs) == 2
            ), f"Expected exactly 2 PCRs for {assembly}, got {len(pcrs)}: {pcrs}"

            pcr1, pcr2 = pcrs

            # Update wells: increment if not None, otherwise use starting well
            if current_pcr1_well is not None:
                current_pcr1_well = _increment_well(current_pcr1_well)
            else:
                current_pcr1_well = pcr1_start

            if current_pcr2_well is not None:
                current_pcr2_well = _increment_well(current_pcr2_well)
            else:
                current_pcr2_well = pcr2_start

            # Check for collisions
            assert (
                current_pcr1_well not in assigned_wells
            ), f"Well collision at {current_pcr1_well} for PCR1 {pcr1}"
            assert (
                current_pcr2_well not in assigned_wells
            ), f"Well collision at {current_pcr2_well} for PCR2 {pcr2}"

            # Assign wells
            assignments.append((plate_name, current_pcr1_well, pcr1))
            assignments.append((plate_name, current_pcr2_well, pcr2))

            assigned_wells.add(current_pcr1_well)
            assigned_wells.add(current_pcr2_well)

    return assignments


def create_echo_instructions(
    pcr_assignments: List[tuple[str, str, str]],
    parser: DNABuildParser,
    primer_volume: float = 2.5,
    template_volume: float = 1.0,
) -> pd.DataFrame:
    """
    Create Echo instructions from PCR plate assignments.

    Args:
        pcr_assignments: List of (plate_name, well, pcr_name) tuples from create_pcr_plate_assignments
        parser: DNABuildParser instance
        primer_volume: Volume of each primer to transfer (μL)
        template_volume: Volume of template to transfer (μL)

    Returns:
        DataFrame with columns: source_plate, source_well, dest_plate, dest_well, volume
    """
    instructions = []

    for plate_name, dest_well, pcr_name in pcr_assignments:
        # Get PCR components
        components = parser.getPCRComponents(pcr_name, primer_volume, template_volume)

        # Add left primer transfer
        instructions.append(
            {
                "source_plate": components.left_primer.source_plate,
                "source_well": components.left_primer.source_well,
                "dest_plate": plate_name,
                "dest_well": dest_well,
                "volume": components.left_primer.volume,
            }
        )

        # Add right primer transfer
        instructions.append(
            {
                "source_plate": components.right_primer.source_plate,
                "source_well": components.right_primer.source_well,
                "dest_plate": plate_name,
                "dest_well": dest_well,
                "volume": components.right_primer.volume,
            }
        )

        # Add template transfer
        instructions.append(
            {
                "source_plate": components.template.source_plate,
                "source_well": components.template.source_well,
                "dest_plate": plate_name,
                "dest_well": dest_well,
                "volume": components.template.volume,
            }
        )

    return pd.DataFrame(instructions)
