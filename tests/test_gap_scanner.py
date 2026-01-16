# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_episteme

from coreason_episteme.components.gap_scanner import MockGapScanner
from coreason_episteme.interfaces import GapScanner
from coreason_episteme.models import KnowledgeGap


def test_interface_definition() -> None:
    """Test that GapScanner interface is importable."""
    assert GapScanner is not None


def test_mock_gap_scanner_scan_found() -> None:
    """Test that the MockGapScanner returns a gap when one is expected."""
    scanner = MockGapScanner()
    gaps = scanner.scan("DiseaseX")

    assert len(gaps) == 1
    assert isinstance(gaps[0], KnowledgeGap)
    assert "DiseaseX" in gaps[0].description
    assert gaps[0].source_nodes == ["PMID:123456", "PMID:789012"]


def test_mock_gap_scanner_scan_not_found() -> None:
    """Test that the MockGapScanner returns an empty list for 'CleanTarget'."""
    scanner = MockGapScanner()
    gaps = scanner.scan("CleanTarget")

    assert len(gaps) == 0
