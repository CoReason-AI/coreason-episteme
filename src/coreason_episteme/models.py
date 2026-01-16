# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_episteme

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ConfidenceLevel(str, Enum):
    SPECULATIVE = "SPECULATIVE"
    PLAUSIBLE = "PLAUSIBLE"
    PROBABLE = "PROBABLE"


class GeneticTarget(BaseModel):
    symbol: str
    ensembl_id: str
    druggability_score: float  # From coreason-prism
    novelty_score: float


class KnowledgeGap(BaseModel):
    description: str
    source_nodes: Optional[List[str]] = None


class Hypothesis(BaseModel):
    id: str
    title: str
    knowledge_gap: str  # The problem statement
    proposed_mechanism: str  # The biological pathway
    target_candidate: GeneticTarget

    # Validation
    causal_validation_score: float
    key_counterfactual: str

    # Experimental Design
    killer_experiment_pico: Dict[str, Any]

    # Provenance
    evidence_chain: List[str]  # Links to source papers/nodes
    confidence: ConfidenceLevel
