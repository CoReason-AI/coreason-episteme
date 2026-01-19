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
from typing import List, Optional

from pydantic import BaseModel, Field


class ConfidenceLevel(str, Enum):
    SPECULATIVE = "SPECULATIVE"
    PLAUSIBLE = "PLAUSIBLE"
    PROBABLE = "PROBABLE"


class KnowledgeGapType(str, Enum):
    CLUSTER_DISCONNECT = "CLUSTER_DISCONNECT"
    LITERATURE_INCONSISTENCY = "LITERATURE_INCONSISTENCY"


class CritiqueSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    FATAL = "FATAL"


class Critique(BaseModel):
    source: str
    content: str
    severity: CritiqueSeverity


class PICO(BaseModel):
    population: str
    intervention: str
    comparator: str
    outcome: str


class GeneticTarget(BaseModel):
    symbol: str
    ensembl_id: str
    druggability_score: float  # From coreason-prism
    novelty_score: float


class KnowledgeGap(BaseModel):
    description: str
    type: KnowledgeGapType
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
    killer_experiment_pico: PICO

    # Provenance
    evidence_chain: List[str]  # Links to source papers/nodes
    confidence: ConfidenceLevel

    # Adversarial Review
    critiques: List[Critique] = Field(default_factory=list)


class BridgeResult(BaseModel):
    """
    Result from the BridgeBuilder component.
    Contains the generated hypothesis (if any) and metadata about the search process.
    """

    hypothesis: Optional[Hypothesis]
    bridges_found_count: int
    considered_candidates: List[str]  # Symbols of all candidates considered


class HypothesisTrace(BaseModel):
    """
    Complete audit trail for the hypothesis generation process.
    Captures the 'Why' and 'How' for scientific provenance.
    """

    hypothesis_id: Optional[str] = None
    gap: KnowledgeGap

    # Bridge Phase
    bridges_found_count: int = 0
    considered_candidates: List[str] = Field(default_factory=list)
    excluded_targets_history: List[str] = Field(default_factory=list)

    # Simulation Phase
    causal_validation_score: Optional[float] = None
    key_counterfactual: Optional[str] = None

    # Critique Phase
    critiques: List[Critique] = Field(default_factory=list)
    refinement_retries: int = 0

    # Outcome
    result: Optional[Hypothesis] = None
    status: str = "PENDING"  # ACCEPTED, DISCARDED, REJECTED
