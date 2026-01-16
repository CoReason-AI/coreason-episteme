# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_episteme

from typing import Any, Dict, List, Optional, Protocol

from coreason_episteme.models import GeneticTarget, Hypothesis, KnowledgeGap

# --- External Service Interfaces ---


class GraphNexusClient(Protocol):
    """Interface for coreason-graph-nexus (The Map)."""

    def find_disconnected_clusters(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Finds disconnected subgraphs matching criteria."""
        ...

    def find_latent_bridges(self, source_cluster_id: str, target_cluster_id: str) -> List[GeneticTarget]:
        """Finds potential bridges between two clusters."""
        ...


class InferenceClient(Protocol):
    """Interface for coreason-inference (The Lab)."""

    def run_counterfactual_simulation(self, mechanism: str, intervention_target: str) -> float:
        """
        Runs a counterfactual simulation.
        Returns a plausibility score (0.0 - 1.0).
        """
        ...


class CodexClient(Protocol):
    """Interface for coreason-codex (The Dictionary)."""

    def get_semantic_similarity(self, entity1: str, entity2: str) -> float:
        """Calculates semantic similarity between two entities."""
        ...

    def validate_target(self, symbol: str) -> Optional[GeneticTarget]:
        """Validates a genetic target and returns its details."""
        ...


class PrismClient(Protocol):
    """Interface for coreason-prism."""

    def check_druggability(self, target_id: str) -> float:
        """Returns a druggability score for the target."""
        ...


class SearchClient(Protocol):
    """Interface for coreason-search (Scout)."""

    def find_literature_inconsistency(self, topic: str) -> List[KnowledgeGap]:
        """Finds inconsistencies in literature regarding a topic."""
        ...

    def verify_citation(self, interaction_claim: str) -> bool:
        """Verifies if a claimed interaction is supported by literature."""
        ...


class VeritasClient(Protocol):
    """Interface for coreason-veritas (The Log)."""

    def log_trace(self, hypothesis_id: str, trace_data: Dict[str, Any]) -> None:
        """Logs the hypothesis generation trace."""
        ...


# --- Internal Component Interfaces ---


class GapScanner(Protocol):
    """Interface for components that identify knowledge gaps."""

    def scan(self, target: str) -> List[KnowledgeGap]:
        """
        Scans for knowledge gaps related to the target.

        Args:
            target: The disease or biological entity to scan for.

        Returns:
            A list of KnowledgeGap objects.
        """
        ...


class BridgeBuilder(Protocol):
    """Interface for the Bridge Builder (Hypothesis Formulator)."""

    def generate_hypothesis(self, gap: KnowledgeGap) -> Optional[Hypothesis]:
        """Generates a hypothesis bridging the knowledge gap."""
        ...


class CausalValidator(Protocol):
    """Interface for the Causal Validator (The Simulator)."""

    def validate(self, hypothesis: Hypothesis) -> Hypothesis:
        """
        Validates the hypothesis using causal simulation.
        Updates the hypothesis with validation score.
        """
        ...


class ProtocolDesigner(Protocol):
    """Interface for the Protocol Designer (The Experimentalist)."""

    def design_experiment(self, hypothesis: Hypothesis) -> Hypothesis:
        """
        Designs the killer experiment for the hypothesis.
        Updates the hypothesis with PICO details.
        """
        ...
