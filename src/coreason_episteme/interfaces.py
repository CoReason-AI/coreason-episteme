# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_episteme

"""
Interface definitions for external clients and internal components.

This module uses Protocol to define the contracts that external dependencies
and internal components must adhere to.
"""

from typing import Any, Dict, List, Optional, Protocol

from coreason_episteme.models import BridgeResult, GeneticTarget, Hypothesis, KnowledgeGap

# --- External Service Interfaces ---


class GraphNexusClient(Protocol):
    """
    Interface for coreason-graph-nexus (The Map).

    Provides access to the Knowledge Graph for finding disconnected clusters
    and latent bridges.
    """

    async def find_disconnected_clusters(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Finds disconnected subgraphs matching criteria.

        Args:
            criteria: A dictionary of criteria to filter clusters.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing disconnected clusters.
        """
        ...

    async def find_latent_bridges(self, source_cluster_id: str, target_cluster_id: str) -> List[GeneticTarget]:
        """
        Finds potential bridges between two clusters.

        Args:
            source_cluster_id: The ID of the source cluster.
            target_cluster_id: The ID of the target cluster.

        Returns:
            List[GeneticTarget]: A list of GeneticTarget objects representing potential bridges.
        """
        ...


class InferenceClient(Protocol):
    """
    Interface for coreason-inference (The Lab).

    Provides capabilities for running causal simulations, toxicology screens,
    and clinical redundancy checks.
    """

    async def run_counterfactual_simulation(self, mechanism: str, intervention_target: str) -> float:
        """
        Runs a counterfactual simulation.

        Args:
            mechanism: The proposed biological mechanism.
            intervention_target: The target to intervene on.

        Returns:
            float: A plausibility score between 0.0 and 1.0.
        """
        ...

    async def run_toxicology_screen(self, target_candidate: GeneticTarget) -> List[str]:
        """
        Runs a toxicology screen on the target.

        Args:
            target_candidate: The genetic target to screen.

        Returns:
            List[str]: A list of potential toxicity risks.
        """
        ...

    async def check_clinical_redundancy(self, mechanism: str, target_candidate: GeneticTarget) -> List[str]:
        """
        Checks if the proposed mechanism/target is redundant with existing clinical interventions.

        Args:
            mechanism: The proposed mechanism.
            target_candidate: The proposed target.

        Returns:
            List[str]: A list of redundancy warnings.
        """
        ...


class CodexClient(Protocol):
    """
    Interface for coreason-codex (The Dictionary).

    Provides ontology services, semantic similarity calculations, and target validation.
    """

    async def get_semantic_similarity(self, entity1: str, entity2: str) -> float:
        """
        Calculates semantic similarity between two entities.

        Args:
            entity1: The first entity.
            entity2: The second entity.

        Returns:
            float: A float representing the semantic similarity score.
        """
        ...

    async def validate_target(self, symbol: str) -> Optional[GeneticTarget]:
        """
        Validates a genetic target and returns its details.

        Args:
            symbol: The gene symbol to validate.

        Returns:
            Optional[GeneticTarget]: A GeneticTarget object if valid, else None.
        """
        ...


class PrismClient(Protocol):
    """
    Interface for coreason-prism.

    Provides druggability assessment for potential targets.
    """

    async def check_druggability(self, target_id: str) -> float:
        """
        Returns a druggability score for the target.

        Args:
            target_id: The identifier of the target.

        Returns:
            float: A float representing the druggability score.
        """
        ...


class SearchClient(Protocol):
    """
    Interface for coreason-search (Scout).

    Provides literature search capabilities, including finding inconsistencies,
    verifying citations, checking patents, and finding disconfirming evidence.
    """

    async def find_literature_inconsistency(self, topic: str) -> List[KnowledgeGap]:
        """
        Finds inconsistencies in literature regarding a topic.

        Args:
            topic: The topic to search for inconsistencies.

        Returns:
            List[KnowledgeGap]: A list of KnowledgeGap objects representing found inconsistencies.
        """
        ...

    async def verify_citation(self, interaction_claim: str) -> bool:
        """
        Verifies if a claimed interaction is supported by literature.

        Args:
            interaction_claim: The claim to verify.

        Returns:
            bool: True if supported, False otherwise.
        """
        ...

    async def check_patent_infringement(self, target_candidate: GeneticTarget, mechanism: str) -> List[str]:
        """
        Checks for potential patent infringement.

        Args:
            target_candidate: The target candidate to check.
            mechanism: The proposed mechanism.

        Returns:
            List[str]: A list of relevant patents or conflicts.
        """
        ...

    async def find_disconfirming_evidence(self, subject: str, object: str, action: str) -> List[str]:
        """
        Searches for evidence that contradicts the proposed relationship.
        e.g., "Gene X does NOT regulate Pathway Y".

        Args:
            subject: The subject of the relationship.
            object: The object of the relationship.
            action: The action or relationship type.

        Returns:
            List[str]: A list of citations or snippets supporting the null hypothesis.
        """
        ...


class VeritasClient(Protocol):
    """
    Interface for coreason-veritas (The Log).

    Responsible for logging the full hypothesis generation trace for provenance.
    """

    async def log_trace(self, hypothesis_id: str, trace_data: Dict[str, Any]) -> None:
        """
        Logs the hypothesis generation trace.

        Args:
            hypothesis_id: The ID of the hypothesis.
            trace_data: A dictionary containing the trace data.
        """
        ...


# --- Internal Component Interfaces ---


class GapScanner(Protocol):
    """
    Interface for components that identify knowledge gaps.
    """

    async def scan(self, target: str) -> List[KnowledgeGap]:
        """
        Scans for knowledge gaps related to the target.

        Args:
            target: The disease or biological entity to scan for.

        Returns:
            List[KnowledgeGap]: A list of KnowledgeGap objects.
        """
        ...


class BridgeBuilder(Protocol):
    """
    Interface for the Bridge Builder (Hypothesis Formulator).
    """

    async def generate_hypothesis(
        self, gap: KnowledgeGap, excluded_targets: Optional[List[str]] = None
    ) -> BridgeResult:
        """
        Generates a hypothesis bridging the knowledge gap.

        Args:
            gap: The KnowledgeGap to bridge.
            excluded_targets: Optional list of target symbols to exclude from consideration.

        Returns:
            BridgeResult: A BridgeResult containing the hypothesis (if found) and metadata.
        """
        ...


class CausalValidator(Protocol):
    """
    Interface for the Causal Validator (The Simulator).
    """

    async def validate(self, hypothesis: Hypothesis) -> Hypothesis:
        """
        Validates the hypothesis using causal simulation.

        Args:
            hypothesis: The hypothesis to validate.

        Returns:
            Hypothesis: The hypothesis updated with validation scores.
        """
        ...


class ProtocolDesigner(Protocol):
    """
    Interface for the Protocol Designer (The Experimentalist).
    """

    async def design_experiment(self, hypothesis: Hypothesis) -> Hypothesis:
        """
        Designs the killer experiment for the hypothesis.

        Args:
            hypothesis: The hypothesis to design an experiment for.

        Returns:
            Hypothesis: The hypothesis updated with PICO experimental design.
        """
        ...


class AdversarialReviewer(Protocol):
    """
    Interface for the Adversarial Reviewer (The Council).
    """

    async def review(self, hypothesis: Hypothesis) -> Hypothesis:
        """
        Conducts an adversarial review of the hypothesis.

        Args:
            hypothesis: The hypothesis to review.

        Returns:
            Hypothesis: The hypothesis updated with critiques from various strategies.
        """
        ...
