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
Main entry point for the coreason-episteme hypothesis engine.

This module provides the high-level API for generating scientific hypotheses,
orchestrating the injection of dependencies and the execution of the EpistemeEngine.
"""

from typing import List, Optional

from coreason_episteme.components.adversarial_reviewer import AdversarialReviewerImpl
from coreason_episteme.components.bridge_builder import BridgeBuilderImpl
from coreason_episteme.components.causal_validator import CausalValidatorImpl
from coreason_episteme.components.gap_scanner import GapScannerImpl
from coreason_episteme.components.protocol_designer import ProtocolDesignerImpl
from coreason_episteme.components.review_strategies import (
    ClinicalRedundancyStrategy,
    PatentStrategy,
    ScientificSkepticStrategy,
    ToxicologyStrategy,
)
from coreason_episteme.components.strategies import ReviewStrategy
from coreason_episteme.config import settings
from coreason_episteme.engine import EpistemeEngine
from coreason_episteme.interfaces import (
    CodexClient,
    GraphNexusClient,
    InferenceClient,
    PrismClient,
    SearchClient,
    VeritasClient,
)
from coreason_episteme.models import Hypothesis
from coreason_episteme.utils.logger import logger


def hello_world() -> str:
    """
    Returns a hello world string.

    Returns:
        str: The string "Hello World!".
    """
    logger.info("Hello World!")
    return "Hello World!"


def generate_hypothesis(
    disease_id: str,
    graph_client: Optional[GraphNexusClient] = None,
    codex_client: Optional[CodexClient] = None,
    search_client: Optional[SearchClient] = None,
    prism_client: Optional[PrismClient] = None,
    inference_client: Optional[InferenceClient] = None,
    veritas_client: Optional[VeritasClient] = None,
) -> List[Hypothesis]:
    """
    Main entry point for generating scientific hypotheses for a given disease.

    This function orchestrates the entire "Scan-Bridge-Simulate-Critique" loop by:
    1. Validating and injecting required external dependencies (clients).
    2. Initializing core components (GapScanner, BridgeBuilder, CausalValidator,
       AdversarialReviewer, ProtocolDesigner).
    3. Configuring the EpistemeEngine with these components.
    4. executing the engine's run loop.

    Args:
        disease_id: The identifier of the disease or condition to investigate (e.g., DOID:12345).
        graph_client: Client for GraphNexus (Traversals and Gap Analysis).
        codex_client: Client for Codex (Ontology and Similarity).
        search_client: Client for Search (Literature, Patents, and Verification).
        prism_client: Client for Prism (Druggability Assessment).
        inference_client: Client for Inference (Causal Simulation and Toxicity).
        veritas_client: Client for Veritas (Provenance Logging).

    Returns:
        List[Hypothesis]: A list of generated, validated, and critiqued Hypothesis objects.

    Raises:
        RuntimeError: If any of the required external clients are not provided.
    """
    logger.info(f"Request received: generate_hypothesis for {disease_id}")

    # Check dependencies and ensure they are not None
    if graph_client is None:
        raise RuntimeError("Missing required external client: GraphNexusClient")
    if codex_client is None:
        raise RuntimeError("Missing required external client: CodexClient")
    if search_client is None:
        raise RuntimeError("Missing required external client: SearchClient")
    if prism_client is None:
        raise RuntimeError("Missing required external client: PrismClient")
    if inference_client is None:
        raise RuntimeError("Missing required external client: InferenceClient")
    if veritas_client is None:
        raise RuntimeError("Missing required external client: VeritasClient")

    # Initialize Components (Dependency Injection)

    gap_scanner = GapScannerImpl(
        graph_client=graph_client,
        codex_client=codex_client,
        search_client=search_client,
        similarity_threshold=settings.GAP_SCANNER_SIMILARITY_THRESHOLD,
    )

    bridge_builder = BridgeBuilderImpl(
        graph_client=graph_client,
        prism_client=prism_client,
        codex_client=codex_client,
        search_client=search_client,
        druggability_threshold=settings.DRUGGABILITY_THRESHOLD,
    )

    causal_validator = CausalValidatorImpl(
        inference_client=inference_client,
    )

    # Initialize Strategies
    strategies: List[ReviewStrategy] = [
        ToxicologyStrategy(inference_client=inference_client),
        ClinicalRedundancyStrategy(inference_client=inference_client),
        PatentStrategy(search_client=search_client),
        ScientificSkepticStrategy(search_client=search_client),
    ]

    adversarial_reviewer = AdversarialReviewerImpl(
        strategies=strategies,
    )

    protocol_designer = ProtocolDesignerImpl()

    # Initialize Engine
    engine = EpistemeEngine(
        gap_scanner=gap_scanner,
        bridge_builder=bridge_builder,
        causal_validator=causal_validator,
        adversarial_reviewer=adversarial_reviewer,
        protocol_designer=protocol_designer,
        veritas_client=veritas_client,
        max_retries=settings.MAX_RETRIES,
    )

    # Run
    results: List[Hypothesis] = engine.run(disease_id)
    return results
