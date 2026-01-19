# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_episteme

from typing import List, Optional

from coreason_episteme.components.adversarial_reviewer import AdversarialReviewerImpl
from coreason_episteme.components.bridge_builder import BridgeBuilderImpl
from coreason_episteme.components.causal_validator import CausalValidatorImpl
from coreason_episteme.components.gap_scanner import GapScannerImpl
from coreason_episteme.components.protocol_designer import ProtocolDesignerImpl
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
    Main entry point for generating hypotheses.

    Requires concrete implementations of all external clients.
    """
    logger.info(f"Request received: generate_hypothesis for {disease_id}")

    # Check dependencies
    if not all(
        [
            graph_client,
            codex_client,
            search_client,
            prism_client,
            inference_client,
            veritas_client,
        ]
    ):
        error_msg = (
            "Missing required external clients. "
            "Please provide implementations for: GraphNexus, Codex, Search, Prism, Inference, and Veritas."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # Initialize Components (Dependency Injection)
    # The type ignores are because mypy might complain about Optional vs Required if we didn't check
    # but we did check above with `if not all(...)`.
    # However, mypy doesn't always infer type narrowing from `all()`.
    # So we can assert or cast, but explicit is better.

    gap_scanner = GapScannerImpl(
        graph_client=graph_client,  # type: ignore
        codex_client=codex_client,  # type: ignore
        search_client=search_client,  # type: ignore
    )

    bridge_builder = BridgeBuilderImpl(
        graph_client=graph_client,  # type: ignore
        prism_client=prism_client,  # type: ignore
        codex_client=codex_client,  # type: ignore
        search_client=search_client,  # type: ignore
    )

    causal_validator = CausalValidatorImpl(
        inference_client=inference_client,  # type: ignore
    )

    adversarial_reviewer = AdversarialReviewerImpl(
        inference_client=inference_client,  # type: ignore
        search_client=search_client,  # type: ignore
    )

    protocol_designer = ProtocolDesignerImpl()

    # Initialize Engine
    engine = EpistemeEngine(
        gap_scanner=gap_scanner,
        bridge_builder=bridge_builder,
        causal_validator=causal_validator,
        adversarial_reviewer=adversarial_reviewer,
        protocol_designer=protocol_designer,
        veritas_client=veritas_client,  # type: ignore[arg-type]
    )

    # Run
    # Mypy cannot infer that engine.run returns List[Hypothesis] if engine is inferred as Any
    # Explicitly hinting result
    results: List[Hypothesis] = engine.run(disease_id)
    return results
