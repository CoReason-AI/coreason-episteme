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

from types import TracebackType
from typing import Any, List, Optional, Type

import anyio
import httpx
from coreason_identity.models import UserContext

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
from coreason_episteme.engine import EpistemeEngineAsync
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


class EpistemeAsync:
    """
    Async-Native Episteme Service.

    The core service responsible for generating scientific hypotheses asynchronously.
    """

    def __init__(
        self,
        graph_client: GraphNexusClient,
        codex_client: CodexClient,
        search_client: SearchClient,
        prism_client: PrismClient,
        inference_client: InferenceClient,
        veritas_client: VeritasClient,
        client: Optional[httpx.AsyncClient] = None,
    ):
        self._internal_client = client is None
        self._client = client or httpx.AsyncClient()

        # Initialize Components
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

        self.engine = EpistemeEngineAsync(
            gap_scanner=gap_scanner,
            bridge_builder=bridge_builder,
            causal_validator=causal_validator,
            adversarial_reviewer=adversarial_reviewer,
            protocol_designer=protocol_designer,
            veritas_client=veritas_client,
            max_retries=settings.MAX_RETRIES,
        )

    async def __aenter__(self) -> "EpistemeAsync":
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        if self._internal_client:
            await self._client.aclose()
        # Ensure engine resources are cleaned up if any
        await self.engine.__aexit__(exc_type, exc_val, exc_tb)

    async def run(self, disease_id: str, *, context: UserContext) -> List[Hypothesis]:
        """
        Executes the hypothesis generation pipeline.
        """
        return await self.engine.run(disease_id, context=context)


class Episteme:
    """
    Sync Facade for Episteme Service.
    """

    def __init__(
        self,
        graph_client: GraphNexusClient,
        codex_client: CodexClient,
        search_client: SearchClient,
        prism_client: PrismClient,
        inference_client: InferenceClient,
        veritas_client: VeritasClient,
    ):
        self._async = EpistemeAsync(
            graph_client=graph_client,
            codex_client=codex_client,
            search_client=search_client,
            prism_client=prism_client,
            inference_client=inference_client,
            veritas_client=veritas_client,
        )

    def __enter__(self) -> "Episteme":
        return self

    def __exit__(self, *args: Any) -> None:
        anyio.run(self._async.__aexit__, *args)

    def run(self, disease_id: str, *, context: UserContext) -> List[Hypothesis]:
        """
        Executes the hypothesis generation pipeline (blocking).
        """

        async def _run() -> List[Hypothesis]:
            return await self._async.run(disease_id, context=context)

        return anyio.run(_run)


def generate_hypothesis(
    disease_id: str,
    graph_client: Optional[GraphNexusClient] = None,
    codex_client: Optional[CodexClient] = None,
    search_client: Optional[SearchClient] = None,
    prism_client: Optional[PrismClient] = None,
    inference_client: Optional[InferenceClient] = None,
    veritas_client: Optional[VeritasClient] = None,
    context: Optional[UserContext] = None,
) -> List[Hypothesis]:
    """
    Main entry point for generating scientific hypotheses for a given disease.
    Uses the Episteme Sync Facade.
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

    if context is None:
        context = UserContext(
                user_id="cli-user",
            sub="cli-user",
            email="cli@coreason.ai",
            permissions=["system"],
            project_context="cli",
        )

    with Episteme(
        graph_client=graph_client,
        codex_client=codex_client,
        search_client=search_client,
        prism_client=prism_client,
        inference_client=inference_client,
        veritas_client=veritas_client,
    ) as service:
        return service.run(disease_id, context=context)
