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
FastAPI Server for coreason-episteme (Theorist Service).
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, List, Optional

import httpx
from coreason_identity.models import UserContext
from fastapi import FastAPI
from pydantic import BaseModel

from coreason_episteme.config import settings
from coreason_episteme.engine import EpistemeEngineAsync
from coreason_episteme.external_clients import (
    HttpCodexClient,
    HttpGraphNexusClient,
    HttpInferenceClient,
    HttpPrismClient,
    HttpSearchClient,
    HttpVeritasClient,
)
from coreason_episteme.main import EpistemeAsync
from coreason_episteme.models import Hypothesis


class HypothesisRequest(BaseModel):
    """Request model for hypothesis generation."""

    disease_id: str
    max_retries: Optional[int] = None
    user_id: Optional[str] = "api-user"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager for the FastAPI application.
    Initializes external clients and the Episteme engine.
    """
    # Initialize shared HTTP client
    http_client = httpx.AsyncClient()

    # Initialize external clients
    graph_client = HttpGraphNexusClient(settings.GRAPH_NEXUS_URL, http_client)
    codex_client = HttpCodexClient(settings.CODEX_URL, http_client)
    search_client = HttpSearchClient(settings.SEARCH_URL, http_client)
    prism_client = HttpPrismClient(settings.PRISM_URL, http_client)
    inference_client = HttpInferenceClient(settings.INFERENCE_URL, http_client)
    veritas_client = HttpVeritasClient(settings.VERITAS_URL, http_client)

    # Initialize Engine
    # We pass the shared client to EpistemeAsync
    engine = EpistemeAsync(
        graph_client=graph_client,
        codex_client=codex_client,
        search_client=search_client,
        prism_client=prism_client,
        inference_client=inference_client,
        veritas_client=veritas_client,
        client=http_client,
    )

    # Store in state
    app.state.engine = engine
    app.state.http_client = http_client

    yield

    # Cleanup
    # EpistemeAsync.__aexit__ handles cleanup of the engine.
    # Since we passed the client, EpistemeAsync will NOT close it (it only closes if it created it).
    await engine.__aexit__(None, None, None)
    await http_client.aclose()


app = FastAPI(lifespan=lifespan, title="CoReason Episteme (Theorist)", version="0.3.0")


@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "version": "0.3.0"}


@app.post("/generate", response_model=List[Hypothesis])
async def generate_hypothesis_endpoint(request: HypothesisRequest) -> List[Hypothesis]:
    """
    Generate hypotheses for a given disease.
    """
    base_engine_async: EpistemeAsync = app.state.engine
    base_engine = base_engine_async.engine

    # Construct UserContext
    context = UserContext(
        user_id=request.user_id,
        sub=request.user_id,
        email=f"{request.user_id}@coreason.ai",
        permissions=["api"],
        project_context="api",
    )

    if request.max_retries is not None:
        # Create a temporary engine with overridden max_retries
        temp_engine = EpistemeEngineAsync(
            gap_scanner=base_engine.gap_scanner,
            bridge_builder=base_engine.bridge_builder,
            causal_validator=base_engine.causal_validator,
            adversarial_reviewer=base_engine.adversarial_reviewer,
            protocol_designer=base_engine.protocol_designer,
            veritas_client=base_engine.veritas_client,
            max_retries=request.max_retries,
        )
        return await temp_engine.run(request.disease_id, context=context)
    else:
        return await base_engine_async.run(request.disease_id, context=context)
