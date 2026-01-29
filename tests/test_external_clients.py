# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_episteme

from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from coreason_episteme.external_clients import (
    HttpCodexClient,
    HttpGraphNexusClient,
    HttpInferenceClient,
    HttpPrismClient,
    HttpSearchClient,
    HttpVeritasClient,
)
from coreason_episteme.models import GeneticTarget, KnowledgeGap


@pytest.fixture
def mock_client():
    client = AsyncMock(spec=httpx.AsyncClient)
    client.post = AsyncMock()
    return client


@pytest.mark.asyncio
async def test_graph_nexus_client(mock_client):
    client = HttpGraphNexusClient("http://base", mock_client)

    # find_disconnected_clusters
    response = Mock()
    response.json.return_value = [{"id": "c1"}]
    response.raise_for_status = Mock()
    mock_client.post.return_value = response

    res = await client.find_disconnected_clusters({})
    assert res == [{"id": "c1"}]

    # find_latent_bridges
    response.json.return_value = [
        {"symbol": "G", "ensembl_id": "E", "druggability_score": 0.5, "novelty_score": 0.5}
    ]
    res = await client.find_latent_bridges("s", "t")
    assert isinstance(res[0], GeneticTarget)


@pytest.mark.asyncio
async def test_codex_client(mock_client):
    client = HttpCodexClient("http://base", mock_client)

    # get_semantic_similarity
    response = Mock()
    response.json.return_value = 0.8
    response.raise_for_status = Mock()
    mock_client.post.return_value = response

    res = await client.get_semantic_similarity("a", "b")
    assert res == 0.8

    # validate_target
    response.status_code = 200
    response.json.return_value = {
        "symbol": "G",
        "ensembl_id": "E",
        "druggability_score": 0.5,
        "novelty_score": 0.5,
    }
    res = await client.validate_target("G")
    assert isinstance(res, GeneticTarget)

    # validate_target 404
    response_404 = Mock()
    response_404.status_code = 404
    mock_client.post.return_value = response_404

    res = await client.validate_target("Missing")
    assert res is None

    # validate_target None
    response.status_code = 200
    response.json.return_value = None
    mock_client.post.return_value = response

    res = await client.validate_target("None")
    assert res is None


@pytest.mark.asyncio
async def test_search_client(mock_client):
    client = HttpSearchClient("http://base", mock_client)

    # find_literature_inconsistency
    response = Mock()
    response.json.return_value = [
        {"id": "k1", "description": "d", "type": "CLUSTER_DISCONNECT"}
    ]
    response.raise_for_status = Mock()
    mock_client.post.return_value = response

    res = await client.find_literature_inconsistency("t")
    assert isinstance(res[0], KnowledgeGap)

    # verify_citation
    response.json.return_value = True
    assert await client.verify_citation("c") is True

    # check_patent_infringement
    response.json.return_value = ["P1"]
    gt = GeneticTarget(symbol="G", ensembl_id="E", druggability_score=0.5, novelty_score=0.5)
    res = await client.check_patent_infringement(gt, "m")
    assert res == ["P1"]

    # find_disconfirming_evidence
    response.json.return_value = ["E1"]
    res = await client.find_disconfirming_evidence("s", "o", "a")
    assert res == ["E1"]


@pytest.mark.asyncio
async def test_prism_client(mock_client):
    client = HttpPrismClient("http://base", mock_client)

    response = Mock()
    response.json.return_value = 0.9
    response.raise_for_status = Mock()
    mock_client.post.return_value = response

    assert await client.check_druggability("t") == 0.9


@pytest.mark.asyncio
async def test_inference_client(mock_client):
    client = HttpInferenceClient("http://base", mock_client)

    # run_counterfactual_simulation
    response = Mock()
    response.json.return_value = 0.7
    response.raise_for_status = Mock()
    mock_client.post.return_value = response

    assert await client.run_counterfactual_simulation("m", "i") == 0.7

    # run_toxicology_screen
    response.json.return_value = ["Tox"]
    gt = GeneticTarget(symbol="G", ensembl_id="E", druggability_score=0.5, novelty_score=0.5)
    assert await client.run_toxicology_screen(gt) == ["Tox"]

    # check_clinical_redundancy
    response.json.return_value = ["Red"]
    assert await client.check_clinical_redundancy("m", gt) == ["Red"]


@pytest.mark.asyncio
async def test_veritas_client(mock_client):
    client = HttpVeritasClient("http://base", mock_client)
    await client.log_trace("h1", {})
    mock_client.post.assert_called_once()
