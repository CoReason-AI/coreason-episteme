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
def mock_client() -> AsyncMock:
    client = AsyncMock(spec=httpx.AsyncClient)
    client.post = AsyncMock()
    return client


@pytest.mark.asyncio
async def test_graph_nexus_client(mock_client: AsyncMock) -> None:
    client = HttpGraphNexusClient("http://base", mock_client)

    # find_disconnected_clusters
    response_clusters = Mock()
    response_clusters.json.return_value = [{"id": "c1"}]
    response_clusters.raise_for_status = Mock()
    mock_client.post.return_value = response_clusters

    res_clusters = await client.find_disconnected_clusters({})
    assert res_clusters == [{"id": "c1"}]

    # find_latent_bridges
    response_bridges = Mock()
    response_bridges.json.return_value = [
        {"symbol": "G", "ensembl_id": "E", "druggability_score": 0.5, "novelty_score": 0.5}
    ]
    response_bridges.raise_for_status = Mock()
    mock_client.post.return_value = response_bridges

    res_bridges = await client.find_latent_bridges("s", "t")
    assert isinstance(res_bridges[0], GeneticTarget)


@pytest.mark.asyncio
async def test_codex_client(mock_client: AsyncMock) -> None:
    client = HttpCodexClient("http://base", mock_client)

    # get_semantic_similarity
    response_sem = Mock()
    response_sem.json.return_value = 0.8
    response_sem.raise_for_status = Mock()
    mock_client.post.return_value = response_sem

    res_sem = await client.get_semantic_similarity("a", "b")
    assert res_sem == 0.8

    # validate_target
    response_target = Mock()
    response_target.status_code = 200
    response_target.json.return_value = {
        "symbol": "G",
        "ensembl_id": "E",
        "druggability_score": 0.5,
        "novelty_score": 0.5,
    }
    response_target.raise_for_status = Mock()
    mock_client.post.return_value = response_target

    res_target = await client.validate_target("G")
    assert isinstance(res_target, GeneticTarget)

    # validate_target 404
    response_404 = Mock()
    response_404.status_code = 404
    mock_client.post.return_value = response_404

    res_404 = await client.validate_target("Missing")
    assert res_404 is None

    # validate_target None
    response_none = Mock()
    response_none.status_code = 200
    response_none.json.return_value = None
    response_none.raise_for_status = Mock()
    mock_client.post.return_value = response_none

    res_none = await client.validate_target("None")
    assert res_none is None


@pytest.mark.asyncio
async def test_search_client(mock_client: AsyncMock) -> None:
    client = HttpSearchClient("http://base", mock_client)

    # find_literature_inconsistency
    response_gaps = Mock()
    response_gaps.json.return_value = [{"id": "k1", "description": "d", "type": "CLUSTER_DISCONNECT"}]
    response_gaps.raise_for_status = Mock()
    mock_client.post.return_value = response_gaps

    res_gaps = await client.find_literature_inconsistency("t")
    assert isinstance(res_gaps[0], KnowledgeGap)

    # verify_citation
    response_verify = Mock()
    response_verify.json.return_value = True
    response_verify.raise_for_status = Mock()
    mock_client.post.return_value = response_verify
    assert await client.verify_citation("c") is True

    # check_patent_infringement
    response_patent = Mock()
    response_patent.json.return_value = ["P1"]
    response_patent.raise_for_status = Mock()
    mock_client.post.return_value = response_patent

    gt = GeneticTarget(symbol="G", ensembl_id="E", druggability_score=0.5, novelty_score=0.5)
    res_patent = await client.check_patent_infringement(gt, "m")
    assert res_patent == ["P1"]

    # find_disconfirming_evidence
    response_evidence = Mock()
    response_evidence.json.return_value = ["E1"]
    response_evidence.raise_for_status = Mock()
    mock_client.post.return_value = response_evidence

    res_evidence = await client.find_disconfirming_evidence("s", "o", "a")
    assert res_evidence == ["E1"]


@pytest.mark.asyncio
async def test_prism_client(mock_client: AsyncMock) -> None:
    client = HttpPrismClient("http://base", mock_client)

    response = Mock()
    response.json.return_value = 0.9
    response.raise_for_status = Mock()
    mock_client.post.return_value = response

    assert await client.check_druggability("t") == 0.9


@pytest.mark.asyncio
async def test_inference_client(mock_client: AsyncMock) -> None:
    client = HttpInferenceClient("http://base", mock_client)

    # run_counterfactual_simulation
    response_sim = Mock()
    response_sim.json.return_value = 0.7
    response_sim.raise_for_status = Mock()
    mock_client.post.return_value = response_sim

    assert await client.run_counterfactual_simulation("m", "i") == 0.7

    # run_toxicology_screen
    response_tox = Mock()
    response_tox.json.return_value = ["Tox"]
    response_tox.raise_for_status = Mock()
    mock_client.post.return_value = response_tox

    gt = GeneticTarget(symbol="G", ensembl_id="E", druggability_score=0.5, novelty_score=0.5)
    assert await client.run_toxicology_screen(gt) == ["Tox"]

    # check_clinical_redundancy
    response_red = Mock()
    response_red.json.return_value = ["Red"]
    response_red.raise_for_status = Mock()
    mock_client.post.return_value = response_red

    assert await client.check_clinical_redundancy("m", gt) == ["Red"]


@pytest.mark.asyncio
async def test_veritas_client(mock_client: AsyncMock) -> None:
    client = HttpVeritasClient("http://base", mock_client)
    await client.log_trace("h1", {})
    mock_client.post.assert_called_once()
