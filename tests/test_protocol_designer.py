from coreason_episteme.components.protocol_designer import ProtocolDesignerImpl
from coreason_episteme.models import ConfidenceLevel, GeneticTarget, Hypothesis


def test_protocol_designer() -> None:
    designer = ProtocolDesignerImpl()

    target = GeneticTarget(
        symbol="GeneX",
        ensembl_id="ENSG000001",
        druggability_score=0.9,
        novelty_score=0.8,
    )
    hypothesis = Hypothesis(
        id="hyp1",
        title="Test",
        knowledge_gap="Gap",
        proposed_mechanism="Regulation of PathwayY via GeneX",
        target_candidate=target,
        causal_validation_score=0.8,
        key_counterfactual="",
        killer_experiment_pico={},
        evidence_chain=[],
        confidence=ConfidenceLevel.PLAUSIBLE,
    )

    updated_hypothesis = designer.design_experiment(hypothesis)

    pico = updated_hypothesis.killer_experiment_pico
    assert pico["Population"] is not None
    assert "GeneX" in pico["Population"]
    assert "GeneX" in pico["Intervention"]
    assert "Regulation of PathwayY via GeneX" in pico["Outcome"]
