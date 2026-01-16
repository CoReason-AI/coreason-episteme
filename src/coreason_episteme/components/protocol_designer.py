# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_episteme

from coreason_episteme.models import Hypothesis
from coreason_episteme.utils.logger import logger


class ProtocolDesignerImpl:
    """
    Implementation of the Protocol Designer (The Experimentalist).
    Designs the exact "Killer Experiment" (Wet Lab) required to prove or disprove the hypothesis.
    """

    def design_experiment(self, hypothesis: Hypothesis) -> Hypothesis:
        """
        Designs the killer experiment for the hypothesis.
        Updates the hypothesis with PICO details.

        PICO Structure:
        - Population: The model system (e.g., Cell line expressing Target).
        - Intervention: Inhibition of Target.
        - Comparator: Vehicle control or wild-type.
        - Outcome: Measurement of the downstream effect predicted by the mechanism.
        """
        logger.info(f"Designing experiment for hypothesis: {hypothesis.id}")

        target_symbol = hypothesis.target_candidate.symbol
        mechanism = hypothesis.proposed_mechanism

        # Simple heuristic to extract downstream effect from mechanism or gap
        # Assuming mechanism string format "Regulation of {target} via {candidate}..."
        # or we look at the knowledge gap.

        # Default construction of PICO
        pico = {
            "Population": f"Human cell line expressing {target_symbol} and relevant disease context",
            "Intervention": f"CRISPR-Cas9 knockout or small molecule inhibition of {target_symbol}",
            "Comparator": "Vehicle control (DMSO) or Scramble gRNA",
            "Outcome": f"Change in biomarkers associated with {mechanism}",
            "Notes": "Must monitor for off-target effects identified in safety review.",
        }

        logger.debug(f"Generated PICO: {pico}")

        hypothesis.killer_experiment_pico = pico

        return hypothesis
