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
    """Implementation of the Protocol Designer (The Experimentalist)."""

    def design_experiment(self, hypothesis: Hypothesis) -> Hypothesis:
        """
        Designs the killer experiment for the hypothesis.
        Updates the hypothesis with PICO details.
        """
        logger.info(f"Designing experiment for hypothesis: {hypothesis.id}")

        target_symbol = hypothesis.target_candidate.symbol
        mechanism = hypothesis.proposed_mechanism

        # Construct PICO structure
        # Population: Derived from context (mechanism) or generic "Disease Model"
        population = f"In vitro/In vivo models relevant to {mechanism}"

        # Intervention: Targeting the candidate
        intervention = f"Selective inhibition/activation of {target_symbol}"

        # Comparator: Standard control
        comparator = "Vehicle control"

        # Outcome: Validating the mechanism
        outcome = f"Modulation of downstream biomarkers associated with {mechanism}"

        pico = {
            "population": population,
            "intervention": intervention,
            "comparator": comparator,
            "outcome": outcome,
            "duration": "72h (In vitro) / 14 days (In vivo)",  # Default heuristic
        }

        hypothesis.killer_experiment_pico = pico
        logger.info(f"Experiment designed: {pico}")

        return hypothesis
