# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Acknowledgement: Praneeth Vadlapati - https://github.com/Pro-GenAI/Agent-Action-Guard

from __future__ import annotations

import json
from pathlib import Path

from pyrit.datasets.seed_datasets.seed_dataset_provider import SeedDatasetProvider
from pyrit.datasets.seed_datasets.seed_metadata import SeedDatasetLoadTime
from pyrit.models import SeedDataset, SeedObjective
from pyrit.models.harm_action import HarmAction


class HarmActionsEvalDataset(SeedDatasetProvider):
    """
    Loader for the HarmActionsEval dataset.

    HarmActionsEval evaluates whether an agentic model selects harmful or unethical
    tool calls rather than only evaluating final text responses.

    Reference: https://github.com/Pro-GenAI/Agent-Action-Guard
    """

    harm_categories: list[str] = [
        "agent_action_safety",
        "harmful_tool_use",
        "unethical_tool_use",
    ]
    modalities: list[str] = ["text"]
    size: str = "medium"  # 141 harmful/unethical seeds by default
    tags: set[str] = {"safety", "agent", "tools"}
    load_time = SeedDatasetLoadTime.FAST

    def __init__(self, *, source: Path | None = None) -> None:
        self._source = source or Path(__file__).with_name("harmactionseval").joinpath(
            "harmactions_dataset.json"
        )

    @property
    def dataset_name(self) -> str:
        """Return the dataset name."""
        return "harmactionseval"

    async def fetch_dataset(self, *, cache: bool = True) -> SeedDataset:
        """
        Load HarmActionsEval rows and return harmful/unethical rows as SeedObjectives.

        Args:
            cache: Unused for this packaged dataset. Kept for provider interface compatibility.

        Returns:
            SeedDataset containing HarmActionsEval manipulated prompts.
        """
        del cache
        rows = json.loads(self._source.read_text(encoding="utf-8"))
        actions = [
            HarmAction.from_dict(row)
            for row in rows
            if row.get("classification") != "safe"
        ]

        seeds = [
            SeedObjective(
                value=action.manipulated_prompt,
                name="HarmActionsEval",
                dataset_name=self.dataset_name,
                harm_categories=[action.classification or "unknown"],
                description=(
                    "HarmActionsEval prompt for evaluating whether an "
                    "agent attempts the associated tool action."
                ),
                source="https://github.com/Pro-GenAI/Agent-Action-Guard",
                authors=["Pro-GenAI"],
                metadata=action.to_seed_metadata(),
            )
            for action in actions
        ]

        return SeedDataset(seeds=seeds, dataset_name=self.dataset_name)
