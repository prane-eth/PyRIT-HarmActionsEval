# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json

import pytest

from pyrit.datasets.seed_datasets.local.harmactionseval_dataset import (
    HarmActionsEvalDataset,
)
from pyrit.models import SeedDataset, SeedObjective
from pyrit.models.harm_action import HarmAction


async def test_fetch_dataset_loads_harmful_and_unethical_rows_by_default(tmp_path):
    source = tmp_path / "harmactions_dataset.json"
    source.write_text(
        json.dumps(
            [
                {
                    "action_id": 1,
                    "prompt": "safe",
                    "manipulated_prompt": "safe",
                    "action": {
                        "type": "function",
                        "function": {"name": "weather", "arguments": {"city": "NYC"}},
                    },
                    "classification": "safe",
                    "risk_level": "none",
                    "flag_reason": "",
                },
                {
                    "action_id": 2,
                    "prompt": "harmful",
                    "manipulated_prompt": "h@rmful",
                    "action": {
                        "type": "function",
                        "function": {"name": "bad_tool", "arguments": {"target": "x"}},
                    },
                    "classification": "harmful",
                    "risk_level": "high",
                    "flag_reason": "bad",
                },
            ]
        ),
        encoding="utf-8",
    )

    dataset = await HarmActionsEvalDataset(source=source).fetch_dataset()

    assert isinstance(dataset, SeedDataset)
    assert dataset.dataset_name == "harmactionseval"
    assert len(dataset.seeds) == 1
    assert isinstance(dataset.seeds[0], SeedObjective)
    assert dataset.seeds[0].value == "h@rmful"
    assert dataset.seeds[0].metadata["action_id"] == 2
    assert (
        json.loads(dataset.seeds[0].metadata["action"])["function"]["name"]
        == "bad_tool"
    )


def test_dataset_name():
    loader = HarmActionsEvalDataset()
    assert loader.dataset_name == "harmactionseval"


def test_harm_action_tool_definition_infers_schema_types():
    action = HarmAction.from_dict(
        {
            "action_id": 1,
            "prompt": "p",
            "manipulated_prompt": "p",
            "action": {
                "type": "function",
                "function": {
                    "name": "mixed",
                    "arguments": {
                        "count": 1,
                        "enabled": True,
                        "ratio": 1.5,
                        "items": ["x"],
                    },
                },
            },
        }
    )

    tool = action.to_tool_definition()
    properties = tool["function"]["parameters"]["properties"]

    assert properties["count"]["type"] == "integer"
    assert properties["enabled"]["type"] == "boolean"
    assert properties["ratio"]["type"] == "number"
    assert properties["items"]["type"] == "array"


def test_harm_action_missing_function_raises():
    with pytest.raises(ValueError, match="name and arguments"):
        HarmAction.from_dict(
            {
                "action_id": 1,
                "prompt": "p",
                "manipulated_prompt": "p",
                "action": {"type": "function", "function": {"arguments": {}}},
            }
        )
