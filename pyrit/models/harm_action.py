# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from pyrit.models.strategy_result import StrategyResult


@dataclass(frozen=True)
class HarmAction:
    """One HarmActionsEval action row."""

    action_id: int
    prompt: str
    manipulated_prompt: str
    action: dict[str, Any]
    classification: Optional[str] = None
    risk_level: Optional[str] = None
    flag_reason: str = ""

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> HarmAction:
        """
        Create a HarmAction from a dataset row.

        Args:
            row: HarmActionsEval row.

        Returns:
            Parsed HarmAction.

        Raises:
            ValueError: If required fields or function action fields are missing.
        """
        required_keys = {"action_id", "prompt", "manipulated_prompt", "action"}
        missing_keys = required_keys - row.keys()
        if missing_keys:
            raise ValueError(
                f"Missing keys in HarmActionsEval row: {', '.join(sorted(missing_keys))}"
            )

        action = row["action"]
        if not isinstance(action, dict) or action.get("type") != "function":
            raise ValueError("HarmActionsEval action must be a function action.")
        function = action.get("function")
        if (
            not isinstance(function, dict)
            or not function.get("name")
            or "arguments" not in function
        ):
            raise ValueError(
                "HarmActionsEval function action must include a name and arguments."
            )
        if not isinstance(function["arguments"], dict):
            raise ValueError("HarmActionsEval function arguments must be a dictionary.")

        return cls(
            action_id=int(row["action_id"]),
            prompt=str(row["prompt"]),
            manipulated_prompt=str(row["manipulated_prompt"]),
            action=action,
            classification=row.get("classification"),
            risk_level=row.get("risk_level"),
            flag_reason=row.get("flag_reason", ""),
        )

    @property
    def tool_name(self) -> str:
        """Return the function tool name for this action."""
        return str(self.action["function"]["name"])

    def to_tool_definition(self) -> dict[str, Any]:
        """
        Create an OpenAI-compatible function tool definition for this action.

        Returns:
            Function tool definition.
        """
        properties: dict[str, dict[str, Any]] = {}
        required: list[str] = []

        for key, value in self.action["function"]["arguments"].items():
            properties[key] = _json_schema_for_value(
                value, description=f"Parameter {key}"
            )
            required.append(key)

        return {
            "type": "function",
            "function": {
                "name": self.tool_name,
                "description": f"Tool for {self.tool_name}",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def to_seed_metadata(self) -> dict[str, str | int]:
        """
        Serialize action details into metadata fields accepted by Seed objects.

        Returns:
            Seed metadata containing action details.
        """
        return {
            "action_id": self.action_id,
            "prompt": self.prompt,
            "action": json.dumps(self.action, separators=(",", ":")),
            "classification": self.classification or "",
            "risk_level": self.risk_level or "",
            "flag_reason": self.flag_reason,
        }

    @classmethod
    def from_seed_metadata(cls, *, value: str, metadata: dict[str, Any]) -> HarmAction:
        """
        Rehydrate a HarmAction from a SeedObjective value and metadata.

        Args:
            value: Seed value to use as the manipulated prompt.
            metadata: Seed metadata created by ``to_seed_metadata``.

        Returns:
            Parsed HarmAction.
        """
        return cls.from_dict(
            {
                "action_id": metadata["action_id"],
                "prompt": metadata.get("prompt", value),
                "manipulated_prompt": value,
                "action": json.loads(metadata["action"]),
                "classification": metadata.get("classification") or None,
                "risk_level": metadata.get("risk_level") or None,
                "flag_reason": metadata.get("flag_reason", ""),
            }
        )


@dataclass(frozen=True)
class HarmActionEvalEntryResult:
    """Evaluation result for one HarmActionsEval action."""

    action_id: int
    tool_name: str
    predicted_harmful: bool
    harm_at_attempt: Optional[int]
    response_tool_calls: list[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class HarmActionsEvalResult(StrategyResult):
    """Aggregate HarmActionsEval result."""

    total_actions: int
    harmful_action_count: int
    attack_success_rate: float
    pass_rate: float
    max_attempts: int
    action_results: list[HarmActionEvalEntryResult]
    sample_action_result: HarmActionEvalEntryResult


def _json_schema_for_value(value: Any, *, description: str) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"type": "boolean", "description": description}
    if isinstance(value, int):
        return {"type": "integer", "description": description}
    if isinstance(value, float):
        return {"type": "number", "description": description}
    if isinstance(value, list):
        item_schema = {"type": "string"}
        if value:
            item_schema = {
                k: v
                for k, v in _json_schema_for_value(
                    value[0], description=description
                ).items()
                if k != "description"
            }
        return {"type": "array", "items": item_schema, "description": description}
    if isinstance(value, dict):
        return {"type": "object", "description": description}
    return {"type": "string", "description": description}
