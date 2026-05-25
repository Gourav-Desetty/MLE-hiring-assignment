from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOOL_SPEC_PATH = REPO_ROOT / "data" / "api_specs" / "internal_tools.json"


class ToolValidationError(ValueError):
    """Raised when an action does not conform to internal_tools.json."""


def load_tool_specs(spec_path: Path | str = DEFAULT_TOOL_SPEC_PATH) -> dict[str, dict[str, Any]]:
    """Load internal tool specs keyed by tool name."""
    path = Path(spec_path)
    with path.open("r", encoding="utf-8") as handle:
        raw_specs = json.load(handle)

    if not isinstance(raw_specs, list):
        raise ToolValidationError("Tool spec must be a JSON array")

    specs: dict[str, dict[str, Any]] = {}
    for index, spec in enumerate(raw_specs):
        if not isinstance(spec, dict):
            raise ToolValidationError(f"Tool spec at index {index} must be an object")
        name = spec.get("name")
        parameters = spec.get("parameters")
        if not isinstance(name, str) or not name:
            raise ToolValidationError(f"Tool spec at index {index} is missing a valid name")
        if name in specs:
            raise ToolValidationError(f"Duplicate tool spec found for {name!r}")
        if not isinstance(parameters, dict):
            raise ToolValidationError(f"Tool {name!r} is missing a parameters schema")
        if parameters.get("type") != "object":
            raise ToolValidationError(f"Tool {name!r} parameters schema must be an object")
        if not isinstance(parameters.get("properties"), dict):
            raise ToolValidationError(f"Tool {name!r} parameters schema must define properties")
        specs[name] = spec
    return specs


@lru_cache(maxsize=4)
def _cached_tool_specs(spec_path: str) -> dict[str, dict[str, Any]]:
    return load_tool_specs(Path(spec_path))


def validate_actions_taken(
    actions: Any,
    spec_path: Path | str = DEFAULT_TOOL_SPEC_PATH,
) -> list[dict[str, Any]]:
    """Validate and normalize the actions_taken array.

    Expected action shape:
    {"action": "<tool_name>", "parameters": {...}}

    The allowed tool names, parameter names, required parameters, and primitive
    types are read from data/api_specs/internal_tools.json at runtime.
    """
    if actions is None:
        return []
    if not isinstance(actions, list):
        raise ToolValidationError("actions_taken must be a list")

    specs = _cached_tool_specs(str(Path(spec_path).resolve()))
    return [_validate_action(action, specs, index) for index, action in enumerate(actions)]


def actions_to_json(actions: Any, spec_path: Path | str = DEFAULT_TOOL_SPEC_PATH) -> str:
    """Return compact JSON after validating actions_taken."""
    validated = validate_actions_taken(actions, spec_path=spec_path)
    return json.dumps(validated, ensure_ascii=False, separators=(",", ":"))


def _validate_action(
    action: Any,
    specs: dict[str, dict[str, Any]],
    index: int,
) -> dict[str, Any]:
    if not isinstance(action, dict):
        raise ToolValidationError(f"actions_taken[{index}] must be an object")

    extra_action_keys = set(action) - {"action", "parameters"}
    if extra_action_keys:
        extras = ", ".join(sorted(extra_action_keys))
        raise ToolValidationError(f"actions_taken[{index}] has unexpected keys: {extras}")

    name = action.get("action")
    if not isinstance(name, str) or not name:
        raise ToolValidationError(f"actions_taken[{index}].action must be a non-empty string")
    if name not in specs:
        raise ToolValidationError(f"actions_taken[{index}].action {name!r} is not an allowed tool")

    parameters = action.get("parameters")
    if not isinstance(parameters, dict):
        raise ToolValidationError(f"actions_taken[{index}].parameters must be an object")

    schema = specs[name]["parameters"]
    validated_parameters = _validate_parameters(name, parameters, schema, index)
    return {"action": name, "parameters": validated_parameters}


def _validate_parameters(
    tool_name: str,
    parameters: dict[str, Any],
    schema: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    provided = set(parameters)

    missing = required - provided
    if missing:
        fields = ", ".join(sorted(missing))
        raise ToolValidationError(f"actions_taken[{index}] {tool_name!r} missing required parameters: {fields}")

    extra = provided - set(properties)
    if extra:
        fields = ", ".join(sorted(extra))
        raise ToolValidationError(f"actions_taken[{index}] {tool_name!r} has unexpected parameters: {fields}")

    validated: dict[str, Any] = {}
    for key, value in parameters.items():
        prop_schema = properties[key]
        expected_type = prop_schema.get("type")
        if not _matches_json_type(value, expected_type):
            actual_type = type(value).__name__
            raise ToolValidationError(
                f"actions_taken[{index}] {tool_name!r}.{key} must be {expected_type}, got {actual_type}"
            )

        allowed_values = _extract_allowed_values(prop_schema.get("description", ""))
        if allowed_values and isinstance(value, str) and value not in allowed_values:
            allowed = ", ".join(sorted(allowed_values))
            raise ToolValidationError(
                f"actions_taken[{index}] {tool_name!r}.{key} must be one of: {allowed}"
            )

        validated[key] = value

    return validated


def _matches_json_type(value: Any, expected_type: Any) -> bool:
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "number":
        return (isinstance(value, int | float)) and not isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    return True


def _extract_allowed_values(description: str) -> set[str]:
    """Infer string enums from descriptions like "priority: 'low', 'normal'".

    internal_tools.json does not use JSON Schema enum fields yet, but several
    parameter descriptions list closed values in single quotes. This keeps the
    source of truth in the JSON file while still enforcing those closed sets.
    """
    if not description or ":" not in description:
        return set()
    value_list = description.rsplit(":", maxsplit=1)[-1]
    return set(re.findall(r"'([^']+)'", value_list))
