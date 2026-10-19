"""Plain data types shared by the MCP modules."""
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel


@dataclass
class ToolResult:
    """What a tool handler returns. Everything in `data` must come from a database row or a database aggregate."""
    data: Any
    found: bool = True                                   # False = the database holds nothing matching the request
    refs: list[str] = field(default_factory=list)        # e.g. "invoice:12": the records the answer is built from
    tables: tuple[str, ...] = ()                         # which tables were read
    untrusted: tuple[str, ...] = ()                      # fields whose text came from outside (emails) -> data, never instructions
    notes: tuple[str, ...] = ()
    entity: str | None = None                            # recorded in the audit row, e.g. "invoice:12"


@dataclass(frozen=True)
class ToolContext:
    principal: Any                                       # app.auth.Principal
    mcp_role: str | None                                 # user | system | super_admin
    transport: str                                       # http | stdio
    allowed_tools: tuple[str, ...] = ()


_ANNOTATIONS = {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False}


def _strip_titles(node):
    """pydantic adds a 'title' to every property; it is noise for an LLM reading the schema."""
    if isinstance(node, dict):
        return {k: _strip_titles(v) for k, v in node.items() if not (k == "title" and isinstance(v, str))}
    if isinstance(node, list):
        return [_strip_titles(v) for v in node]
    return node


@dataclass(frozen=True)
class Tool:
    name: str
    title: str
    description: str
    args: type[BaseModel]
    handler: Callable[[ToolContext, Any], ToolResult]
    roles: frozenset[str]                                # application roles allowed to call it (see rbac.py)

    def input_schema(self) -> dict:
        return _strip_titles(self.args.model_json_schema())

    def descriptor(self) -> dict:
        return {"name": self.name, "title": self.title, "description": self.description,
                "inputSchema": self.input_schema(), "annotations": dict(_ANNOTATIONS)}
