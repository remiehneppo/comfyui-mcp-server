"""Workflow management tools for ComfyUI MCP Server"""

import logging
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP
from tools.helpers import register_and_build_response

logger = logging.getLogger("MCP_Server")


def _convert_litegraph_to_api(ui_graph: dict) -> dict:
    """Convert LiteGraph UI format (nodes, links) to ComfyUI API prompt format."""
    if not isinstance(ui_graph, dict) or "nodes" not in ui_graph:
        return ui_graph

    links_map = {}
    for link in ui_graph.get("links", []):
        if isinstance(link, (list, tuple)) and len(link) >= 3:
            link_id = link[0]
            origin_id = str(link[1])
            origin_slot = link[2]
            links_map[link_id] = [origin_id, origin_slot]

    api_prompt = {}
    for node in ui_graph.get("nodes", []):
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id"))
        node_type = node.get("type")
        if not node_type:
            continue

        inputs = {}
        widgets_values = list(node.get("widgets_values") or [])
        w_idx = 0

        for inp in node.get("inputs", []):
            if not isinstance(inp, dict):
                continue
            name = inp.get("name")
            link_id = inp.get("link")
            if link_id is not None and link_id in links_map:
                inputs[name] = links_map[link_id]
            elif inp.get("widget") is not None or "widget" in inp:
                if w_idx < len(widgets_values):
                    inputs[name] = widgets_values[w_idx]
                    w_idx += 1

        api_prompt[node_id] = {
            "class_type": node_type,
            "inputs": inputs
        }
    return api_prompt


def register_workflow_tools(
    mcp: FastMCP,
    workflow_manager,
    comfyui_client,
    defaults_manager,
    asset_registry
):
    """Register workflow tools with the MCP server"""
    
    @mcp.tool()
    def list_workflows() -> dict:
        """List all available workflows in the workflow directory.
        
        Returns a catalog of workflows with their IDs, names, descriptions,
        available inputs, and optional metadata.
        """
        catalog = workflow_manager.get_workflow_catalog()
        return {
            "workflows": catalog,
            "count": len(catalog),
            "workflow_dir": str(workflow_manager.workflows_dir)
        }

    @mcp.tool()
    def run_workflow(
        workflow_id: str,
        overrides: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
        return_inline_preview: bool = False
    ) -> dict:
        """Run a saved ComfyUI workflow with constrained parameter overrides.
        
        Args:
            workflow_id: The workflow ID (filename stem, e.g., "generate_image")
            overrides: Optional dict of parameter overrides (e.g., {"prompt": "a cat", "width": 1024})
            options: Optional dict of execution options (reserved for future use)
            return_inline_preview: If True, include a small thumbnail base64 in response (256px, ~100KB)
        
        Returns:
            Result with asset_url, workflow_id, and execution metadata. If return_inline_preview=True,
            also includes inline_preview_base64 for immediate viewing.
        """
        if overrides is None:
            overrides = {}
        
        # Load workflow
        workflow = workflow_manager.load_workflow(workflow_id)
        if not workflow:
            return {"error": f"Workflow '{workflow_id}' not found"}
        
        try:
            # Apply overrides with constraints
            workflow = workflow_manager.apply_workflow_overrides(
                workflow, workflow_id, overrides, defaults_manager
            )

            # Extract and remove override report before submitting to ComfyUI
            override_report = workflow.pop("__override_report__", None)

            # Determine output preferences
            output_preferences = workflow_manager._guess_output_preferences(workflow)

            # Execute workflow
            result = comfyui_client.run_custom_workflow(
                workflow,
                preferred_output_keys=output_preferences,
            )

            # Register asset and build response
            response = register_and_build_response(
                result,
                workflow_id,
                asset_registry,
                tool_name=None,
                return_inline_preview=return_inline_preview,
                session_id=None
            )

            # Include override report so the agent can see what was applied/dropped
            if override_report and override_report.get("overrides_dropped"):
                response["overrides_applied"] = override_report["overrides_applied"]
                response["overrides_dropped"] = override_report["overrides_dropped"]

            return response
        except Exception as exc:
            logger.exception("Workflow '%s' failed", workflow_id)
            return {"error": str(exc)}

    @mcp.tool()
    def execute_custom_workflow(
        workflow: Any,
        return_inline_preview: bool = False
    ) -> dict:
        """Execute any custom ComfyUI workflow JSON graph directly without needing a predefined file.
        
        Args:
            workflow: ComfyUI workflow prompt graph as a dictionary or JSON string.
                      Format: {"node_id": {"class_type": "...", "inputs": {...}}}
            return_inline_preview: If True, include a small thumbnail base64 in response
            
        Returns:
            Result with asset_url, filename, and execution metadata.
        """
        import json
        if isinstance(workflow, str):
            try:
                workflow = json.loads(workflow)
            except Exception as e:
                return {"error": f"Invalid JSON string: {e}"}

        if not isinstance(workflow, dict):
            return {"error": "Workflow must be a dictionary or a JSON string representing a dict."}

        # If wrapped like {"prompt": {...}}
        if "prompt" in workflow and isinstance(workflow["prompt"], dict):
            workflow = workflow["prompt"]

        # Auto-convert if LiteGraph UI format (nodes + links)
        if "nodes" in workflow and "links" in workflow:
            logger.info("Auto-converting LiteGraph UI workflow to ComfyUI API prompt format")
            workflow = _convert_litegraph_to_api(workflow)

        try:
            output_preferences = ("images", "image", "gifs", "gif", "audio", "audios", "files")
            result = comfyui_client.run_custom_workflow(
                workflow,
                preferred_output_keys=output_preferences,
            )

            response = register_and_build_response(
                result,
                "custom_workflow",
                asset_registry,
                tool_name="execute_custom_workflow",
                return_inline_preview=return_inline_preview,
                session_id=None
            )
            return response
        except Exception as exc:
            logger.exception("Custom workflow execution failed")
            return {"error": str(exc)}
