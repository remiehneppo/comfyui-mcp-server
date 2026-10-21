"""Configuration tools for ComfyUI MCP Server"""

from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP


def register_configuration_tools(
    mcp: FastMCP,
    comfyui_client,
    defaults_manager
):
    """Register configuration tools with the MCP server"""
    
    @mcp.tool()
    def list_models() -> dict:
        """List all available models in ComfyUI categorized by model type.
        
        Returns available diffusion models/UNets, text encoders/CLIP, VAE, checkpoints,
        upscalers, LoRAs, and ControlNet. Helps AI agents choose appropriate models
        and configure workflows.
        """
        categorized = comfyui_client.get_categorized_models() if hasattr(comfyui_client, "get_categorized_models") else {}
        all_models = comfyui_client.available_models
        default_model = defaults_manager.get_default("image", "model") if defaults_manager else (all_models[0] if all_models else None)

        return {
            "models": all_models,
            "count": len(all_models),
            "default": default_model,
            "diffusion_models": categorized.get("diffusion_models", []),
            "text_encoders": categorized.get("text_encoders", []),
            "vae": categorized.get("vae", []),
            "upscale_models": categorized.get("upscale_models", []),
            "checkpoints": categorized.get("checkpoints", []),
            "loras": categorized.get("loras", []),
            "controlnet": categorized.get("controlnet", []),
            "categorized": categorized
        }

    @mcp.tool()
    def get_defaults() -> dict:
        """Get current effective defaults for image, audio, and video generation.
        
        Returns merged defaults from all sources (runtime, config, env, hardcoded).
        Shows what values will be used when parameters are not explicitly provided.
        """
        return defaults_manager.get_all_defaults()

    @mcp.tool()
    def set_defaults(
        image: Optional[Dict[str, Any]] = None,
        audio: Optional[Dict[str, Any]] = None,
        video: Optional[Dict[str, Any]] = None,
        persist: bool = False
    ) -> dict:
        """Set runtime defaults for image, audio, and/or video generation.
        
        Args:
            image: Optional dict of default values for image generation (e.g., {"model": "sd_xl_base_1.0.safetensors", "width": 1024})
            audio: Optional dict of default values for audio generation (e.g., {"model": "ace_step_v1_3.5b.safetensors", "seconds": 30})
            video: Optional dict of default values for video generation (e.g., {"model": "wan2.2_vae.safetensors", "width": 1280, "duration": 5})
            persist: If True, write defaults to config file (~/.config/comfy-mcp/config.json). Otherwise, changes are ephemeral.
        
        Returns:
            Success status and any validation errors (e.g., invalid model names).
        """
        results = {}
        errors = []
        
        if image:
            result = defaults_manager.set_defaults("image", image, validate_models=True)
            if "error" in result or "errors" in result:
                errors.extend(result.get("errors", [result.get("error")]))
            else:
                results["image"] = result
                if persist:
                    persist_result = defaults_manager.persist_defaults("image", image)
                    if "error" in persist_result:
                        errors.append(f"Failed to persist image defaults: {persist_result['error']}")
        
        if audio:
            result = defaults_manager.set_defaults("audio", audio, validate_models=True)
            if "error" in result or "errors" in result:
                errors.extend(result.get("errors", [result.get("error")]))
            else:
                results["audio"] = result
                if persist:
                    persist_result = defaults_manager.persist_defaults("audio", audio)
                    if "error" in persist_result:
                        errors.append(f"Failed to persist audio defaults: {persist_result['error']}")
        
        if video:
            result = defaults_manager.set_defaults("video", video, validate_models=True)
            if "error" in result or "errors" in result:
                errors.extend(result.get("errors", [result.get("error")]))
            else:
                results["video"] = result
                if persist:
                    persist_result = defaults_manager.persist_defaults("video", video)
                    if "error" in persist_result:
                        errors.append(f"Failed to persist video defaults: {persist_result['error']}")
        
        if errors:
            return {"success": False, "errors": errors}
        
        return {"success": True, "updated": results}

    @mcp.tool()
    def free_vram() -> dict:
        """Unload models and free GPU memory (VRAM) in ComfyUI.
        
        Useful to reclaim GPU memory after generation, especially when
        switching between ComfyUI and local LLM/inference.
        """
        if hasattr(comfyui_client, "free_vram"):
            return comfyui_client.free_vram()
        return {"success": False, "error": "free_vram not supported on client"}
