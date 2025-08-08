import os
import yaml
from typing import Dict, Any, Optional


class AssistantConfig:
    """Simple configuration class for the autonomous AI assistant"""
    
    def __init__(self, config_file: str = "settings.yaml"):
        # Initialize with empty config
        self.raw_config = {}
        self.settings = {}
        
        # Load YAML configuration
        config_path = self._find_config_file(config_file)
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    self.raw_config = yaml.safe_load(f) or {}
                print(f"✅ Configuration loaded from {config_path}")
            except Exception as e:
                print(f"⚠️ Error loading config file: {e}")
                self.raw_config = {}
        else:
            print(f"⚠️ Config file {config_file} not found, using defaults")
            self.raw_config = {}
        
        # Set up defaults and update from YAML
        self._setup_defaults()
        self._update_from_yaml()
    
    def _find_config_file(self, config_file: str) -> Optional[str]:
        """Find configuration file in various locations"""
        # Get backend config directory (preferred location)
        backend_config = os.path.join(os.path.dirname(__file__), "..", "config")
        backend_config = os.path.abspath(backend_config)
        
        # Get project root (for fallback)
        project_root = os.path.join(os.path.dirname(__file__), "..", "..")
        project_root = os.path.abspath(project_root)
        
        possible_paths = [
            # Primary: backend/config directory (user's preferred location)
            os.path.join(backend_config, config_file),
            
            # Secondary: project root (for compatibility)
            os.path.join(project_root, config_file),
            
            # Fallbacks: other locations
            config_file,  # Current directory
            os.path.join(os.getcwd(), config_file),
            os.path.join(os.path.expanduser("~"), config_file)
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None
    
    def _setup_defaults(self):
        """Set up minimal default configuration values (most come from settings.yaml)"""
        # Only essential defaults that don't duplicate settings.yaml
        self.settings = {}
        self.model_routes = {}
        
    
    def _update_from_yaml(self):
        """Update configuration from YAML data"""
        if not self.raw_config:
            return
            
        # Update model routing from your existing LLM structure
        if "llm" in self.raw_config:
            llm_config = self.raw_config["llm"]
            
            # Support AutoGen LLM configuration
            if "autogen" in llm_config:
                self.settings["autogen"] = llm_config["autogen"]
            
            # Preserve existing model routing
            if "model_routes" in llm_config:
                self.model_routes.update(llm_config["model_routes"])
            
            # Build model routes from your provider structure
            self.model_routes = {}
            
            # Map your providers to model routes
            for provider in ["ollama", "openrouter", "gemini", "openai", "anthropic"]:
                if provider in llm_config:
                    provider_config = llm_config[provider]
                    if "models" in provider_config:
                        for tier, model in provider_config["models"].items():
                            self.model_routes[f"{provider}_{tier}"] = f"{provider}/{model}"
        
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot notation (e.g., 'llm.ollama.base_url')"""
        keys = key.split('.')
        value = self.raw_config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a specific provider from the config file."""
        return self.get(f"providers.{provider}.api_key")
    
    def get_default_user_name(self) -> str:
        """Get the default user name from user profile configuration"""
        user_name = self.get("user.name")
        if user_name is None:
            raise ValueError("user.name not found in configuration")
        return user_name
    
    def get_user_profile(self) -> Dict[str, Any]:
        """Get user profile configuration"""
        user_profile = self.get("user")
        if user_profile is None:
            raise ValueError("user profile not found in configuration")
        return user_profile
    
    def get_assistant_profile(self) -> Dict[str, Any]:
        """Get assistant profile configuration"""
        assistant_profile = self.get("assistant")
        if assistant_profile is None:
            raise ValueError("assistant profile not found in configuration")
        return assistant_profile
    
    
    
    # Properties for easy access to common config sections
    @property
    def models(self):
        return self.settings.get("models", {})
    
    @property
    def memory(self):
        return self.settings.get("memory", {})
    
    # =========================================
    # UNIFIED MEMORY CONTROL METHODS
    # =========================================
    
    def get_memory_ttl(self, memory_type: str, importance_level: str = "default") -> int:
        """Get TTL for specific memory type and importance level"""
        if memory_type == "short_term":
            return self.get(f"memory_control.ttl.short_term.{importance_level}", 86400)
        elif memory_type == "working_memory":
            return self.get("memory_control.ttl.working_memory", 604800)
        elif memory_type == "session":
            return self.get("memory_control.ttl.session", 0)
        elif memory_type == "activity_extension":
            return self.get("memory_control.ttl.activity_extension", 86400)
        else:
            return 86400  # 24 hours default
    
    def get_memory_limit(self, memory_type: str) -> int:
        """Get item limit for specific memory type"""
        if memory_type == "working_memory":
            return self.get("memory_control.limits.working_memory_items", 7)
        elif memory_type == "short_term":
            return self.get("memory_control.limits.short_term_items", 100)
        elif memory_type == "session":
            return self.get("memory_control.limits.session_conversations", 50)
        elif memory_type == "search_results":
            return self.get("memory_control.limits.search_results_limit", 20)
        else:
            return 10  # Default limit
    
    def get_short_term_ttl_values(self) -> Dict[str, int]:
        """Get all short-term memory TTL values for memory writer agent"""
        return {
            "minimal": self.get("memory_control.ttl.short_term.minimal", 86400),
            "short": self.get("memory_control.ttl.short_term.short", 604800),
            "medium": self.get("memory_control.ttl.short_term.medium", 2592000),
            "extended": self.get("memory_control.ttl.short_term.extended", 7776000),
            "default": self.get("memory_control.ttl.short_term.default", 86400)
        }
    
    
    @property
    def features(self):
        return self.settings.get("features", {})
    
    def get_token_limit(self, operation_type: str) -> int:
        """Get token limit for Organizer Agent (only agent using external LLMs)"""
        token_limits = self.get("token_limits", {})
        
        # Only Organizer Agent uses external LLMs, others use local transformers
        if operation_type in ["organizer_agent", "organizer", "reasoning"]:
            return token_limits.get("default", 4096)
        
        # For local transformers (Memory Reader, Memory Writer, Knowledge), no token limits needed
        return 512  # Small limit for local processing
    
    def get_safe_token_limit(self, operation_type: str, requested_tokens: int = None) -> int:
        """Get safe token limit with bounds checking - only for Organizer Agent"""
        token_limits = self.get("token_limits", {})
        
        # Get base limit for operation
        base_limit = self.get_token_limit(operation_type)
        
        # Use requested if provided, otherwise use base
        target_limit = requested_tokens if requested_tokens else base_limit
        
        # Apply safety bounds (only for Organizer Agent external LLM usage)
        if operation_type in ["organizer_agent", "organizer", "reasoning"]:
            max_limit = token_limits.get("absolute_maximum", 32768)
            min_limit = token_limits.get("minimum_safe", 100)
            return max(min_limit, min(target_limit, max_limit))
        
        # For local transformers, return small fixed limit
        return 512
    
    def get_model_for_category(self, category: str) -> str:
        """Get model for specific category - only Organizer Agent uses external LLMs"""
        # Only organizer_agent uses external LLMs in the new 4-agent architecture
        valid_categories = ["organizer_agent", "organizer", "reasoning"]
        
        if category not in valid_categories:
            raise ValueError(f"Invalid category '{category}'. Only Organizer Agent uses external LLMs. Other agents use local transformers.")
        
        # Get the external model list for Organizer Agent
        model_list = self.get("organizer_external_models")
        if not model_list or not isinstance(model_list, list) or len(model_list) == 0:
            raise ValueError(f"Missing configuration: organizer_external_models not found or empty in settings.yaml. Please configure organizer_external_models section.")
        
        # Return the first model in the list (primary model)
        return model_list[0]
    
    async def call_llm_with_fallback(self, category: str, messages: list, **kwargs) -> Any:
        """Call LLM with automatic fallback to next model in category if primary fails"""
        import litellm
        
        # Get all models for this category
        model_list = self.get_model_list_for_category(category)
        
        last_error = None
        for i, model in enumerate(model_list):
            try:
                if self.get("development.debug_mode", False) and i > 0:
                    print(f"🔄 Trying fallback model {i+1}/{len(model_list)}: {model}")
                
                # Set default parameters
                # For OpenAI (default provider), strip prefix. For others, we'll set the model name in provider-specific logic
                actual_model = model
                if "openai/" in model.lower():
                    actual_model = model.replace("openai/", "")
                
                call_params = {
                    "model": actual_model,
                    "messages": messages,
                    "max_tokens": kwargs.get("max_tokens", 1000),
                    "temperature": kwargs.get("temperature", 0.7),
                    "timeout": kwargs.get("timeout", 30)
                }
                
                # Add API key and configure provider-specific settings
                if "groq/" in model.lower():
                    api_key = self.get("providers.groq.api_key")
                    if api_key:
                        call_params["api_key"] = api_key
                        # LiteLLM format for Groq: groq/model_name
                        call_params["model"] = model  # Keep the groq/ prefix
                        actual_model = model  # Override the stripped version
                elif "anthropic/" in model.lower() or "claude" in model.lower():
                    api_key = self.get("providers.anthropic.api_key")
                    if api_key:
                        call_params["api_key"] = api_key
                        # LiteLLM format for Anthropic: claude-3-haiku-20240307 or keep anthropic/ prefix
                        call_params["model"] = model  # Keep the anthropic/ prefix
                        actual_model = model
                elif "openai/" in model.lower() or "gpt" in model.lower():
                    api_key = self.get("providers.openai.api_key")
                    if api_key:
                        call_params["api_key"] = api_key
                        # For OpenAI, we can strip the prefix as it's the default
                        # call_params["model"] stays as actual_model
                elif "gemini/" in model.lower() or "google" in model.lower():
                    api_key = self.get("providers.gemini.api_key")
                    if api_key:
                        call_params["api_key"] = api_key
                        # LiteLLM format for Gemini: gemini/gemini-1.5-flash
                        call_params["model"] = model  # Keep the gemini/ prefix
                        actual_model = model
                
                call_params.update(kwargs)
                
                response = await litellm.acompletion(**call_params)
                
                if self.get("development.debug_mode", False) and i > 0:
                    print(f"✅ Fallback successful with model: {model}")
                
                return response
                
            except Exception as e:
                last_error = e
                if self.get("development.debug_mode", False):
                    print(f"❌ Model {model} failed: {str(e)[:100]}...")
                
                # Continue to next model if available
                if i < len(model_list) - 1:
                    continue
                else:
                    # All models failed
                    break
        
        # If we get here, all models failed
        raise Exception(f"All models in category '{category}' failed. Last error: {last_error}")
        
    def get_model_list_for_category(self, category: str) -> list:
        """Get full model list for category - only valid for Organizer Agent"""
        valid_categories = ["organizer_agent", "organizer", "reasoning"]
        
        if category not in valid_categories:
            raise ValueError(f"Invalid category '{category}'. Only Organizer Agent uses external LLMs. Other agents use local transformers.")
        
        # Get the external model list for Organizer Agent
        model_list = self.get("organizer_external_models")
        if not model_list or not isinstance(model_list, list) or len(model_list) == 0:
            raise ValueError(f"Missing configuration: organizer_external_models not found or empty in settings.yaml. Please configure organizer_external_models section.")
            
        return model_list
    
    def get_llm_config(self, function_type: str = "organizer_agent") -> Dict[str, Any]:
        """
        Get LLM configuration - only for Organizer Agent (external LLM usage)
        """
        try:
            # Only Organizer Agent uses external LLMs
            if function_type not in ["organizer_agent", "organizer", "reasoning", "chat"]:
                raise ValueError(f"LLM configuration only available for Organizer Agent. '{function_type}' uses local transformers.")
            
            # Get model for the function type
            model = self.get_model_for_category("organizer_agent")
            
            # Base configuration
            llm_config = {
                "model": model,
                "temperature": self.get("model_settings.temperature", 0.7),
                "max_tokens": self.get_safe_token_limit("organizer_agent"),
                "timeout": self.get("model_settings.timeout", 30)
            }
            
            # Add API keys based on model provider (using existing settings.yaml structure)
            if "openai" in model.lower() or "gpt" in model.lower():
                api_key = self.get("providers.openai.api_key") or os.getenv("OPENAI_API_KEY")
                if api_key:
                    llm_config["api_key"] = api_key
                    
            elif "claude" in model.lower() or "anthropic" in model.lower():
                api_key = self.get("providers.anthropic.api_key") or os.getenv("ANTHROPIC_API_KEY")
                if api_key:
                    llm_config["api_key"] = api_key
                    
            elif "gemini" in model.lower() or "google" in model.lower():
                api_key = self.get("providers.gemini.api_key") or os.getenv("GOOGLE_API_KEY")
                if api_key:
                    llm_config["api_key"] = api_key
                    
            elif "groq" in model.lower():
                api_key = self.get("providers.groq.api_key") or os.getenv("GROQ_API_KEY")
                if api_key:
                    llm_config["api_key"] = api_key
                    
            elif "openrouter" in model.lower():
                api_key = self.get("providers.openrouter.api_key") or os.getenv("OPENROUTER_API_KEY")
                base_url = self.get("providers.openrouter.base_url")
                if api_key:
                    llm_config["api_key"] = api_key
                if base_url:
                    llm_config["base_url"] = base_url
                    
            # Don't merge entire autogen section - it contains agent configs too
            # Only merge specific autogen llm_config if needed
            
            return llm_config
            
        except Exception as e:
            # Fallback configuration for Organizer Agent
            return {
                "model": "anthropic/claude-3-sonnet-20240229",  # Use first model from organizer_external_models
                "temperature": 0.7,
                "max_tokens": 4096,
                "timeout": 30
            }
    
    
