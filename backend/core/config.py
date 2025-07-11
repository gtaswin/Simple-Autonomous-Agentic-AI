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
        """Set up default configuration values"""
        self.settings = {
            "models": {
                "default_model": "mixtral-8x7b",
                "max_tokens": 500,
                "temperature": 0.7,
                "streaming": True
            },
            "memory": {
                "cache_ttl": 3600,
                "max_session_messages": 50,
                "embedding_model": "all-MiniLM-L6-v2"
            },
            "features": {
                "proactive_suggestions": True,
                "auto_learn_preferences": True,
                "task_management": True,
                "pattern_detection": True
            }
        }
        
        # Default model routing settings
        self.model_routes = {
            "instant": "openrouter/mistral-7b",
            "fast": "openrouter/mixtral-8x7b", 
            "balanced": "gemini-1.5-flash",
            "powerful": "gemini-1.5-pro",
            "creative": "claude-3-sonnet"
        }
        
    
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
    
    def get_default_user_id(self) -> str:
        """Get the default user ID from system configuration"""
        return self.get("system.default_user_id", "admin")
    
    def get_system_config(self) -> Dict[str, Any]:
        """Get system configuration settings"""
        return self.get("system", {
            "default_user_id": "admin",
            "session_timeout_hours": 24,
            "max_concurrent_sessions": 5
        })
    
    def get_autogen_config(self) -> Dict[str, Any]:
        """Get AutoGen-specific configuration"""
        return self.get("autogen", {})
    
    
    # Properties for easy access to common config sections
    @property
    def models(self):
        return self.settings.get("models", {})
    
    @property
    def memory(self):
        return self.settings.get("memory", {})
    
    
    @property
    def features(self):
        return self.settings.get("features", {})
    
    def get_token_limit(self, operation_type: str) -> int:
        """Get token limit for specific operation type from settings"""
        token_limits = self.get("token_limits", {})
        
        # Check for specific operation override first
        if operation_type in token_limits:
            return token_limits[operation_type]
        
        # Check by category mapping
        category = self.get(f"ai_functions.{operation_type}")
        if category:
            category_key = f"{category}_operations"
            if category_key in token_limits:
                return token_limits[category_key]
        
        # Default based on operation type patterns
        if "fast" in operation_type or "quick" in operation_type:
            return token_limits.get("fast_operations", 800)
        elif "quality" in operation_type or "complex" in operation_type:
            return token_limits.get("quality_operations", 2500)
        elif "premium" in operation_type or "planning" in operation_type:
            return token_limits.get("premium_operations", 4000)
        else:
            return token_limits.get("balanced_operations", 1500)
    
    def get_safe_token_limit(self, operation_type: str, requested_tokens: int = None) -> int:
        """Get safe token limit with bounds checking"""
        token_limits = self.get("token_limits", {})
        
        # Get base limit for operation
        base_limit = self.get_token_limit(operation_type)
        
        # Use requested if provided, otherwise use base
        target_limit = requested_tokens if requested_tokens else base_limit
        
        # Apply safety bounds
        max_limit = token_limits.get("absolute_maximum", 8000)
        min_limit = token_limits.get("minimum_safe", 100)
        
        return max(min_limit, min(target_limit, max_limit))
    
    def get_model_for_category(self, category: str) -> str:
        """Get model for specific category using new category-based system"""
        valid_categories = [
            "memory", "coordination", "chat", "thinking", "reasoning", 
            "research", "creative", "decisions"
        ]
        
        if category not in valid_categories:
            raise ValueError(f"Invalid category '{category}'. Must be one of: {', '.join(valid_categories)}")
        
        # Get the model category (fast/balanced/quality/premium) for this function
        function_category = self.get(f"ai_functions.{category}")
        if not function_category:
            raise ValueError(f"Missing configuration: ai_functions.{category} not found in settings.yaml. Please configure ai_functions section.")
        
        # Get the model list for this category
        model_list = self.get(f"model_categories.{function_category}")
        if not model_list or not isinstance(model_list, list) or len(model_list) == 0:
            raise ValueError(f"Missing configuration: model_categories.{function_category} not found or empty in settings.yaml. Please configure model_categories section.")
        
        # Return the first model in the list (primary model)
        return model_list[0]
        
    def get_model_list_for_category(self, category: str) -> list:
        """Get full model list for category (for fallback support)"""
        function_category = self.get(f"ai_functions.{category}")
        if not function_category:
            raise ValueError(f"Missing configuration: ai_functions.{category} not found in settings.yaml. Please configure ai_functions section.")
            
        model_list = self.get(f"model_categories.{function_category}")
        if not model_list or not isinstance(model_list, list) or len(model_list) == 0:
            raise ValueError(f"Missing configuration: model_categories.{function_category} not found or empty in settings.yaml. Please configure model_categories section.")
            
        return model_list
    
    def get_ai_analysis_model(self) -> str:
        """Get the AI analysis model (uses memory category)"""
        return self.get_model_for_category("memory")
    
    def get_llm_config(self, function_type: str = "chat") -> Dict[str, Any]:
        """
        Get AutoGen-compatible LLM configuration for specific function type
        """
        try:
            # Get model for the function type
            model = self.get_model_for_category(function_type)
            
            # Base configuration
            llm_config = {
                "model": model,
                "temperature": self.get("models.temperature", 0.7),
                "max_tokens": self.get_safe_token_limit(function_type),
                "timeout": 30
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
                    llm_config["base_url"] = "https://api.groq.com/openai/v1"
                    
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
            # Fallback configuration
            return {
                "model": "gpt-3.5-turbo",
                "api_key": os.getenv("OPENAI_API_KEY", "dummy"),
                "temperature": 0.7,
                "max_tokens": 500,
                "timeout": 30
            }
    
    def get_autogen_config(self) -> Dict[str, Any]:
        """Get AutoGen-specific configuration"""
        return {
            "memory_agent": self.get_llm_config("memory"),
            "research_agent": self.get_llm_config("research"),
            "intelligence_agent": self.get_llm_config("thinking"),
            "coordination": self.get_llm_config("coordination"),
            "tavily_api_key": self.get("tools.tavily.api_key"),
            "group_chat_settings": {
                "max_round": self.get("autogen.max_rounds", 10),
                "speaker_selection_method": self.get("autogen.speaker_selection", "auto"),
                "allow_repeat_speaker": self.get("autogen.allow_repeat_speaker", False)
            }
        }
    
    def resolve_model_tier(self, tier: str) -> str:
        """Resolve a model tier to actual LiteLLM model specification"""
        if not tier:
            raise ValueError("Model tier is required but not specified in configuration")
            
        # Check if tier is a category name
        valid_categories = [
            "memory", "learning", "reasoning", "reflection", "planning", 
            "agents", "coordination", "chat", "creative", "analysis", "decisions"
        ]
        
        if tier in valid_categories:
            return self.get_model_for_category(tier)
            
        if tier == "default":
            # Use the memory model as default
            return self.get_model_for_category("memory")
            
        # If it's not a category or "default", assume it's already a full model specification
        return tier