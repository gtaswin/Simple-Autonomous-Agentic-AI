import re
import time
import httpx
import json
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

try:
    import litellm
except ImportError:
    print("Warning: LiteLLM not available for ModelRouter simple response generation")
    litellm = None

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    INSTANT = "instant"
    FAST = "fast"
    BALANCED = "balanced"
    POWERFUL = "powerful"
    CREATIVE = "creative"


@dataclass
class QueryContext:
    user_id: str
    query: str
    conversation_history: List[Dict[str, str]]
    user_preferences: Dict[str, Any]
    current_time: float = None
    
    def __post_init__(self):
        if self.current_time is None:
            self.current_time = time.time()


@dataclass
class ModelResponse:
    content: str
    model_used: str
    tokens_used: int
    response_time: float
    cached: bool = False


class ModelRouter:
    """
    Legacy model router class.
    
    NOTE: This class is maintained for backward compatibility.
    For new development, use the PerformanceOptimizer class which provides
    better caching, connection pooling, and LiteLLM integration.
    """
    
    def __init__(self, config):
        self.config = config
        self.technical_terms = {
            "api", "database", "server", "client", "algorithm", "function",
            "class", "method", "variable", "array", "object", "json", "xml",
            "http", "https", "rest", "graphql", "sql", "python", "javascript",
            "react", "vue", "angular", "node", "express", "django", "flask"
        }
        self.reasoning_words = {
            "why", "how", "analyze", "explain", "compare", "evaluate",
            "calculate", "solve", "determine", "identify", "describe"
        }
        
        # Provider hierarchy: Get from config or use default
        if config and hasattr(config, 'raw_config'):
            performance_config = config.raw_config.get('performance', {})
            self.provider_hierarchy = performance_config.get('fallback_chain', ["openrouter", "gemini", "openai", "ollama"])
            
            # Get routing from config
            llm_config = config.raw_config.get('llm', {})
            routing_config = llm_config.get('routing', {})
        else:
            self.provider_hierarchy = ["openrouter", "gemini", "openai", "ollama"]
            routing_config = {}
        
        self.provider_health = {provider: True for provider in self.provider_hierarchy}
        
        # Note: Removed unused preferred_provider - not needed with simplified routing
    
    def select_model(self, query: str, context: QueryContext) -> tuple[str, str]:
        """Select provider and model based on simple heuristics"""
        
        # Simple provider selection based on availability
        provider = self.select_provider()
        
        # Simple model tier selection based on query length and basic patterns
        if self.is_greeting_or_simple(query):
            model_tier = "fast"
        elif len(query.split()) > 50 or self.requires_analysis(query):
            model_tier = "powerful"
        else:
            model_tier = "balanced"
            
        return provider, model_tier
    
    def select_provider(self, task_type: str = "default") -> str:
        """Select the best available provider"""
        
        # Simple fallback through hierarchy based on health
        for provider in self.provider_hierarchy:
            if self.provider_health.get(provider, False):
                return provider
                
        # Last resort
        return "openrouter"
    
    def is_greeting_or_simple(self, query: str) -> bool:
        simple_patterns = [
            r'\b(hi|hello|hey|thanks|thank you|yes|no|ok|okay)\b',
            r'\b(good morning|good afternoon|good evening)\b',
            r'\b(what time|what date|weather)\b'
        ]
        return any(re.search(pattern, query.lower()) for pattern in simple_patterns)
    
    def needs_memory_or_context(self, query: str, context: QueryContext) -> bool:
        """Check if query needs memory or context (kept for compatibility)"""
        memory_keywords = ["remember", "you said", "earlier", "before", "my preference", "I like"]
        return any(keyword in query.lower() for keyword in memory_keywords)
    
    def requires_analysis(self, query: str) -> bool:
        return any(word in query.lower() for word in self.reasoning_words)
    
    def needs_creativity(self, query: str) -> bool:
        creative_keywords = ["write", "create", "generate", "story", "poem", "joke", "creative"]
        return any(keyword in query.lower() for keyword in creative_keywords)
    
    async def call_model(self, provider: str, model_tier: str, messages: List[Dict], **kwargs) -> ModelResponse:
        """Call the specified provider and model"""
        
        start_time = time.time()
        
        try:
            if provider == "ollama":
                response = await self.call_ollama(model_tier, messages, **kwargs)
            elif provider == "openrouter":
                response = await self.call_openrouter(model_tier, messages, **kwargs)
            elif provider == "gemini":
                response = await self.call_gemini(model_tier, messages, **kwargs)
            elif provider == "openai":
                response = await self.call_openai(model_tier, messages, **kwargs)
            elif provider == "anthropic":
                response = await self.call_anthropic(model_tier, messages, **kwargs)
            else:
                raise ValueError(f"Unknown provider: {provider}")
            
            response_time = time.time() - start_time
            self.provider_health[provider] = True
            
            return ModelResponse(
                content=response["content"],
                model_used=f"{provider}:{response['model']}",
                tokens_used=response.get("tokens", 0),
                response_time=response_time
            )
            
        except Exception as e:
            logger.error(f"Error calling {provider}: {e}")
            self.provider_health[provider] = False
            
            # Try fallback provider
            if provider != "openrouter":  # Prevent infinite recursion
                fallback_provider = self.select_provider()
                if fallback_provider != provider:
                    return await self.call_model(fallback_provider, model_tier, messages, **kwargs)
            
            raise e
    
    async def generate_response(self, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Simple response generation method (legacy wrapper).
        
        For new code, use PerformanceOptimizer.llm_call_with_cache() instead.
        """
        if not litellm:
            return {
                "response": "LiteLLM not available. Please use PerformanceOptimizer for model calls.",
                "model_used": "fallback",
                "tokens": 0,
                "error": "LiteLLM not installed"
            }
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            # Get appropriate token limit from config
            from core.config import AssistantConfig
            config = AssistantConfig()
            max_tokens = config.get_safe_token_limit("chat_response")
            
            # Use configuration-based model selection
            model = config.get_model_for_category("chat")
            provider = model.split('/')[0] if '/' in model else None
            if not provider:
                raise ValueError(f"Model '{model}' must include provider prefix (e.g., 'groq/model-name')")
            api_key = config.get_api_key(provider) if config else None
            
            # Get temperature from configuration
            temperature = config.get("model_parameters.temperature", 0.7)
            
            # Prepare LiteLLM call with API key
            call_kwargs = {
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            if api_key:
                call_kwargs['api_key'] = api_key
            
            response = await litellm.acompletion(
                model=model,
                messages=messages,
                **call_kwargs
            )
            
            return {
                "response": response.choices[0].message.content,
                "model_used": "openrouter/mistral-7b",
                "tokens": response.usage.total_tokens if response.usage else 0
            }
            
        except Exception as e:
            logger.error(f"Simple response generation failed: {e}")
            return {
                "response": "I apologize, but I'm having trouble generating a response right now.",
                "model_used": "fallback",
                "tokens": 0,
                "error": str(e)
            }
    
    async def call_ollama(self, model_tier: str, messages: List[Dict], **kwargs) -> Dict:
        """Call Ollama local model"""
        
        model_map = self.config.get("llm", {}).get("ollama", {}).get("models", {})
        model = model_map.get(model_tier)
        
        base_url = self.config.get("llm", {}).get("ollama", {}).get("base_url")
        
        # Convert messages to Ollama format
        prompt = self._convert_messages_to_prompt(messages)
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": kwargs.get("temperature"),
                        "top_p": kwargs.get("top_p"),
                        "max_tokens": kwargs.get("max_tokens")
                    }
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.status_code}")
            
            result = response.json()
            
            return {
                "content": result.get("response", ""),
                "model": model,
                "tokens": result.get("eval_count", 0),
            }
    
    async def call_openrouter(self, model_tier: str, messages: List[Dict], **kwargs) -> Dict:
        """Call OpenRouter API"""
        
        model_map = self.config.get("llm", {}).get("openrouter", {}).get("models", {})
        model = model_map.get(model_tier)
        
        api_key = self.config.get("llm", {}).get("openrouter", {}).get("api_key")
        
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": kwargs.get("temperature"),
                    "max_tokens": kwargs.get("max_tokens")
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"OpenRouter API error: {response.status_code}")
            
            result = response.json()
            choice = result["choices"][0]
            
            return {
                "content": choice["message"]["content"],
                "model": model,
                "tokens": result.get("usage", {}).get("total_tokens", 0),
            }
    
    async def call_gemini(self, model_tier: str, messages: List[Dict], **kwargs) -> Dict:
        """Call Gemini API"""
        
        model = self.config.get("providers.gemini.model")
        
        api_key = self.config.get_api_key("gemini")
        
        # Convert messages to Gemini format
        gemini_messages = self._convert_to_gemini_format(messages)
        
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                headers={
                    "Content-Type": "application/json"
                },
                params={"key": api_key},
                json={
                    "contents": gemini_messages,
                    "generationConfig": {
                        "temperature": kwargs.get("temperature"),
                        "maxOutputTokens": kwargs.get("max_tokens")
                    }
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"Gemini API error: {response.status_code}")
            
            result = response.json()
            content = result["candidates"][0]["content"]["parts"][0]["text"]
            
            return {
                "content": content,
                "model": model,
                "tokens": result.get("usageMetadata", {}).get("totalTokenCount", 0),
            }
    
    async def call_openai(self, model_tier: str, messages: List[Dict], **kwargs) -> Dict:
        """Call OpenAI API (fallback)"""
        raise NotImplementedError("OpenAI provider not implemented. Use LiteLLM routing through performance layer instead.")
    
    async def call_anthropic(self, model_tier: str, messages: List[Dict], **kwargs) -> Dict:
        """Call Anthropic API (fallback)"""
        raise NotImplementedError("Anthropic provider not implemented. Use LiteLLM routing through performance layer instead.")
    
    def _convert_messages_to_prompt(self, messages: List[Dict]) -> str:
        """Convert OpenAI format messages to a prompt string"""
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"Human: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"
        prompt += "Assistant: "
        return prompt
    
    def _convert_to_gemini_format(self, messages: List[Dict]) -> List[Dict]:
        """Convert OpenAI format to Gemini format"""
        gemini_messages = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                # Gemini doesn't have system role, prepend to first user message
                if gemini_messages and gemini_messages[-1]["role"] == "user":
                    gemini_messages[-1]["parts"][0]["text"] = content + "\n\n" + gemini_messages[-1]["parts"][0]["text"]
                else:
                    gemini_messages.append({
                        "role": "user",
                        "parts": [{"text": content}]
                    })
            elif role == "user":
                gemini_messages.append({
                    "role": "user", 
                    "parts": [{"text": content}]
                })
            elif role == "assistant":
                gemini_messages.append({
                    "role": "model",
                    "parts": [{"text": content}]
                })
                
        return gemini_messages
    
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all providers"""
        
        health_status = {}
        
        # Check Ollama
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get("http://localhost:11434/api/tags")
                health_status["ollama"] = response.status_code == 200
        except:
            health_status["ollama"] = False
        
        # Check other providers with simple API calls
        # (Implementation would test each provider)
        
        self.provider_health.update(health_status)
        return health_status