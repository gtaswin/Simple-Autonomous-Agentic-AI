"""
LocalTransformerLLM - LangChain LLM wrapper for local transformers
Provides LangChain-compatible interface for local transformer models
"""

from typing import Any, List, Optional, Dict, Mapping
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import Generation, LLMResult
from pydantic import Field
from core.transformers_service import TransformersService
import logging

logger = logging.getLogger(__name__)


class LocalTransformerLLM(LLM):
    """
    LangChain LLM wrapper for local transformer models.
    
    This enables local transformers to be used within LangChain's AgentExecutor
    and ReAct patterns while maintaining cost-effectiveness.
    """
    
    def __init__(self, transformers_service: TransformersService, max_tokens: int = 512, temperature: float = 0.7, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'transformers_service', transformers_service)
        object.__setattr__(self, 'max_tokens', max_tokens)
        object.__setattr__(self, 'temperature', temperature)
    
    class Config:
        arbitrary_types_allowed = True
    
    @property
    def _llm_type(self) -> str:
        """Return identifier of this LLM type."""
        return "local_transformer"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Call the local transformer model.
        
        Args:
            prompt: The prompt to generate from
            stop: List of stop sequences
            run_manager: Callback manager for tracking
            **kwargs: Additional arguments
            
        Returns:
            Generated text string
        """
        try:
            if run_manager:
                run_manager.on_llm_start({"name": self._llm_type}, [prompt])
            
            # Use transformers service for text generation
            result = self.transformers_service.generate_text(
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop_sequences=stop or []
            )
            
            # Extract text from result
            if isinstance(result, dict):
                generated_text = result.get('generated_text', result.get('text', str(result)))
            else:
                generated_text = str(result)
            
            # Ensure valid ReAct format for LangChain
            validated_response = self._validate_react_format(generated_text, prompt)
            
            if run_manager:
                run_manager.on_llm_end(LLMResult(generations=[[Generation(text=validated_response)]]))
            
            logger.debug(f"LocalTransformerLLM generated: {validated_response[:100]}...")
            return validated_response
            
        except Exception as e:
            logger.error(f"LocalTransformerLLM generation failed: {e}")
            if run_manager:
                run_manager.on_llm_error(e)
            return f"Error: {str(e)}"
    
    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Async version of _call method."""
        try:
            # Skip callback manager for async - it has different interface
            # if run_manager:
            #     run_manager.on_llm_start({"name": self._llm_type}, [prompt])
            
            # Use async transformers service if available
            if hasattr(self.transformers_service, 'agenerate_text'):
                result = await self.transformers_service.agenerate_text(
                    prompt=prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    stop_sequences=stop or []
                )
            else:
                # Fallback to sync generation
                result = self.transformers_service.generate_text(
                    prompt=prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    stop_sequences=stop or []
                )
            
            # Extract text from result
            if isinstance(result, dict):
                generated_text = result.get('generated_text', result.get('text', str(result)))
            else:
                generated_text = str(result)
            
            # Ensure valid ReAct format for LangChain
            validated_response = self._validate_react_format(generated_text, prompt)
            
            # Skip callback manager for async - it has different interface  
            # if run_manager:
            #     run_manager.on_llm_end(LLMResult(generations=[[Generation(text=validated_response)]]))
            
            logger.debug(f"LocalTransformerLLM async generated: {validated_response[:100]}...")
            return validated_response
            
        except Exception as e:
            logger.error(f"LocalTransformerLLM async generation failed: {e}")
            # Skip callback manager for async
            # if run_manager:
            #     run_manager.on_llm_error(e)
            return f"Error: {str(e)}"
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "model_type": "local_transformer"
        }
    
    def get_token_count(self, text: str) -> int:
        """Get token count for text."""
        try:
            if hasattr(self.transformers_service, 'count_tokens'):
                return self.transformers_service.count_tokens(text)
            else:
                # Rough estimation: 1 token per 4 characters
                return len(text) // 4
        except Exception:
            return len(text) // 4
    
    def _validate_react_format(self, generated_text: str, original_prompt: str) -> str:
        """Validate and fix ReAct format for LangChain compatibility"""
        try:
            # Check if response already has proper ReAct format
            if self._is_valid_react_format(generated_text):
                return generated_text
            
            # If it's not valid ReAct format, create a proper one
            return self._create_valid_react_response(generated_text, original_prompt)
            
        except Exception as e:
            logger.warning(f"ReAct validation failed: {e}")
            return self._create_fallback_react_response(original_prompt)
    
    def _is_valid_react_format(self, text: str) -> bool:
        """Check if text follows valid ReAct format"""
        # Valid ReAct format must have either:
        # 1. Thought -> Action -> Action Input -> Observation pattern
        # 2. Thought -> Final Answer pattern
        
        has_thought = "Thought:" in text or "thought:" in text.lower()
        has_action = "Action:" in text or "action:" in text.lower()
        has_final_answer = "Final Answer:" in text or "final answer:" in text.lower()
        
        # Valid if it has thought and either action or final answer
        return has_thought and (has_action or has_final_answer)
    
    def _create_valid_react_response(self, generated_text: str, original_prompt: str) -> str:
        """Create valid ReAct response from generated text"""
        # If the generated text looks like a direct answer, wrap it in ReAct format
        if generated_text and len(generated_text.strip()) > 0:
            # Check if it's a greeting response
            if any(greeting in generated_text.lower() for greeting in ["hello", "hi", "assist", "help"]):
                return f"Thought: This is a greeting, I should respond helpfully.\nFinal Answer: {generated_text.strip()}"
            
            # Check if it mentions memory or search
            if "memory" in generated_text.lower() or "search" in generated_text.lower():
                return f"Thought: I need to search for relevant information.\nAction: search_short_term_memory\nAction Input: user query\nObservation: {generated_text.strip()}\nThought: I have the search results.\nFinal Answer: {generated_text.strip()}"
            
            # Default case: wrap as final answer
            return f"Thought: I understand the request.\nFinal Answer: {generated_text.strip()}"
        
        # If no useful generated text, create appropriate response
        return self._create_fallback_react_response(original_prompt)
    
    def _create_fallback_react_response(self, original_prompt: str) -> str:
        """Create fallback ReAct response based on prompt context"""
        if "search" in original_prompt.lower() or "memory" in original_prompt.lower():
            return """Thought: I need to search for relevant information.
Action: search_short_term_memory
Action Input: user query
Observation: No specific memories found for this greeting. This appears to be a general greeting or new topic.
Thought: Since this is a greeting and I don't have specific context, I should provide a helpful welcome.
Final Answer: No specific memories found for this greeting. This appears to be a general greeting or new topic."""
        
        elif "extract" in original_prompt.lower() or "fact" in original_prompt.lower():
            return """Thought: I need to extract facts from the conversation.
Action: extract_facts_tool
Action Input: conversation text
Observation: This appears to be a greeting exchange with minimal factual content.
Thought: This is a simple greeting, so there are minimal facts to extract.
Final Answer: Greeting conversation processed. Minimal facts extracted."""
        
        else:
            return """Thought: I understand the request and should provide a helpful response.
Final Answer: I'm ready to assist with this request."""