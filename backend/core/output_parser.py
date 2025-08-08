"""
Phase 5: Output Parsing Infrastructure
Integrates structured output parsing with the existing 4-agent system.
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Type
from functools import wraps

from .output_schemas import (
    BaseAgentOutput, AgentType, ProcessingModel, OperationStatus,
    MemoryReaderOutput, KnowledgeAgentOutput, OrganizerAgentOutput, MemoryWriterOutput,
    WorkflowExecutionOutput, WorkflowPattern, AutonomousOperationOutput,
    OutputParser as SchemaParser, serialize_agent_output
)

logger = logging.getLogger(__name__)


# ================================================================================
# DECORATORS FOR AUTOMATIC OUTPUT PARSING
# ================================================================================

def structured_output(agent_type: AgentType, capture_timing: bool = True):
    """
    Decorator to automatically parse agent outputs into structured schemas.
    
    Args:
        agent_type: Type of agent for schema selection
        capture_timing: Whether to capture execution timing
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = datetime.now()
            agent_instance = args[0] if args else None
            
            try:
                # Execute the original function
                raw_output = await func(*args, **kwargs)
                
                # Calculate processing time
                processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
                
                # Parse into structured output
                structured = await OutputParsingService.parse_agent_output(
                    agent_type=agent_type,
                    raw_output=raw_output,
                    agent_instance=agent_instance,
                    processing_time_ms=processing_time_ms if capture_timing else 0.0
                )
                
                return structured
                
            except Exception as e:
                # Create error output on failure
                processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
                
                return BaseAgentOutput(
                    agent_name=getattr(agent_instance, 'agent_name', 'unknown'),
                    agent_type=agent_type,
                    processing_model=ProcessingModel.LOCAL_TRANSFORMERS,
                    operation_status=OperationStatus.ERROR,
                    processing_time_ms=processing_time_ms,
                    error_details=str(e),
                    metadata={"original_error": str(e)}
                )
        
        return wrapper
    return decorator


def workflow_output(capture_performance: bool = True):
    """
    Decorator for workflow-level output parsing.
    
    Args:
        capture_performance: Whether to capture performance metrics
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = datetime.now()
            
            try:
                raw_result = await func(*args, **kwargs)
                processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
                
                # Parse workflow output
                structured = await OutputParsingService.parse_workflow_output(
                    raw_result=raw_result,
                    processing_time_ms=processing_time_ms if capture_performance else 0.0
                )
                
                return structured
                
            except Exception as e:
                processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
                
                logger.error(f"Workflow execution failed: {e}")
                return WorkflowExecutionOutput(
                    agent_name="workflow_orchestrator",
                    processing_model=ProcessingModel.HYBRID,
                    final_response=f"Workflow failed: {str(e)}",
                    workflow_pattern=WorkflowPattern.SIMPLE_MEMORY_ONLY,
                    operation_status=OperationStatus.ERROR,
                    processing_time_ms=processing_time_ms,
                    error_details=str(e)
                )
        
        return wrapper
    return decorator


# ================================================================================
# OUTPUT PARSING SERVICE
# ================================================================================

class OutputParsingService:
    """Service for parsing and structuring agent outputs"""
    
    @staticmethod
    async def parse_agent_output(
        agent_type: AgentType,
        raw_output: Dict[str, Any],
        agent_instance: Any = None,
        processing_time_ms: float = 0.0
    ) -> BaseAgentOutput:
        """Parse raw agent output into structured schema"""
        
        try:
            # Enhance raw output with additional metadata
            enhanced_output = await OutputParsingService._enhance_raw_output(
                raw_output, agent_type, agent_instance, processing_time_ms
            )
            
            # Use schema parser to create structured output
            structured = SchemaParser.parse_agent_output(agent_type.value, enhanced_output)
            
            logger.debug(f"✅ Parsed {agent_type.value} output: {len(enhanced_output)} fields")
            return structured
            
        except Exception as e:
            logger.error(f"❌ Failed to parse {agent_type.value} output: {e}")
            
            # Return fallback schema
            return BaseAgentOutput(
                agent_name=getattr(agent_instance, 'agent_name', agent_type.value),
                agent_type=agent_type,
                processing_model=ProcessingModel.LOCAL_TRANSFORMERS,
                operation_status=OperationStatus.ERROR,
                processing_time_ms=processing_time_ms,
                error_details=f"Output parsing failed: {str(e)}",
                metadata={"raw_output": raw_output}
            )
    
    @staticmethod
    async def parse_workflow_output(
        raw_result: Dict[str, Any],
        processing_time_ms: float = 0.0
    ) -> WorkflowExecutionOutput:
        """Parse workflow execution result into structured output"""
        
        try:
            # Extract workflow information
            workflow_data = {
                "agent_name": "langgraph_orchestrator",
                "final_response": raw_result.get("response", ""),
                "workflow_pattern": WorkflowPattern(raw_result.get("metadata", {}).get("workflow_pattern", WorkflowPattern.SIMPLE_MEMORY_ONLY)),
                "agents_executed": raw_result.get("metadata", {}).get("agents_executed", []),
                "execution_order": raw_result.get("metadata", {}).get("execution_order", []),
                "total_processing_time_ms": processing_time_ms,
                "complexity_score": raw_result.get("metadata", {}).get("complexity_score", 0.0),
                "memory_context_available": raw_result.get("metadata", {}).get("memory_context_available", False),
                "knowledge_context_available": raw_result.get("metadata", {}).get("knowledge_context_available", False),
                "research_performed": "knowledge_agent" in raw_result.get("metadata", {}).get("agents_executed", []),
                "processing_model": ProcessingModel.HYBRID,
                "operation_status": OperationStatus.SUCCESS if not raw_result.get("metadata", {}).get("processing_failed") else OperationStatus.ERROR,
                "timestamp": datetime.now(),
                "metadata": raw_result.get("metadata", {})
            }
            
            # Handle parallel execution metrics
            if raw_result.get("metadata", {}).get("parallel_execution"):
                from .output_schemas import ParallelExecutionMetrics
                workflow_data["parallel_execution"] = ParallelExecutionMetrics(
                    parallel_agents=raw_result["metadata"].get("parallel_agents", []),
                    execution_phase=raw_result["metadata"].get("execution_phase", "sequential"),
                    speedup_factor=raw_result["metadata"].get("speedup_factor", 1.0),
                    concurrent_time_saved_ms=raw_result["metadata"].get("concurrent_time_saved_ms", 0.0)
                )
            
            return WorkflowExecutionOutput(**workflow_data)
            
        except Exception as e:
            logger.error(f"❌ Failed to parse workflow output: {e}")
            return WorkflowExecutionOutput(
                agent_name="langgraph_orchestrator",
                processing_model=ProcessingModel.HYBRID,
                final_response=raw_result.get("response", "Workflow parsing failed"),
                workflow_pattern=WorkflowPattern.SIMPLE_MEMORY_ONLY,
                operation_status=OperationStatus.ERROR,
                processing_time_ms=processing_time_ms,
                error_details=str(e),
                metadata={"raw_result": raw_result}
            )
    
    @staticmethod
    async def _enhance_raw_output(
        raw_output: Dict[str, Any],
        agent_type: AgentType,
        agent_instance: Any,
        processing_time_ms: float
    ) -> Dict[str, Any]:
        """Enhance raw output with additional metadata and standardized fields"""
        
        enhanced = raw_output.copy()
        
        # Add standard fields
        enhanced["processing_time_ms"] = processing_time_ms
        enhanced["timestamp"] = datetime.now()
        enhanced["operation_status"] = OperationStatus.SUCCESS
        
        # Add agent-specific enhancements
        if agent_type == AgentType.MEMORY_READER:
            enhanced = await OutputParsingService._enhance_memory_reader_output(enhanced, agent_instance)
        elif agent_type == AgentType.KNOWLEDGE_AGENT:
            enhanced = await OutputParsingService._enhance_knowledge_agent_output(enhanced, agent_instance)
        elif agent_type == AgentType.ORGANIZER_AGENT:
            enhanced = await OutputParsingService._enhance_organizer_agent_output(enhanced, agent_instance)
        elif agent_type == AgentType.MEMORY_WRITER:
            enhanced = await OutputParsingService._enhance_memory_writer_output(enhanced, agent_instance)
        
        return enhanced
    
    @staticmethod
    async def _enhance_memory_reader_output(output: Dict[str, Any], agent_instance: Any) -> Dict[str, Any]:
        """Enhance Memory Reader output with structured data"""
        
        # Parse memory items if context is available
        if "context_summary" in output:
            output["memories_found"] = 0
            output["short_term_memories"] = []
            output["long_term_memories"] = []
            output["working_memories"] = []
            output["retrieval_method"] = output.get("retrieval_method", "langchain_vector_store_retriever")
            output["context_relevance_score"] = 0.8  # Default score
            output["memory_coverage"] = {"short_term": 0, "long_term": 0, "working": 0}
            
            # Extract search query from agent instance or output
            output["search_query"] = getattr(agent_instance, 'last_query', output.get("query", "unknown"))
        
        return output
    
    @staticmethod
    async def _enhance_knowledge_agent_output(output: Dict[str, Any], agent_instance: Any) -> Dict[str, Any]:
        """Enhance Knowledge Agent output with structured data"""
        
        if "knowledge_summary" in output:
            output["search_results"] = []
            output["research_type"] = output.get("research_type", "general")
            output["sources_consulted"] = ["wikipedia", "wikidata"]
            output["search_completed"] = True
            output["results_found"] = 1 if output["knowledge_summary"] else 0
            output["average_relevance"] = 0.7
            output["was_cached"] = "cached" in output.get("processing_model", "").lower()
            
            # Extract search query
            output["search_query"] = getattr(agent_instance, 'last_query', output.get("query", "unknown"))
        
        return output
    
    @staticmethod
    async def _enhance_organizer_agent_output(output: Dict[str, Any], agent_instance: Any) -> Dict[str, Any]:
        """Enhance Organizer Agent output with structured data"""
        
        if "response" in output:
            from .output_schemas import ContextQualityMetrics, SynthesisQuality
            
            # Add synthesis quality assessment
            output["synthesis_successful"] = output.get("synthesis_successful", True)
            output["synthesis_quality"] = SynthesisQuality.GOOD.value
            
            # Add context quality metrics
            output["context_quality"] = {
                "memory_context_quality": output.get("memory_context_quality", 0.0),
                "knowledge_context_quality": output.get("knowledge_context_quality", 0.0),
                "overall_quality": 0.0,
                "context_completeness": 0.0
            }
            
            # Calculate overall quality
            mem_quality = output["context_quality"]["memory_context_quality"]
            know_quality = output["context_quality"]["knowledge_context_quality"]
            output["context_quality"]["overall_quality"] = (mem_quality + know_quality) / 2
            output["context_quality"]["context_completeness"] = 0.8
            
            # Add usage tracking
            output["memory_context_used"] = mem_quality > 0
            output["knowledge_context_used"] = know_quality > 0
            output["llm_model_used"] = "external_llm"
            output["token_usage"] = {"total": int(len(output["response"].split()) * 1.3)}  # Rough estimate as integer
            output["temperature"] = 0.7
            output["confidence_score"] = 0.8
        
        return output
    
    @staticmethod
    async def _enhance_memory_writer_output(output: Dict[str, Any], agent_instance: Any) -> Dict[str, Any]:
        """Enhance Memory Writer output with structured data"""
        
        # Restructure storage statistics
        from .output_schemas import MemoryStorageStats
        
        storage_stats = MemoryStorageStats(
            facts_extracted=output.get("facts_extracted", 0),
            short_term_stored=output.get("short_term_stored", 0),
            long_term_stored=output.get("long_term_stored", 0),
            session_stored=output.get("session_stored", True),
            working_memory_updated=output.get("working_memory_updated", False),
            duplicates_found=0,
            storage_errors=0
        )
        
        output["storage_stats"] = storage_stats.dict()
        output["extracted_facts"] = []  # Would need to be populated from actual extraction
        output["user_message_processed"] = True
        output["ai_response_processed"] = True
        output["conversation_metadata"] = {}
        output["long_term_storage_blocked"] = False
        
        return output


# ================================================================================
# OUTPUT VALIDATION SERVICE
# ================================================================================

class OutputValidationService:
    """Service for validating structured outputs"""
    
    @staticmethod
    async def validate_agent_output(output: BaseAgentOutput) -> Dict[str, Any]:
        """Validate structured agent output"""
        
        validation_result = SchemaParser.validate_output_structure(output)
        
        # Additional validation checks
        validation_result.update({
            "timing_validation": OutputValidationService._validate_timing(output),
            "metadata_validation": OutputValidationService._validate_metadata(output),
            "field_completeness": OutputValidationService._assess_field_completeness(output)
        })
        
        return validation_result
    
    @staticmethod
    def _validate_timing(output: BaseAgentOutput) -> Dict[str, Any]:
        """Validate timing information"""
        return {
            "has_timestamp": output.timestamp is not None,
            "has_processing_time": output.processing_time_ms >= 0,
            "timing_reasonable": 0 <= output.processing_time_ms <= 300000,  # Max 5 minutes
            "timestamp_recent": (datetime.now() - output.timestamp).total_seconds() < 3600  # Within 1 hour
        }
    
    @staticmethod
    def _validate_metadata(output: BaseAgentOutput) -> Dict[str, Any]:
        """Validate metadata structure"""
        return {
            "has_metadata": bool(output.metadata),
            "metadata_serializable": True,  # Pydantic ensures this
            "agent_name_present": bool(output.agent_name),
            "agent_type_valid": output.agent_type in AgentType.__members__.values()
        }
    
    @staticmethod
    def _assess_field_completeness(output: BaseAgentOutput) -> Dict[str, Any]:
        """Assess completeness of required fields"""
        required_fields = ["agent_name", "agent_type", "processing_model", "operation_status"]
        optional_fields = ["error_details", "metadata"]
        
        completeness = {
            "required_fields_present": all(getattr(output, field, None) is not None for field in required_fields),
            "optional_fields_populated": sum(1 for field in optional_fields if getattr(output, field, None) is not None),
            "total_field_count": len(output.__fields__),
            "completeness_score": 0.0
        }
        
        # Calculate completeness score
        required_score = 0.7 if completeness["required_fields_present"] else 0.0
        optional_score = 0.3 * (completeness["optional_fields_populated"] / len(optional_fields))
        completeness["completeness_score"] = required_score + optional_score
        
        return completeness


# ================================================================================
# TYPE CONVERSION UTILITIES
# ================================================================================

class TypeConversionService:
    """Service for converting between structured outputs and legacy formats"""
    
    @staticmethod
    def to_legacy_format(structured_output: BaseAgentOutput) -> Dict[str, Any]:
        """Convert structured output back to legacy dictionary format"""
        
        legacy_dict = structured_output.dict()
        
        # Transform to match legacy API expectations
        if isinstance(structured_output, MemoryReaderOutput):
            return {
                "context_summary": structured_output.context_summary,
                "memories_found": structured_output.memories_found,
                "retrieval_method": structured_output.retrieval_method
            }
        
        elif isinstance(structured_output, KnowledgeAgentOutput):
            return {
                "knowledge_summary": structured_output.knowledge_summary,
                "sources_found": structured_output.results_found,
                "research_type": structured_output.research_type,
                "confidence": structured_output.average_relevance,
                "processing_model": structured_output.processing_model
            }
        
        elif isinstance(structured_output, OrganizerAgentOutput):
            return {
                "response": structured_output.response,
                "synthesis_successful": structured_output.synthesis_successful,
                "memory_context_quality": structured_output.context_quality.memory_context_quality,
                "knowledge_context_quality": structured_output.context_quality.knowledge_context_quality,
                "processing_model": structured_output.processing_model
            }
        
        elif isinstance(structured_output, MemoryWriterOutput):
            return {
                "facts_extracted": structured_output.storage_stats.facts_extracted,
                "short_term_stored": structured_output.storage_stats.short_term_stored,
                "long_term_stored": structured_output.storage_stats.long_term_stored,
                "session_stored": structured_output.storage_stats.session_stored,
                "processing_model": structured_output.processing_model
            }
        
        # Default: return full dictionary
        return legacy_dict
    
    @staticmethod
    def extract_api_response(structured_output: Union[BaseAgentOutput, WorkflowExecutionOutput]) -> Dict[str, Any]:
        """Extract API response format from structured output"""
        
        if isinstance(structured_output, WorkflowExecutionOutput):
            return {
                "response": structured_output.final_response,
                "metadata": {
                    "architecture": "langgraph_multi_agent",
                    "agents_executed": structured_output.agents_executed,
                    "execution_order": structured_output.execution_order,
                    "workflow_pattern": structured_output.workflow_pattern,
                    "processing_time": structured_output.total_processing_time_ms / 1000,
                    "complexity_score": structured_output.complexity_score,
                    "memory_context_available": structured_output.memory_context_available,
                    "knowledge_context_available": structured_output.knowledge_context_available
                },
                "timestamp": structured_output.timestamp.isoformat(),
                "user_name": "user"  # Would be extracted from context
            }
        
        # For individual agent outputs, return basic format
        return {
            "response": str(structured_output.dict()),
            "metadata": {
                "agent_type": structured_output.agent_type,
                "processing_model": structured_output.processing_model,
                "operation_status": structured_output.operation_status,
                "processing_time": structured_output.processing_time_ms / 1000
            },
            "timestamp": structured_output.timestamp.isoformat()
        }


# ================================================================================
# EXPORT SERVICE INSTANCE
# ================================================================================

# Global service instances
output_parsing_service = OutputParsingService()
output_validation_service = OutputValidationService()
type_conversion_service = TypeConversionService()

# Export for easy importing
__all__ = [
    "structured_output", "workflow_output",
    "OutputParsingService", "OutputValidationService", "TypeConversionService",
    "output_parsing_service", "output_validation_service", "type_conversion_service"
]