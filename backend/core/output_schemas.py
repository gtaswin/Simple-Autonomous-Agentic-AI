"""
Phase 5: Structured Output Parsing Schemas
Pydantic schemas for standardizing agent outputs across the 4-agent system.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Literal
from pydantic import BaseModel, Field, validator
import json


# ================================================================================
# BASE OUTPUT SCHEMAS
# ================================================================================

class AgentType(str, Enum):
    """Supported agent types in the system"""
    MEMORY_READER = "memory_reader"
    MEMORY_WRITER = "memory_writer"
    KNOWLEDGE_AGENT = "knowledge_agent"
    ORGANIZER_AGENT = "organizer_agent"
    AUTONOMOUS_ROUTER = "autonomous_router"


class ProcessingModel(str, Enum):
    """Processing model types"""
    LOCAL_TRANSFORMERS = "local_transformers_only"
    EXTERNAL_LLM = "external_llm_only"
    HYBRID = "hybrid"
    CACHED_RESULT = "cached_result"


class OperationStatus(str, Enum):
    """Operation completion status"""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    ERROR = "error"


class BaseAgentOutput(BaseModel):
    """Base schema for all agent outputs"""
    agent_name: str = Field(..., description="Name of the agent")
    agent_type: AgentType = Field(..., description="Type of agent")
    processing_model: ProcessingModel = Field(..., description="Processing model used")
    operation_status: OperationStatus = Field(default=OperationStatus.SUCCESS)
    timestamp: datetime = Field(default_factory=datetime.now)
    processing_time_ms: float = Field(default=0.0, description="Processing time in milliseconds")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error_details: Optional[str] = Field(None, description="Error details if operation failed")

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ================================================================================
# MEMORY READER AGENT OUTPUTS
# ================================================================================

class MemoryType(str, Enum):
    """Memory types retrieved"""
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    WORKING = "working"
    SESSION = "session"


class MemoryItem(BaseModel):
    """Individual memory item"""
    content: str = Field(..., description="Memory content")
    importance_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    memory_type: MemoryType = Field(..., description="Type of memory")
    stored_at: datetime = Field(..., description="When memory was stored")
    ttl_expires: Optional[datetime] = Field(None, description="When memory expires (for short-term)")
    source: Optional[str] = Field(None, description="Source of the memory")
    
    class Config:
        use_enum_values = True


class MemoryReaderOutput(BaseAgentOutput):
    """Structured output for Memory Reader Agent"""
    agent_type: Literal[AgentType.MEMORY_READER] = Field(default=AgentType.MEMORY_READER)
    processing_model: Literal[ProcessingModel.LOCAL_TRANSFORMERS] = Field(default=ProcessingModel.LOCAL_TRANSFORMERS)
    
    # Context results
    context_summary: str = Field(..., description="Summary of retrieved context")
    memories_found: int = Field(default=0, ge=0)
    
    # Detailed memory breakdown
    short_term_memories: List[MemoryItem] = Field(default_factory=list)
    long_term_memories: List[MemoryItem] = Field(default_factory=list)
    working_memories: List[MemoryItem] = Field(default_factory=list)
    
    # Search and retrieval details
    search_query: str = Field(..., description="Original search query")
    retrieval_method: str = Field(default="langchain_vector_store_retriever")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    
    # Quality metrics
    context_relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    memory_coverage: Dict[str, int] = Field(default_factory=dict, description="Coverage by memory type")


# ================================================================================
# KNOWLEDGE AGENT OUTPUTS
# ================================================================================

class SearchResultSource(str, Enum):
    """Knowledge search result sources"""
    WIKIPEDIA = "wikipedia"
    WIKIDATA = "wikidata"
    COMBINED = "wikipedia_wikidata"
    CACHED = "cached"


class KnowledgeSearchResult(BaseModel):
    """Individual knowledge search result"""
    content: str = Field(..., description="Search result content")
    source: SearchResultSource = Field(..., description="Source of information")
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    url: Optional[str] = Field(None, description="Source URL if available")
    
    class Config:
        use_enum_values = True


class KnowledgeAgentOutput(BaseAgentOutput):
    """Structured output for Knowledge Agent"""
    agent_type: Literal[AgentType.KNOWLEDGE_AGENT] = Field(default=AgentType.KNOWLEDGE_AGENT)
    processing_model: Literal[ProcessingModel.LOCAL_TRANSFORMERS] = Field(default=ProcessingModel.LOCAL_TRANSFORMERS)
    
    # Research results
    knowledge_summary: str = Field(..., description="Summary of research findings")
    search_results: List[KnowledgeSearchResult] = Field(default_factory=list)
    
    # Search details
    search_query: str = Field(..., description="Original search query")
    research_type: str = Field(default="general", description="Type of research conducted")
    sources_consulted: List[SearchResultSource] = Field(default_factory=list)
    
    # Quality and performance metrics
    search_completed: bool = Field(default=True)
    results_found: int = Field(default=0, ge=0)
    average_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    was_cached: bool = Field(default=False, description="Whether result was retrieved from cache")


# ================================================================================
# ORGANIZER AGENT OUTPUTS
# ================================================================================

class SynthesisQuality(str, Enum):
    """Quality of synthesis performed"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ADEQUATE = "adequate"
    POOR = "poor"


class ContextQualityMetrics(BaseModel):
    """Quality metrics for input contexts"""
    memory_context_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    knowledge_context_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    overall_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    context_completeness: float = Field(default=0.0, ge=0.0, le=1.0)


class OrganizerAgentOutput(BaseAgentOutput):
    """Structured output for Organizer Agent"""
    agent_type: Literal[AgentType.ORGANIZER_AGENT] = Field(default=AgentType.ORGANIZER_AGENT)
    processing_model: Literal[ProcessingModel.EXTERNAL_LLM] = Field(default=ProcessingModel.EXTERNAL_LLM)
    
    # Response generation
    response: str = Field(..., description="Final synthesized response")
    synthesis_successful: bool = Field(default=True)
    synthesis_quality: SynthesisQuality = Field(default=SynthesisQuality.GOOD)
    
    # Context analysis
    context_quality: ContextQualityMetrics = Field(default_factory=ContextQualityMetrics)
    memory_context_used: bool = Field(default=False)
    knowledge_context_used: bool = Field(default=False)
    
    # LLM usage details
    llm_model_used: Optional[str] = Field(None, description="Specific LLM model used")
    token_usage: Dict[str, int] = Field(default_factory=dict, description="Token usage statistics")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    
    # Reasoning details (optional for debugging)
    reasoning_steps: Optional[List[str]] = Field(None, description="Reasoning process steps")
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)

    class Config:
        use_enum_values = True


# ================================================================================
# MEMORY WRITER AGENT OUTPUTS
# ================================================================================

class FactType(str, Enum):
    """Types of facts extracted"""
    PERSONAL_INFO = "personal_information"
    PREFERENCE = "preference"
    GOAL = "goal"
    EXPERIENCE = "experience"
    SKILL = "skill"
    STATEMENT = "statement"
    ENTITY = "entity"
    UNKNOWN = "unknown"


class ExtractedFact(BaseModel):
    """Individual extracted fact"""
    content: str = Field(..., description="Fact content")
    fact_type: FactType = Field(..., description="Type of fact")
    importance_score: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source: str = Field(..., description="Source of fact (user_message/ai_response)")
    storage_destination: str = Field(..., description="Where fact was stored (short_term/long_term)")
    ttl_seconds: Optional[int] = Field(None, description="TTL for short-term storage")
    extracted_at: datetime = Field(default_factory=datetime.now)

    class Config:
        use_enum_values = True


class MemoryStorageStats(BaseModel):
    """Memory storage statistics"""
    facts_extracted: int = Field(default=0, ge=0)
    short_term_stored: int = Field(default=0, ge=0)
    long_term_stored: int = Field(default=0, ge=0)
    session_stored: bool = Field(default=True)
    working_memory_updated: bool = Field(default=False)
    duplicates_found: int = Field(default=0, ge=0)
    storage_errors: int = Field(default=0, ge=0)


class MemoryWriterOutput(BaseAgentOutput):
    """Structured output for Memory Writer Agent"""
    agent_type: Literal[AgentType.MEMORY_WRITER] = Field(default=AgentType.MEMORY_WRITER)
    processing_model: Literal[ProcessingModel.LOCAL_TRANSFORMERS] = Field(default=ProcessingModel.LOCAL_TRANSFORMERS)
    
    # Processing results
    storage_stats: MemoryStorageStats = Field(default_factory=MemoryStorageStats)
    extracted_facts: List[ExtractedFact] = Field(default_factory=list)
    
    # Input processing details
    user_message_processed: bool = Field(default=False)
    ai_response_processed: bool = Field(default=False)
    conversation_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Storage constraints applied
    storage_constraints_applied: Optional[Dict[str, Any]] = Field(None)
    long_term_storage_blocked: bool = Field(default=False)
    importance_cap_applied: Optional[float] = Field(None)


# ================================================================================
# WORKFLOW AND ORCHESTRATOR OUTPUTS
# ================================================================================

class WorkflowPattern(str, Enum):
    """Workflow execution patterns"""
    SIMPLE_MEMORY_ONLY = "simple_memory_only"
    RESEARCH_ENHANCED = "research_enhanced"
    COMPLEX_REASONING = "complex_reasoning"
    PARALLEL_EXECUTION = "parallel_execution"
    AUTONOMOUS_OPERATION = "autonomous_operation"


class ParallelExecutionMetrics(BaseModel):
    """Metrics for parallel execution"""
    parallel_agents: List[str] = Field(default_factory=list)
    execution_phase: str = Field(default="sequential")
    speedup_factor: float = Field(default=1.0, ge=1.0)
    concurrent_time_saved_ms: float = Field(default=0.0, ge=0.0)
    

class WorkflowExecutionOutput(BaseAgentOutput):
    """Output for complete workflow execution"""
    agent_type: Literal[AgentType.AUTONOMOUS_ROUTER] = Field(default=AgentType.AUTONOMOUS_ROUTER)
    
    # Workflow results
    final_response: str = Field(..., description="Final workflow response")
    workflow_pattern: WorkflowPattern = Field(..., description="Execution pattern used")
    agents_executed: List[str] = Field(default_factory=list)
    execution_order: List[str] = Field(default_factory=list)
    
    # Performance metrics
    total_processing_time_ms: float = Field(default=0.0, ge=0.0)
    complexity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    parallel_execution: Optional[ParallelExecutionMetrics] = Field(None)
    
    # Context availability
    memory_context_available: bool = Field(default=False)
    knowledge_context_available: bool = Field(default=False)
    research_performed: bool = Field(default=False)
    
    # Quality metrics
    response_quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    user_satisfaction_predicted: float = Field(default=0.0, ge=0.0, le=1.0)

    class Config:
        use_enum_values = True


# ================================================================================
# AUTONOMOUS OPERATION OUTPUTS  
# ================================================================================

class AutonomousTrigger(str, Enum):
    """Autonomous operation triggers"""
    SCHEDULED = "scheduled"
    EVENT_DRIVEN = "event_driven"
    THRESHOLD = "threshold"
    USER_INITIATED = "user_initiated"
    SYSTEM_MAINTENANCE = "system_maintenance"


class AutonomousOperationType(str, Enum):
    """Types of autonomous operations"""
    THINKING = "autonomous_thinking"
    PATTERN_DISCOVERY = "pattern_discovery"
    INSIGHT_GENERATION = "insight_generation"
    MILESTONE_TRACKING = "milestone_tracking"
    LIFE_EVENT_DETECTION = "life_event_detection"
    MEMORY_MAINTENANCE = "memory_maintenance"


class AutonomousInsight(BaseModel):
    """Structured autonomous insight"""
    insight_id: str = Field(..., description="Unique insight identifier")
    insight_type: AutonomousOperationType = Field(..., description="Type of insight")
    title: str = Field(..., description="Insight title")
    content: str = Field(..., description="Insight content")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    target_user: str = Field(..., description="User the insight is about")
    generated_at: datetime = Field(default_factory=datetime.now)
    
    # Insight metadata
    evidence: List[str] = Field(default_factory=list, description="Supporting evidence")
    actionable_suggestions: List[str] = Field(default_factory=list)
    importance_level: str = Field(default="medium")
    
    class Config:
        use_enum_values = True


class AutonomousOperationOutput(BaseAgentOutput):
    """Output for autonomous operations"""
    agent_type: Literal[AgentType.AUTONOMOUS_ROUTER] = Field(default=AgentType.AUTONOMOUS_ROUTER)
    
    # Operation details
    operation_type: AutonomousOperationType = Field(..., description="Type of autonomous operation")
    trigger_source: AutonomousTrigger = Field(..., description="What triggered the operation")
    target_user: str = Field(..., description="User being analyzed")
    
    # Results
    operation_result: str = Field(..., description="Result of the operation")
    insights_generated: List[AutonomousInsight] = Field(default_factory=list)
    patterns_discovered: List[str] = Field(default_factory=list)
    
    # Analysis details
    memory_analysis_performed: bool = Field(default=False)
    research_performed: bool = Field(default=False)
    synthesis_quality: SynthesisQuality = Field(default=SynthesisQuality.GOOD)
    
    # Broadcasting and storage
    insight_stored: bool = Field(default=False)
    broadcast_sent: bool = Field(default=False)
    broadcast_data: Optional[Dict[str, Any]] = Field(None)

    class Config:
        use_enum_values = True


# ================================================================================
# OUTPUT VALIDATION AND PARSING UTILITIES
# ================================================================================

class OutputParser:
    """Utility class for parsing and validating agent outputs"""
    
    AGENT_OUTPUT_MAPPING = {
        AgentType.MEMORY_READER: MemoryReaderOutput,
        AgentType.MEMORY_WRITER: MemoryWriterOutput,
        AgentType.KNOWLEDGE_AGENT: KnowledgeAgentOutput,
        AgentType.ORGANIZER_AGENT: OrganizerAgentOutput,
        AgentType.AUTONOMOUS_ROUTER: WorkflowExecutionOutput
    }
    
    @classmethod
    def parse_agent_output(cls, agent_type: str, output_data: Dict[str, Any]) -> BaseAgentOutput:
        """Parse raw agent output into structured schema"""
        try:
            # Map string to enum
            agent_enum = AgentType(agent_type)
            schema_class = cls.AGENT_OUTPUT_MAPPING.get(agent_enum)
            
            if not schema_class:
                raise ValueError(f"No schema defined for agent type: {agent_type}")
            
            # Parse and validate
            return schema_class(**output_data)
            
        except Exception as e:
            # Fallback to base schema if parsing fails
            try:
                agent_enum = AgentType(agent_type)
            except:
                agent_enum = AgentType.MEMORY_READER  # Default fallback
                
            return BaseAgentOutput(
                agent_name=output_data.get("agent_name", "unknown"),
                agent_type=agent_enum,
                processing_model=ProcessingModel.LOCAL_TRANSFORMERS,
                operation_status=OperationStatus.ERROR,
                error_details=f"Schema parsing failed: {str(e)}",
                metadata=output_data
            )
    
    @classmethod
    def validate_output_structure(cls, output: BaseAgentOutput) -> Dict[str, Any]:
        """Validate output structure and return validation results"""
        try:
            # Try to serialize to ensure all fields are valid
            serialized = output.dict()
            
            return {
                "valid": True,
                "agent_type": output.agent_type,
                "schema_version": "1.0",
                "field_count": len(serialized),
                "validation_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "validation_timestamp": datetime.now().isoformat()
            }


# ================================================================================
# SERIALIZATION HELPERS
# ================================================================================

def serialize_agent_output(output: BaseAgentOutput) -> str:
    """Serialize agent output to JSON string"""
    import json
    from datetime import datetime
    
    def datetime_serializer(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    return json.dumps(output.dict(), indent=2, ensure_ascii=False, default=datetime_serializer)


def deserialize_agent_output(json_str: str, agent_type: str) -> BaseAgentOutput:
    """Deserialize JSON string to agent output schema"""
    data = json.loads(json_str)
    return OutputParser.parse_agent_output(agent_type, data)


# Export all schemas for easy importing
__all__ = [
    # Base schemas
    "BaseAgentOutput", "AgentType", "ProcessingModel", "OperationStatus",
    
    # Agent-specific outputs
    "MemoryReaderOutput", "MemoryItem", "MemoryType",
    "KnowledgeAgentOutput", "KnowledgeSearchResult", "SearchResultSource", 
    "OrganizerAgentOutput", "SynthesisQuality", "ContextQualityMetrics",
    "MemoryWriterOutput", "ExtractedFact", "FactType", "MemoryStorageStats",
    
    # Workflow outputs
    "WorkflowExecutionOutput", "WorkflowPattern", "ParallelExecutionMetrics",
    "AutonomousOperationOutput", "AutonomousInsight", "AutonomousOperationType", "AutonomousTrigger",
    
    # Utilities
    "OutputParser", "serialize_agent_output", "deserialize_agent_output"
]