/**
 * Phase 5: Auto-generated TypeScript Interfaces
 * Generated from Pydantic schemas for structured agent outputs
 * Generated at: 2025-07-29T22:12:27.727026
 * DO NOT EDIT MANUALLY - This file is auto-generated
 */

// ================================================================================
// ENUMS
// ================================================================================

export enum AgentType {
  MEMORY_READER = 'memory_reader',
  MEMORY_WRITER = 'memory_writer',
  KNOWLEDGE_AGENT = 'knowledge_agent',
  ORGANIZER_AGENT = 'organizer_agent',
  AUTONOMOUS_ROUTER = 'autonomous_router',
}

export enum ProcessingModel {
  LOCAL_TRANSFORMERS = 'local_transformers_only',
  EXTERNAL_LLM = 'external_llm_only',
  HYBRID = 'hybrid',
  CACHED_RESULT = 'cached_result',
}

export enum OperationStatus {
  SUCCESS = 'success',
  PARTIAL_SUCCESS = 'partial_success',
  FAILED = 'failed',
  ERROR = 'error',
}

export enum MemoryType {
  SHORT_TERM = 'short_term',
  LONG_TERM = 'long_term',
  WORKING = 'working',
  SESSION = 'session',
}

export enum SearchResultSource {
  WIKIPEDIA = 'wikipedia',
  WIKIDATA = 'wikidata',
  COMBINED = 'wikipedia_wikidata',
  CACHED = 'cached',
}

export enum SynthesisQuality {
  EXCELLENT = 'excellent',
  GOOD = 'good',
  ADEQUATE = 'adequate',
  POOR = 'poor',
}

export enum FactType {
  PERSONAL_INFO = 'personal_information',
  PREFERENCE = 'preference',
  GOAL = 'goal',
  EXPERIENCE = 'experience',
  SKILL = 'skill',
  STATEMENT = 'statement',
  ENTITY = 'entity',
  UNKNOWN = 'unknown',
}

export enum WorkflowPattern {
  SIMPLE_MEMORY_ONLY = 'simple_memory_only',
  RESEARCH_ENHANCED = 'research_enhanced',
  COMPLEX_REASONING = 'complex_reasoning',
  PARALLEL_EXECUTION = 'parallel_execution',
  AUTONOMOUS_OPERATION = 'autonomous_operation',
}

export enum AutonomousTrigger {
  SCHEDULED = 'scheduled',
  EVENT_DRIVEN = 'event_driven',
  THRESHOLD = 'threshold',
  USER_INITIATED = 'user_initiated',
  SYSTEM_MAINTENANCE = 'system_maintenance',
}

export enum AutonomousOperationType {
  THINKING = 'autonomous_thinking',
  PATTERN_DISCOVERY = 'pattern_discovery',
  INSIGHT_GENERATION = 'insight_generation',
  MILESTONE_TRACKING = 'milestone_tracking',
  LIFE_EVENT_DETECTION = 'life_event_detection',
  MEMORY_MAINTENANCE = 'memory_maintenance',
}


// ================================================================================
// INTERFACES
// ================================================================================

export interface BaseAgentOutput {
  /** Name of the agent */
  agent_name?: string;
  /** Type of agent */
  agent_type?: AgentType;
  /** Processing model used */
  processing_model?: ProcessingModel;
  operation_status?: OperationStatus;
  timestamp?: string;
  /** Processing time in milliseconds */
  processing_time_ms?: number;
  metadata?: string;
  /** Error details if operation failed */
  error_details: string;
}

export interface MemoryItem {
  /** Memory content */
  content?: string;
  importance_score: number;
  /** Type of memory */
  memory_type?: MemoryType;
  /** When memory was stored */
  stored_at?: string;
  /** When memory expires (for short-term) */
  ttl_expires: string;
  /** Source of the memory */
  source: string;
}

export interface MemoryReaderOutput extends BaseAgentOutput {
  /** Name of the agent */
  agent_name?: string;
  agent_type?: Literal;
  processing_model?: Literal;
  operation_status?: OperationStatus;
  timestamp?: string;
  /** Processing time in milliseconds */
  processing_time_ms?: number;
  metadata?: string;
  /** Error details if operation failed */
  error_details: string;
  /** Summary of retrieved context */
  context_summary?: string;
  memories_found?: number;
  short_term_memories?: core.output_schemas.MemoryItem[];
  long_term_memories?: core.output_schemas.MemoryItem[];
  working_memories?: core.output_schemas.MemoryItem[];
  /** Original search query */
  search_query?: string;
  retrieval_method?: string;
  similarity_threshold?: number;
  context_relevance_score?: number;
  /** Coverage by memory type */
  memory_coverage?: string;
}

export interface KnowledgeSearchResult {
  /** Search result content */
  content?: string;
  /** Source of information */
  source?: SearchResultSource;
  relevance_score?: number;
  confidence?: number;
  /** Source URL if available */
  url: string;
}

export interface KnowledgeAgentOutput extends BaseAgentOutput {
  /** Name of the agent */
  agent_name?: string;
  agent_type?: Literal;
  processing_model?: Literal;
  operation_status?: OperationStatus;
  timestamp?: string;
  /** Processing time in milliseconds */
  processing_time_ms?: number;
  metadata?: string;
  /** Error details if operation failed */
  error_details: string;
  /** Summary of research findings */
  knowledge_summary?: string;
  search_results?: core.output_schemas.KnowledgeSearchResult[];
  /** Original search query */
  search_query?: string;
  /** Type of research conducted */
  research_type?: string;
  sources_consulted?: core.output_schemas.SearchResultSource[];
  search_completed?: boolean;
  results_found?: number;
  average_relevance?: number;
  /** Whether result was retrieved from cache */
  was_cached?: boolean;
}

export interface ContextQualityMetrics {
  memory_context_quality?: number;
  knowledge_context_quality?: number;
  overall_quality?: number;
  context_completeness?: number;
}

export interface OrganizerAgentOutput extends BaseAgentOutput {
  /** Name of the agent */
  agent_name?: string;
  agent_type?: Literal;
  processing_model?: Literal;
  operation_status?: OperationStatus;
  timestamp?: string;
  /** Processing time in milliseconds */
  processing_time_ms?: number;
  metadata?: string;
  /** Error details if operation failed */
  error_details: string;
  /** Final synthesized response */
  response?: string;
  synthesis_successful?: boolean;
  synthesis_quality?: SynthesisQuality;
  context_quality?: ContextQualityMetrics;
  memory_context_used?: boolean;
  knowledge_context_used?: boolean;
  /** Specific LLM model used */
  llm_model_used: string;
  /** Token usage statistics */
  token_usage?: string;
  temperature?: number;
  /** Reasoning process steps */
  reasoning_steps: string;
  confidence_score?: number;
}

export interface ExtractedFact {
  /** Fact content */
  content?: string;
  /** Type of fact */
  fact_type?: FactType;
  importance_score?: number;
  confidence?: number;
  /** Source of fact (user_message/ai_response) */
  source?: string;
  /** Where fact was stored (short_term/long_term) */
  storage_destination?: string;
  /** TTL for short-term storage */
  ttl_seconds: number;
  extracted_at?: string;
}

export interface MemoryStorageStats {
  facts_extracted?: number;
  short_term_stored?: number;
  long_term_stored?: number;
  session_stored?: boolean;
  working_memory_updated?: boolean;
  duplicates_found?: number;
  storage_errors?: number;
}

export interface MemoryWriterOutput extends BaseAgentOutput {
  /** Name of the agent */
  agent_name?: string;
  agent_type?: Literal;
  processing_model?: Literal;
  operation_status?: OperationStatus;
  timestamp?: string;
  /** Processing time in milliseconds */
  processing_time_ms?: number;
  metadata?: string;
  /** Error details if operation failed */
  error_details: string;
  storage_stats?: MemoryStorageStats;
  extracted_facts?: core.output_schemas.ExtractedFact[];
  user_message_processed?: boolean;
  ai_response_processed?: boolean;
  conversation_metadata?: string;
  storage_constraints_applied: string;
  long_term_storage_blocked?: boolean;
  importance_cap_applied: number;
}

export interface ParallelExecutionMetrics {
  parallel_agents?: string;
  execution_phase?: string;
  speedup_factor?: number;
  concurrent_time_saved_ms?: number;
}

export interface WorkflowExecutionOutput extends BaseAgentOutput {
  /** Name of the agent */
  agent_name?: string;
  agent_type?: Literal;
  /** Processing model used */
  processing_model?: ProcessingModel;
  operation_status?: OperationStatus;
  timestamp?: string;
  /** Processing time in milliseconds */
  processing_time_ms?: number;
  metadata?: string;
  /** Error details if operation failed */
  error_details: string;
  /** Final workflow response */
  final_response?: string;
  /** Execution pattern used */
  workflow_pattern?: WorkflowPattern;
  agents_executed?: string;
  execution_order?: string;
  total_processing_time_ms?: number;
  complexity_score?: number;
  parallel_execution: Optional;
  memory_context_available?: boolean;
  knowledge_context_available?: boolean;
  research_performed?: boolean;
  response_quality_score?: number;
  user_satisfaction_predicted?: number;
}

export interface AutonomousInsight {
  /** Unique insight identifier */
  insight_id?: string;
  /** Type of insight */
  insight_type?: AutonomousOperationType;
  /** Insight title */
  title?: string;
  /** Insight content */
  content?: string;
  confidence?: number;
  /** User the insight is about */
  target_user?: string;
  generated_at?: string;
  /** Supporting evidence */
  evidence?: string;
  actionable_suggestions?: string;
  importance_level?: string;
}

export interface AutonomousOperationOutput extends BaseAgentOutput {
  /** Name of the agent */
  agent_name?: string;
  agent_type?: Literal;
  /** Processing model used */
  processing_model?: ProcessingModel;
  operation_status?: OperationStatus;
  timestamp?: string;
  /** Processing time in milliseconds */
  processing_time_ms?: number;
  metadata?: string;
  /** Error details if operation failed */
  error_details: string;
  /** Type of autonomous operation */
  operation_type?: AutonomousOperationType;
  /** What triggered the operation */
  trigger_source?: AutonomousTrigger;
  /** User being analyzed */
  target_user?: string;
  /** Result of the operation */
  operation_result?: string;
  insights_generated?: core.output_schemas.AutonomousInsight[];
  patterns_discovered?: string;
  memory_analysis_performed?: boolean;
  research_performed?: boolean;
  synthesis_quality?: SynthesisQuality;
  insight_stored?: boolean;
  broadcast_sent?: boolean;
  broadcast_data: string;
}


// ================================================================================
// UTILITY TYPES
// ================================================================================

export type AgentOutputUnion = MemoryReaderOutput | KnowledgeAgentOutput | OrganizerAgentOutput | MemoryWriterOutput;

export type WorkflowOutputUnion = WorkflowExecutionOutput | AutonomousOperationOutput;

export interface ApiResponse<T = any> {
  response: string;
  metadata: Record<string, any>;
  timestamp: string;
  user_name?: string;
  structured_output?: T;
}

export interface StructuredApiResponse extends ApiResponse {
  structured_output: AgentOutputUnion | WorkflowOutputUnion;
}
