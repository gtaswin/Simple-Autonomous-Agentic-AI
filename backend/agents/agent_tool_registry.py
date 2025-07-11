"""
Agent Tool Registry - Specialized Data Access Control for 3-Agent Architecture

This registry manages data source access permissions and tool availability
for each specialized agent, ensuring privacy protection and data isolation.
"""

from typing import Dict, List, Any, Set, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio


class DataSourceType(str, Enum):
    """Types of data sources available to agents"""
    # External sources (Research Agent only)
    TAVILY_API = "tavily_api"
    WEB_SEARCH = "web_search" 
    NEWS_APIS = "news_apis"
    REAL_TIME_DATA = "real_time_data"
    ACADEMIC_SEARCH = "academic_search"
    
    # Internal sources (Memory Agent only)
    REDIS_MEMORY = "redis_memory"
    QDRANT_MEMORY = "qdrant_memory"
    USER_PROFILES = "user_profiles"
    CONVERSATION_HISTORY = "conversation_history"
    BEHAVIORAL_PATTERNS = "behavioral_patterns"
    PERSONAL_PREFERENCES = "personal_preferences"
    
    # Coordination sources (Coordinator Agent only)
    AGENT_OUTPUTS = "agent_outputs"
    STRATEGIC_PLANNING = "strategic_planning"
    DECISION_SYNTHESIS = "decision_synthesis"
    RESOURCE_MANAGEMENT = "resource_management"


@dataclass
class AgentPermissions:
    """Permissions and restrictions for an agent"""
    agent_name: str
    allowed_sources: Set[DataSourceType]
    restricted_sources: Set[DataSourceType]
    tool_permissions: Set[str]
    privacy_level: str  # "public", "internal", "coordinator"
    description: str


class AgentToolRegistry:
    """Central registry for agent tool permissions and data access control"""
    
    def __init__(self):
        self.agent_permissions: Dict[str, AgentPermissions] = {}
        self.data_source_tools: Dict[DataSourceType, List[str]] = {}
        self.privacy_violations: List[Dict[str, Any]] = []
        
        # Initialize agent permissions
        self._setup_agent_permissions()
        self._setup_data_source_mappings()
    
    def _setup_agent_permissions(self):
        """Define permissions for each specialized agent"""
        
        # Research Agent - External World Knowledge Only
        self.agent_permissions["research_agent"] = AgentPermissions(
            agent_name="research_agent",
            allowed_sources={
                DataSourceType.TAVILY_API,
                DataSourceType.WEB_SEARCH,
                DataSourceType.NEWS_APIS,
                DataSourceType.REAL_TIME_DATA,
                DataSourceType.ACADEMIC_SEARCH
            },
            restricted_sources={
                DataSourceType.REDIS_MEMORY,
                DataSourceType.QDRANT_MEMORY,
                DataSourceType.USER_PROFILES,
                DataSourceType.CONVERSATION_HISTORY,
                DataSourceType.BEHAVIORAL_PATTERNS,
                DataSourceType.PERSONAL_PREFERENCES
            },
            tool_permissions={
                "tavily_search", "web_scraping", "news_api", "academic_search",
                "fact_verification", "source_validation"
            },
            privacy_level="public",
            description="External research and knowledge gathering - NO personal data access"
        )
        
        # Memory Agent - User Understanding Only
        self.agent_permissions["memory_agent"] = AgentPermissions(
            agent_name="memory_agent",
            allowed_sources={
                DataSourceType.REDIS_MEMORY,
                DataSourceType.QDRANT_MEMORY,
                DataSourceType.USER_PROFILES,
                DataSourceType.CONVERSATION_HISTORY,
                DataSourceType.BEHAVIORAL_PATTERNS,
                DataSourceType.PERSONAL_PREFERENCES
            },
            restricted_sources={
                DataSourceType.TAVILY_API,
                DataSourceType.WEB_SEARCH,
                DataSourceType.NEWS_APIS,
                DataSourceType.REAL_TIME_DATA,
                DataSourceType.ACADEMIC_SEARCH
            },
            tool_permissions={
                "memory_retrieval", "pattern_analysis", "user_profiling",
                "conversation_analysis", "preference_extraction", "goal_tracking"
            },
            privacy_level="internal",
            description="User context and personalization - NO external data access"
        )
        
        # Coordinator Agent - Decision Synthesis Only
        self.agent_permissions["coordinator_agent"] = AgentPermissions(
            agent_name="coordinator_agent",
            allowed_sources={
                DataSourceType.AGENT_OUTPUTS,
                DataSourceType.STRATEGIC_PLANNING,
                DataSourceType.DECISION_SYNTHESIS,
                DataSourceType.RESOURCE_MANAGEMENT
            },
            restricted_sources={
                # Coordinator doesn't directly access raw data sources
                DataSourceType.REDIS_MEMORY,
                DataSourceType.QDRANT_MEMORY,
                DataSourceType.TAVILY_API,
                DataSourceType.WEB_SEARCH
            },
            tool_permissions={
                "decision_synthesis", "strategic_planning", "resource_allocation",
                "priority_management", "response_generation"
            },
            privacy_level="coordinator",
            description="Strategic coordination - processes agent outputs only"
        )
    
    def _setup_data_source_mappings(self):
        """Map data sources to their corresponding tools"""
        
        self.data_source_tools = {
            # External sources
            DataSourceType.TAVILY_API: ["tavily_search", "deep_research"],
            DataSourceType.WEB_SEARCH: ["web_scraping", "browser_automation"],
            DataSourceType.NEWS_APIS: ["news_search", "real_time_news"],
            DataSourceType.REAL_TIME_DATA: ["live_data_feeds", "event_monitoring"],
            DataSourceType.ACADEMIC_SEARCH: ["scholarly_search", "paper_analysis"],
            
            # Internal sources
            DataSourceType.REDIS_MEMORY: ["working_memory_access", "session_retrieval"],
            DataSourceType.QDRANT_MEMORY: ["semantic_search", "vector_retrieval"],
            DataSourceType.USER_PROFILES: ["profile_analysis", "identity_extraction"],
            DataSourceType.CONVERSATION_HISTORY: ["chat_analysis", "context_extraction"],
            DataSourceType.BEHAVIORAL_PATTERNS: ["pattern_recognition", "habit_analysis"],
            DataSourceType.PERSONAL_PREFERENCES: ["preference_learning", "customization"],
            
            # Coordination sources
            DataSourceType.AGENT_OUTPUTS: ["output_processing", "result_synthesis"],
            DataSourceType.STRATEGIC_PLANNING: ["goal_planning", "strategy_development"],
            DataSourceType.DECISION_SYNTHESIS: ["decision_making", "choice_optimization"],
            DataSourceType.RESOURCE_MANAGEMENT: ["resource_allocation", "efficiency_optimization"]
        }
    
    def check_access_permission(self, agent_name: str, data_source: DataSourceType) -> bool:
        """Check if an agent has permission to access a data source"""
        
        if agent_name not in self.agent_permissions:
            self._log_violation(agent_name, "unknown_agent", str(data_source))
            return False
        
        permissions = self.agent_permissions[agent_name]
        
        # Check if explicitly allowed
        if data_source in permissions.allowed_sources:
            return True
        
        # Check if explicitly restricted
        if data_source in permissions.restricted_sources:
            self._log_violation(agent_name, "restricted_access", str(data_source))
            return False
        
        # Default deny for unlisted sources
        self._log_violation(agent_name, "unlisted_source", str(data_source))
        return False
    
    def check_tool_permission(self, agent_name: str, tool_name: str) -> bool:
        """Check if an agent has permission to use a specific tool"""
        
        if agent_name not in self.agent_permissions:
            return False
        
        permissions = self.agent_permissions[agent_name]
        return tool_name in permissions.tool_permissions
    
    def get_available_tools(self, agent_name: str) -> List[str]:
        """Get all tools available to a specific agent"""
        
        if agent_name not in self.agent_permissions:
            return []
        
        permissions = self.agent_permissions[agent_name]
        available_tools = list(permissions.tool_permissions)
        
        # Add tools from allowed data sources
        for source in permissions.allowed_sources:
            if source in self.data_source_tools:
                available_tools.extend(self.data_source_tools[source])
        
        return list(set(available_tools))  # Remove duplicates
    
    def get_restricted_sources(self, agent_name: str) -> List[str]:
        """Get data sources that are restricted for an agent"""
        
        if agent_name not in self.agent_permissions:
            return []
        
        permissions = self.agent_permissions[agent_name]
        return [source.value for source in permissions.restricted_sources]
    
    def validate_agent_request(self, agent_name: str, requested_sources: List[DataSourceType], 
                             requested_tools: List[str]) -> Dict[str, Any]:
        """Validate an agent's request for data sources and tools"""
        
        validation_result = {
            "agent_name": agent_name,
            "allowed_sources": [],
            "denied_sources": [],
            "allowed_tools": [],
            "denied_tools": [],
            "violations": [],
            "privacy_concerns": []
        }
        
        # Validate data sources
        for source in requested_sources:
            if self.check_access_permission(agent_name, source):
                validation_result["allowed_sources"].append(source.value)
            else:
                validation_result["denied_sources"].append(source.value)
                validation_result["violations"].append(f"Access denied to {source.value}")
        
        # Validate tools
        for tool in requested_tools:
            if self.check_tool_permission(agent_name, tool):
                validation_result["allowed_tools"].append(tool)
            else:
                validation_result["denied_tools"].append(tool)
                validation_result["violations"].append(f"Tool access denied: {tool}")
        
        # Check for privacy concerns
        if agent_name == "research_agent":
            personal_sources = [s for s in requested_sources if s in {
                DataSourceType.REDIS_MEMORY, DataSourceType.QDRANT_MEMORY,
                DataSourceType.USER_PROFILES, DataSourceType.CONVERSATION_HISTORY
            }]
            if personal_sources:
                validation_result["privacy_concerns"].append(
                    f"Research agent attempting to access personal data: {[s.value for s in personal_sources]}"
                )
        
        return validation_result
    
    def _log_violation(self, agent_name: str, violation_type: str, details: str):
        """Log access violations for security monitoring"""
        
        violation = {
            "timestamp": asyncio.get_event_loop().time(),
            "agent_name": agent_name,
            "violation_type": violation_type,
            "details": details,
            "severity": "high" if violation_type == "restricted_access" else "medium"
        }
        
        self.privacy_violations.append(violation)
        print(f"🚨 PRIVACY VIOLATION: {agent_name} - {violation_type} - {details}")
    
    def get_agent_capabilities(self, agent_name: str) -> Dict[str, Any]:
        """Get comprehensive capabilities for an agent"""
        
        if agent_name not in self.agent_permissions:
            return {"error": f"Unknown agent: {agent_name}"}
        
        permissions = self.agent_permissions[agent_name]
        
        return {
            "agent_name": agent_name,
            "description": permissions.description,
            "privacy_level": permissions.privacy_level,
            "data_sources": {
                "allowed": [source.value for source in permissions.allowed_sources],
                "restricted": [source.value for source in permissions.restricted_sources]
            },
            "tools": {
                "permitted": list(permissions.tool_permissions),
                "available": self.get_available_tools(agent_name)
            },
            "specialization": self._get_agent_specialization(agent_name)
        }
    
    def _get_agent_specialization(self, agent_name: str) -> str:
        """Get agent specialization description"""
        
        specializations = {
            "research_agent": "External world knowledge and real-time information gathering",
            "memory_agent": "User understanding, personalization, and behavioral analysis", 
            "coordinator_agent": "Strategic decision making and agent output synthesis"
        }
        
        return specializations.get(agent_name, "Unknown specialization")
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get overview of the entire agent tool registry system"""
        
        return {
            "total_agents": len(self.agent_permissions),
            "agents": list(self.agent_permissions.keys()),
            "data_source_types": len(DataSourceType),
            "privacy_violations": len(self.privacy_violations),
            "architecture": {
                "research_agent": {
                    "focus": "External data only",
                    "privacy": "No personal data access",
                    "sources": len(self.agent_permissions["research_agent"].allowed_sources)
                },
                "memory_agent": {
                    "focus": "User data only",
                    "privacy": "No external data access", 
                    "sources": len(self.agent_permissions["memory_agent"].allowed_sources)
                },
                "coordinator_agent": {
                    "focus": "Agent coordination",
                    "privacy": "Processed outputs only",
                    "sources": len(self.agent_permissions["coordinator_agent"].allowed_sources)
                }
            },
            "privacy_protection": {
                "data_isolation": "✅ Enforced",
                "access_control": "✅ Active",
                "violation_monitoring": "✅ Enabled"
            }
        }


# Global registry instance
agent_registry = AgentToolRegistry()


# Convenience functions
def check_agent_access(agent_name: str, data_source: DataSourceType) -> bool:
    """Quick access permission check"""
    return agent_registry.check_access_permission(agent_name, data_source)


def get_agent_tools(agent_name: str) -> List[str]:
    """Get available tools for an agent"""
    return agent_registry.get_available_tools(agent_name)


def validate_request(agent_name: str, sources: List[DataSourceType], tools: List[str]) -> Dict[str, Any]:
    """Validate an agent's resource request"""
    return agent_registry.validate_agent_request(agent_name, sources, tools)


if __name__ == "__main__":
    # Test the registry
    registry = AgentToolRegistry()
    
    print("🔐 Agent Tool Registry Test")
    print("=" * 50)
    
    # Test Research Agent permissions
    print("\n📡 Research Agent:")
    research_caps = registry.get_agent_capabilities("research_agent")
    print(f"Allowed sources: {len(research_caps['data_sources']['allowed'])}")
    print(f"Restricted sources: {len(research_caps['data_sources']['restricted'])}")
    
    # Test Memory Agent permissions
    print("\n🧠 Memory Agent:")
    memory_caps = registry.get_agent_capabilities("memory_agent")
    print(f"Allowed sources: {len(memory_caps['data_sources']['allowed'])}")
    print(f"Restricted sources: {len(memory_caps['data_sources']['restricted'])}")
    
    # Test Coordinator Agent permissions
    print("\n⚖️ Coordinator Agent:")
    coord_caps = registry.get_agent_capabilities("coordinator_agent")
    print(f"Allowed sources: {len(coord_caps['data_sources']['allowed'])}")
    print(f"Restricted sources: {len(coord_caps['data_sources']['restricted'])}")
    
    # Test privacy violation detection
    print("\n🚨 Privacy Protection Test:")
    research_violation = registry.check_access_permission("research_agent", DataSourceType.USER_PROFILES)
    memory_violation = registry.check_access_permission("memory_agent", DataSourceType.TAVILY_API)
    print(f"Research agent accessing user data: {research_violation} (should be False)")
    print(f"Memory agent accessing external APIs: {memory_violation} (should be False)")
    
    # System overview
    print("\n📊 System Overview:")
    overview = registry.get_system_overview()
    print(f"Total agents: {overview['total_agents']}")
    print(f"Privacy violations detected: {overview['privacy_violations']}")
    print(f"Architecture: 3-agent specialized system with data isolation")