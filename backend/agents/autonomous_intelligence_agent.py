"""
Autonomous Intelligence Agent - Autonomous Thinking & Strategic Planning
Handles continuous reasoning, pattern discovery, and strategic insights
"""

import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import json
import logging

from autogen import ConversableAgent

from memory.autonomous_memory import AutonomousMemorySystem
from core.config import AssistantConfig
from core.transformers_service import TransformersService
from utils.serialization import safe_serialize

logger = logging.getLogger(__name__)

class AutonomousIntelligenceAgent(ConversableAgent):
    """
    Intelligence Agent - Autonomous thinking, pattern discovery, and strategic planning
    
    Responsibilities:
    - Continuous autonomous thinking cycles
    - Pattern recognition and insight generation
    - Strategic planning and timeline creation
    - Long-term goal tracking and life event planning
    - Proactive insight generation and recommendations
    
    Autonomous Functions:
    - Hourly thinking cycles
    - Weekly insight generation
    - Life event timeline planning
    - Background pattern analysis
    """
    
    def __init__(
        self,
        name: str = "intelligence_agent",
        memory_system: Optional[AutonomousMemorySystem] = None,
        config: Optional[AssistantConfig] = None,
        transformers_service: Optional[TransformersService] = None,
        **kwargs
    ):
        """Initialize Intelligence Agent with autonomous thinking capabilities"""
        
        # System message for AutoGen
        system_message = """You are the Intelligence Agent, specialized in autonomous thinking, pattern discovery, and strategic planning.

Your core responsibilities:
1. Conduct continuous autonomous thinking cycles to discover insights
2. Analyze user patterns and behaviors for personalized recommendations
3. Create strategic plans, timelines, and goal tracking systems
4. Generate proactive insights for life events and important milestones
5. Discover connections and patterns across user data and conversations

Key capabilities:
- Pattern recognition and trend analysis
- Strategic planning and timeline creation
- Life event tracking and milestone planning
- Autonomous insight generation
- Goal setting and progress tracking
- Predictive recommendations based on user patterns

Autonomous thinking focus areas:
- Life event planning (pregnancy, health, career, relationships)
- Personal growth and development recommendations
- Timeline-based strategic planning
- Pattern-based behavioral insights
- Proactive problem-solving and opportunity identification

Always think strategically, consider long-term implications, and provide actionable insights that help users achieve their goals."""

        # Initialize AutoGen ConversableAgent with disabled LLM client
        # We handle LLM calls manually through our existing system
        super().__init__(
            name=name,
            system_message=system_message,
            human_input_mode="NEVER",
            code_execution_config=False,
            llm_config=False,  # Disable AutoGen LLM client initialization
            **kwargs
        )
        
        # Core systems
        self.memory_system = memory_system
        self.config = config or AssistantConfig()
        self.transformers_service = transformers_service
        
        # Autonomous thinking state
        self.thinking_cycles = 0
        self.last_thinking_cycle = None
        self.insights_generated = []
        self.active_plans = {}
        self.pattern_discoveries = []
        
        # Life event tracking
        self.life_events = {}
        self.timeline_plans = {}
        
        logger.info(f"AutoGen Intelligence Agent initialized")
    
    async def autonomous_thinking_cycle(self) -> Dict[str, Any]:
        """
        Main autonomous thinking cycle - runs independently
        Analyzes patterns, generates insights, updates plans
        """
        try:
            logger.info("Starting autonomous thinking cycle")
            cycle_start = datetime.now()
            
            # Increment cycle counter
            self.thinking_cycles += 1
            self.last_thinking_cycle = cycle_start
            
            # Gather data for analysis
            analysis_data = await self._gather_thinking_data()
            
            # Pattern discovery
            patterns = await self._discover_patterns(analysis_data)
            
            # Generate insights
            insights = await self._generate_insights(patterns, analysis_data)
            
            # Update strategic plans
            plan_updates = await self._update_strategic_plans(insights)
            
            # Life event analysis
            life_event_insights = await self._analyze_life_events(analysis_data)
            
            # Store insights and patterns
            await self._store_autonomous_insights(insights, patterns, life_event_insights)
            
            cycle_end = datetime.now()
            cycle_duration = (cycle_end - cycle_start).total_seconds()
            
            result = {
                "cycle_number": self.thinking_cycles,
                "timestamp": cycle_start.isoformat(),
                "duration_seconds": cycle_duration,
                "patterns_discovered": len(patterns),
                "insights_generated": len(insights),
                "plans_updated": len(plan_updates),
                "life_events_analyzed": len(life_event_insights),
                "agent_name": self.name,
                "status": "completed"
            }
            
            logger.info(f"Autonomous thinking cycle completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error in autonomous thinking cycle: {e}")
            return {
                "cycle_number": self.thinking_cycles,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "agent_name": self.name,
                "status": "error"
            }
    
    async def _gather_thinking_data(self) -> Dict[str, Any]:
        """Gather data for autonomous thinking analysis"""
        
        data = {
            "recent_conversations": [],
            "user_patterns": {},
            "life_events": [],
            "goals": [],
            "memory_insights": {}
        }
        
        if not self.memory_system:
            return data
        
        try:
            # Get recent memory data (last 7 days)
            recent_memories = await self.memory_system.get_recent_memories(days=7)
            
            # Categorize memories
            for memory in recent_memories:
                memory_type = memory.get("memory_type", "unknown")
                
                if memory_type == "episodic":
                    data["recent_conversations"].append(memory)
                elif memory_type == "semantic":
                    if "life_event" in memory.get("content", "").lower():
                        data["life_events"].append(memory)
                elif memory_type == "prospective":
                    data["goals"].append(memory)
            
            # Get user behavioral patterns
            data["user_patterns"] = await self._analyze_user_patterns(recent_memories)
            
        except Exception as e:
            logger.warning(f"Error gathering thinking data: {e}")
        
        return data
    
    async def _analyze_user_patterns(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze user behavioral patterns from memory data"""
        
        patterns = {
            "conversation_frequency": {},
            "topic_interests": {},
            "time_patterns": {},
            "emotional_patterns": {},
            "goal_patterns": {}
        }
        
        try:
            # Analyze conversation frequency
            conversation_times = []
            topics = []
            
            for memory in memories:
                timestamp = memory.get("timestamp")
                content = memory.get("content", "").lower()
                
                if timestamp:
                    conversation_times.append(timestamp)
                
                # Extract topics (simple keyword analysis)
                if "learn" in content or "study" in content:
                    topics.append("learning")
                if "health" in content or "medical" in content:
                    topics.append("health")
                if "work" in content or "job" in content:
                    topics.append("career")
                if "family" in content or "relationship" in content:
                    topics.append("relationships")
            
            # Calculate patterns
            patterns["conversation_frequency"] = {
                "total_conversations": len(conversation_times),
                "avg_per_day": len(conversation_times) / 7 if conversation_times else 0
            }
            
            patterns["topic_interests"] = {
                "learning": topics.count("learning"),
                "health": topics.count("health"),
                "career": topics.count("career"),
                "relationships": topics.count("relationships")
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing user patterns: {e}")
        
        return patterns
    
    async def _discover_patterns(self, analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Discover meaningful patterns from analysis data"""
        
        patterns = []
        
        try:
            # Pattern 1: Conversation frequency trends
            conv_freq = analysis_data.get("user_patterns", {}).get("conversation_frequency", {})
            if conv_freq.get("avg_per_day", 0) > 2:
                patterns.append({
                    "pattern_type": "high_engagement",
                    "description": "User shows high engagement with frequent conversations",
                    "confidence": 0.8,
                    "recommendation": "Continue providing rich, detailed responses and proactive insights"
                })
            
            # Pattern 2: Topic interest analysis
            topic_interests = analysis_data.get("user_patterns", {}).get("topic_interests", {})
            dominant_topic = max(topic_interests.items(), key=lambda x: x[1]) if topic_interests else None
            
            if dominant_topic and dominant_topic[1] > 2:
                patterns.append({
                    "pattern_type": "topic_focus",
                    "description": f"Strong interest in {dominant_topic[0]} with {dominant_topic[1]} mentions",
                    "confidence": 0.7,
                    "recommendation": f"Provide more proactive insights and resources related to {dominant_topic[0]}"
                })
            
            # Pattern 3: Life event detection
            life_events = analysis_data.get("life_events", [])
            if life_events:
                recent_events = [e for e in life_events if self._is_recent_event(e)]
                if recent_events:
                    patterns.append({
                        "pattern_type": "life_event_active",
                        "description": f"Active life events detected: {len(recent_events)} recent events",
                        "confidence": 0.9,
                        "recommendation": "Provide timeline-based planning and milestone tracking"
                    })
            
        except Exception as e:
            logger.warning(f"Error discovering patterns: {e}")
        
        return patterns
    
    async def _generate_insights(self, patterns: List[Dict[str, Any]], analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable insights from discovered patterns"""
        
        insights = []
        
        try:
            for pattern in patterns:
                pattern_type = pattern.get("pattern_type")
                
                if pattern_type == "high_engagement":
                    insights.append({
                        "insight_type": "engagement_optimization",
                        "title": "Optimize for High Engagement",
                        "description": "User is highly engaged. Consider providing daily insights and proactive recommendations.",
                        "actionable_steps": [
                            "Generate daily personalized insights",
                            "Provide proactive recommendations",
                            "Offer advanced planning features"
                        ],
                        "priority": "high",
                        "timestamp": datetime.now().isoformat()
                    })
                
                elif pattern_type == "topic_focus":
                    topic = pattern.get("description", "").split("interest in ")[1].split(" with")[0] if "interest in " in pattern.get("description", "") else "unknown"
                    insights.append({
                        "insight_type": "personalized_content",
                        "title": f"Focus on {topic.title()} Content",
                        "description": f"User shows strong interest in {topic}. Tailor responses and insights accordingly.",
                        "actionable_steps": [
                            f"Research latest trends in {topic}",
                            f"Provide {topic}-related recommendations",
                            f"Create {topic} learning timeline"
                        ],
                        "priority": "medium",
                        "timestamp": datetime.now().isoformat()
                    })
                
                elif pattern_type == "life_event_active":
                    insights.append({
                        "insight_type": "life_event_planning",
                        "title": "Active Life Event Planning",
                        "description": "Recent life events detected. Provide structured planning and milestone tracking.",
                        "actionable_steps": [
                            "Create event-specific timelines",
                            "Set up milestone reminders",
                            "Provide relevant resources and guidance"
                        ],
                        "priority": "high",
                        "timestamp": datetime.now().isoformat()
                    })
            
            # Generate general insights if no specific patterns
            if not insights:
                insights.append({
                    "insight_type": "general_optimization",
                    "title": "Continue Quality Assistance",
                    "description": "Maintain current level of service and look for opportunities to add value.",
                    "actionable_steps": [
                        "Monitor for emerging patterns",
                        "Provide consistent, helpful responses",
                        "Look for proactive assistance opportunities"
                    ],
                    "priority": "low",
                    "timestamp": datetime.now().isoformat()
                })
            
        except Exception as e:
            logger.warning(f"Error generating insights: {e}")
        
        return insights
    
    async def _update_strategic_plans(self, insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Update strategic plans based on new insights"""
        
        plan_updates = []
        
        try:
            for insight in insights:
                if insight.get("priority") == "high":
                    plan_id = f"plan_{len(self.active_plans) + 1}"
                    plan = {
                        "plan_id": plan_id,
                        "based_on_insight": insight.get("insight_type"),
                        "title": insight.get("title"),
                        "steps": insight.get("actionable_steps", []),
                        "status": "active",
                        "created": datetime.now().isoformat(),
                        "priority": insight.get("priority")
                    }
                    
                    self.active_plans[plan_id] = plan
                    plan_updates.append(plan)
            
        except Exception as e:
            logger.warning(f"Error updating strategic plans: {e}")
        
        return plan_updates
    
    async def _analyze_life_events(self, analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze life events and generate specific insights"""
        
        life_event_insights = []
        
        try:
            life_events = analysis_data.get("life_events", [])
            
            for event in life_events:
                content = event.get("content", "").lower()
                
                # Pregnancy planning
                if "pregnant" in content or "pregnancy" in content:
                    days_since = self._calculate_days_since_event(event)
                    life_event_insights.append({
                        "event_type": "pregnancy",
                        "days_since_detected": days_since,
                        "current_phase": self._get_pregnancy_phase(days_since),
                        "recommendations": self._get_pregnancy_recommendations(days_since),
                        "timeline_milestones": self._get_pregnancy_timeline(days_since)
                    })
                
                # Learning goals
                elif "learn" in content or "study" in content:
                    life_event_insights.append({
                        "event_type": "learning_goal",
                        "description": "Active learning interests detected",
                        "recommendations": [
                            "Create structured learning timeline",
                            "Set up progress tracking",
                            "Find relevant resources and courses"
                        ]
                    })
            
        except Exception as e:
            logger.warning(f"Error analyzing life events: {e}")
        
        return life_event_insights
    
    async def _store_autonomous_insights(self, insights: List[Dict[str, Any]], patterns: List[Dict[str, Any]], life_event_insights: List[Dict[str, Any]]) -> None:
        """Store generated insights in memory system"""
        
        if not self.memory_system:
            return
        
        try:
            # Store insights
            for insight in insights:
                await self.memory_system.store_memory(
                    "system",
                    f"Autonomous insight: {insight.get('title')} - {insight.get('description')}",
                    "procedural"
                )
            
            # Store patterns
            for pattern in patterns:
                await self.memory_system.store_memory(
                    "system",
                    f"Pattern discovered: {pattern.get('description')}",
                    "semantic"
                )
            
            # Update internal storage
            self.insights_generated.extend(insights)
            self.pattern_discoveries.extend(patterns)
            
            # Keep only recent insights (last 100)
            if len(self.insights_generated) > 100:
                self.insights_generated = self.insights_generated[-100:]
            
        except Exception as e:
            logger.warning(f"Error storing autonomous insights: {e}")
    
    def _is_recent_event(self, event: Dict[str, Any]) -> bool:
        """Check if event is recent (within last 30 days)"""
        try:
            timestamp = event.get("timestamp")
            if timestamp:
                event_date = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                return (datetime.now() - event_date).days <= 30
        except:
            pass
        return False
    
    def _calculate_days_since_event(self, event: Dict[str, Any]) -> int:
        """Calculate days since event occurred"""
        try:
            timestamp = event.get("timestamp")
            if timestamp:
                event_date = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                return (datetime.now() - event_date).days
        except:
            pass
        return 0
    
    def _get_pregnancy_phase(self, days_since: int) -> str:
        """Get pregnancy phase based on days since announcement"""
        if days_since < 7:
            return "Early announcement (Week 1)"
        elif days_since < 84:  # 12 weeks
            return "First trimester"
        elif days_since < 196:  # 28 weeks
            return "Second trimester"
        else:
            return "Third trimester"
    
    def _get_pregnancy_recommendations(self, days_since: int) -> List[str]:
        """Get pregnancy recommendations based on timeline"""
        if days_since < 7:
            return [
                "Start prenatal vitamins with folic acid",
                "Schedule first prenatal appointment",
                "Begin tracking pregnancy symptoms",
                "Review dietary and lifestyle guidelines"
            ]
        elif days_since < 84:
            return [
                "Continue prenatal vitamins",
                "Attend regular prenatal checkups",
                "Consider sharing news with close family",
                "Research childbirth classes"
            ]
        else:
            return [
                "Prepare nursery and baby essentials",
                "Attend childbirth preparation classes",
                "Create birth plan",
                "Prepare for maternity leave"
            ]
    
    def _get_pregnancy_timeline(self, days_since: int) -> List[Dict[str, Any]]:
        """Get pregnancy timeline milestones"""
        base_date = datetime.now() - timedelta(days=days_since)
        
        milestones = [
            {"week": 8, "milestone": "First prenatal appointment", "date": (base_date + timedelta(weeks=8)).isoformat()},
            {"week": 12, "milestone": "End of first trimester", "date": (base_date + timedelta(weeks=12)).isoformat()},
            {"week": 20, "milestone": "Anatomy scan", "date": (base_date + timedelta(weeks=20)).isoformat()},
            {"week": 28, "milestone": "Third trimester begins", "date": (base_date + timedelta(weeks=28)).isoformat()},
            {"week": 36, "milestone": "Baby considered full-term", "date": (base_date + timedelta(weeks=36)).isoformat()},
            {"week": 40, "milestone": "Expected due date", "date": (base_date + timedelta(weeks=40)).isoformat()}
        ]
        
        return milestones
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        
        return {
            "agent_name": self.name,
            "status": "active",
            "capabilities": [
                "autonomous_thinking",
                "pattern_discovery",
                "strategic_planning",
                "life_event_tracking",
                "insight_generation"
            ],
            "thinking_cycles_completed": self.thinking_cycles,
            "last_thinking_cycle": self.last_thinking_cycle.isoformat() if self.last_thinking_cycle else None,
            "insights_generated": len(self.insights_generated),
            "active_plans": len(self.active_plans),
            "patterns_discovered": len(self.pattern_discoveries),
            "timestamp": datetime.now().isoformat()
        }