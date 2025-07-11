"""
Autonomous Scheduler - Manages autonomous thinking cycles and proactive intelligence
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from core.autonomous_orchestrator import AutonomousOrchestrator
from memory.autonomous_memory import AutonomousMemorySystem
from api.websocket import AgentStreamManager

logger = logging.getLogger(__name__)

class AutonomousScheduler:
    """
    Manages autonomous thinking cycles and proactive intelligence generation
    
    Features:
    - Hourly thinking cycles
    - Weekly insight generation  
    - Life event milestone tracking
    - Adaptive scheduling based on user activity
    - Real-time broadcasting of insights
    """
    
    def __init__(
        self,
        autonomous_orchestrator: Optional[AutonomousOrchestrator] = None,
        memory_system: Optional[AutonomousMemorySystem] = None,
        websocket_manager: Optional[AgentStreamManager] = None
    ):
        """Initialize autonomous scheduler"""
        
        self.autonomous_orchestrator = autonomous_orchestrator
        self.memory_system = memory_system
        self.websocket_manager = websocket_manager
        
        self.scheduler = AsyncIOScheduler()
        self.running = False
        
        # Tracking
        self.thinking_cycles_completed = 0
        self.insights_generated = 0
        self.last_thinking_cycle = None
        self.last_insight_generation = None
        
        # Performance metrics
        self.cycle_performance = {
            "average_duration": 0.0,
            "success_rate": 0.0,
            "total_cycles": 0,
            "failed_cycles": 0
        }
        
        logger.info("Autonomous Scheduler initialized")
    
    async def start(self):
        """Start the autonomous scheduler"""
        
        if self.running:
            logger.warning("Autonomous scheduler is already running")
            return
        
        try:
            # Schedule autonomous thinking cycles (every hour)
            self.scheduler.add_job(
                self._autonomous_thinking_cycle,
                trigger=IntervalTrigger(hours=1),
                id="autonomous_thinking",
                name="Autonomous Thinking Cycle",
                max_instances=1,
                replace_existing=True
            )
            
            # Schedule weekly insight generation (every Sunday at 9 AM)
            self.scheduler.add_job(
                self._weekly_insight_generation,
                trigger=CronTrigger(day_of_week='sun', hour=9, minute=0),
                id="weekly_insights",
                name="Weekly Insight Generation",
                max_instances=1,
                replace_existing=True
            )
            
            # Schedule daily life event check (every day at 8 AM)
            self.scheduler.add_job(
                self._daily_life_event_check,
                trigger=CronTrigger(hour=8, minute=0),
                id="daily_life_events",
                name="Daily Life Event Check",
                max_instances=1,
                replace_existing=True
            )
            
            # Schedule memory consolidation (every day at 2 AM)
            self.scheduler.add_job(
                self._memory_consolidation,
                trigger=CronTrigger(hour=2, minute=0),
                id="memory_consolidation",
                name="Memory Consolidation",
                max_instances=1,
                replace_existing=True
            )
            
            # Start the scheduler
            self.scheduler.start()
            self.running = True
            
            logger.info("✅ Autonomous scheduler started with 4 scheduled jobs")
            
            # Run initial thinking cycle after 5 minutes
            self.scheduler.add_job(
                self._autonomous_thinking_cycle,
                trigger=IntervalTrigger(minutes=5),
                id="initial_thinking",
                name="Initial Thinking Cycle",
                max_instances=1
            )
            
        except Exception as e:
            logger.error(f"Failed to start autonomous scheduler: {e}")
            raise e
    
    async def stop(self):
        """Stop the autonomous scheduler"""
        
        if not self.running:
            return
        
        try:
            self.scheduler.shutdown()
            self.running = False
            logger.info("Autonomous scheduler stopped")
        except Exception as e:
            logger.error(f"Error stopping autonomous scheduler: {e}")
    
    async def _autonomous_thinking_cycle(self):
        """Main autonomous thinking cycle"""
        
        if not self.autonomous_orchestrator:
            logger.warning("No AutoGen orchestrator available for thinking cycle")
            return
        
        cycle_start = datetime.now()
        logger.info(f"🧠 Starting autonomous thinking cycle #{self.thinking_cycles_completed + 1}")
        
        try:
            # Run thinking cycle through Intelligence Agent
            result = await self.autonomous_orchestrator.autonomous_thinking_cycle()
            
            if result and not result.get("error"):
                self.thinking_cycles_completed += 1
                self.last_thinking_cycle = cycle_start
                
                # Update performance metrics
                cycle_duration = (datetime.now() - cycle_start).total_seconds()
                self._update_cycle_performance(cycle_duration, True)
                
                # Broadcast results if WebSocket manager available
                if self.websocket_manager:
                    await self._broadcast_thinking_results(result)
                
                logger.info(f"✅ Autonomous thinking cycle completed in {cycle_duration:.2f}s")
                
                # Check if we need to generate immediate insights
                if result.get("insights_generated", 0) > 0:
                    await self._process_immediate_insights(result)
                
            else:
                error_msg = result.get("error", "Unknown error") if result else "No result returned"
                logger.error(f"❌ Autonomous thinking cycle failed: {error_msg}")
                self._update_cycle_performance(0, False)
                
        except Exception as e:
            logger.error(f"❌ Error in autonomous thinking cycle: {e}")
            self._update_cycle_performance(0, False)
    
    async def _weekly_insight_generation(self):
        """Generate comprehensive weekly insights"""
        
        logger.info("📊 Starting weekly insight generation")
        
        try:
            if not self.memory_system:
                logger.warning("No memory system available for weekly insights")
                return
            
            # Get weekly data
            weekly_data = await self._gather_weekly_data()
            
            # Generate insights through AutoGen collaboration
            if self.autonomous_orchestrator and weekly_data:
                insight_prompt = self._create_weekly_insight_prompt(weekly_data)
                
                # Use AutoGen to generate collaborative insights
                insight_result = await self.autonomous_orchestrator.handle_user_message(
                    user_id="system",
                    message=insight_prompt,
                    context={"type": "weekly_insight_generation", "data": weekly_data}
                )
                
                if insight_result:
                    self.insights_generated += 1
                    self.last_insight_generation = datetime.now()
                    
                    # Store insights in memory
                    await self._store_weekly_insights(insight_result, weekly_data)
                    
                    # Broadcast insights
                    if self.websocket_manager:
                        await self._broadcast_weekly_insights(insight_result)
                    
                    logger.info("✅ Weekly insights generated successfully")
                
        except Exception as e:
            logger.error(f"Error in weekly insight generation: {e}")
    
    async def _daily_life_event_check(self):
        """Check for life event milestones and updates"""
        
        logger.info("📅 Running daily life event check")
        
        try:
            if not self.memory_system:
                return
            
            # Get recent life events
            life_events = await self.memory_system.get_memories_by_type("episodic", days=30)
            
            # Check for important milestones
            milestones = await self._check_life_event_milestones(life_events)
            
            if milestones:
                # Generate milestone insights
                milestone_prompt = self._create_milestone_prompt(milestones)
                
                if self.autonomous_orchestrator:
                    milestone_result = await self.autonomous_orchestrator.handle_user_message(
                        user_id="system",
                        message=milestone_prompt,
                        context={"type": "milestone_check", "milestones": milestones}
                    )
                    
                    if milestone_result and self.websocket_manager:
                        await self._broadcast_milestone_update(milestone_result)
                
                logger.info(f"✅ Found {len(milestones)} life event milestones")
            
        except Exception as e:
            logger.error(f"Error in daily life event check: {e}")
    
    async def _memory_consolidation(self):
        """Consolidate and optimize memory storage"""
        
        logger.info("🗂️ Starting memory consolidation")
        
        try:
            if not self.memory_system:
                return
            
            # Run memory consolidation
            consolidation_result = await self.memory_system.consolidate_memories()
            
            if consolidation_result:
                logger.info(f"✅ Memory consolidation completed: {consolidation_result}")
                
                # Broadcast consolidation results
                if self.websocket_manager:
                    await self.websocket_manager.broadcast_memory_update({
                        "type": "memory_consolidation",
                        "result": consolidation_result,
                        "timestamp": datetime.now().isoformat()
                    })
            
        except Exception as e:
            logger.error(f"Error in memory consolidation: {e}")
    
    async def _gather_weekly_data(self) -> Dict[str, Any]:
        """Gather data for weekly insight generation"""
        
        weekly_data = {
            "conversations": [],
            "life_events": [],
            "patterns": {},
            "goals": [],
            "achievements": []
        }
        
        try:
            if self.memory_system:
                # Get last 7 days of data
                recent_memories = await self.memory_system.get_recent_memories(days=7)
                
                # Categorize memories
                for memory in recent_memories:
                    memory_type = memory.get("memory_type", "unknown")
                    content = memory.get("content", "")
                    
                    if memory_type == "episodic":
                        weekly_data["conversations"].append(memory)
                    elif "life_event" in content.lower():
                        weekly_data["life_events"].append(memory)
                    elif memory_type == "prospective":
                        weekly_data["goals"].append(memory)
                
                # Get conversation patterns
                if self.autonomous_orchestrator:
                    recent_conversations = self.autonomous_orchestrator.get_recent_conversations(limit=50)
                    weekly_data["patterns"] = self._analyze_conversation_patterns(recent_conversations)
            
        except Exception as e:
            logger.warning(f"Error gathering weekly data: {e}")
        
        return weekly_data
    
    def _analyze_conversation_patterns(self, conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze conversation patterns for insights"""
        
        patterns = {
            "frequency": len(conversations),
            "topics": {},
            "response_times": [],
            "collaboration_rate": 0
        }
        
        try:
            collaborative_conversations = 0
            
            for conv in conversations:
                # Track response times
                if conv.get("response_time"):
                    patterns["response_times"].append(conv["response_time"])
                
                # Track collaboration
                if conv.get("routing_decision", {}).get("strategy") == "collaborative":
                    collaborative_conversations += 1
                
                # Simple topic extraction (can be enhanced)
                message = conv.get("message", "").lower()
                for topic in ["health", "work", "family", "learning", "planning"]:
                    if topic in message:
                        patterns["topics"][topic] = patterns["topics"].get(topic, 0) + 1
            
            # Calculate collaboration rate
            if conversations:
                patterns["collaboration_rate"] = collaborative_conversations / len(conversations)
            
            # Calculate average response time
            if patterns["response_times"]:
                patterns["avg_response_time"] = sum(patterns["response_times"]) / len(patterns["response_times"])
            
        except Exception as e:
            logger.warning(f"Error analyzing conversation patterns: {e}")
        
        return patterns
    
    def _create_weekly_insight_prompt(self, weekly_data: Dict[str, Any]) -> str:
        """Create prompt for weekly insight generation"""
        
        prompt_parts = [
            "🧠 WEEKLY INTELLIGENCE GENERATION 🧠",
            "",
            "Please analyze the past week's data and generate comprehensive insights:",
            "",
            f"📊 WEEK SUMMARY:",
            f"- Conversations: {len(weekly_data.get('conversations', []))}",
            f"- Life Events: {len(weekly_data.get('life_events', []))}",
            f"- Active Goals: {len(weekly_data.get('goals', []))}",
            "",
            "🎯 ANALYSIS REQUIRED:",
            "1. User engagement patterns and trends",
            "2. Progress on life events and goals",
            "3. Emerging patterns or concerns",
            "4. Proactive recommendations for next week",
            "5. Timeline updates for ongoing life events",
            "",
            "💡 GENERATE:",
            "- Weekly summary insights",
            "- Actionable recommendations",
            "- Timeline updates",
            "- Proactive suggestions for improvement",
            "",
            "Please collaborate between Memory, Research, and Intelligence agents to provide comprehensive weekly insights."
        ]
        
        return "\n".join(prompt_parts)
    
    def _create_milestone_prompt(self, milestones: List[Dict[str, Any]]) -> str:
        """Create prompt for milestone insights"""
        
        prompt_parts = [
            "🎯 LIFE EVENT MILESTONE CHECK 🎯",
            "",
            f"Found {len(milestones)} important milestones to address:",
            ""
        ]
        
        for i, milestone in enumerate(milestones, 1):
            prompt_parts.append(f"{i}. {milestone.get('description', 'Unknown milestone')}")
            prompt_parts.append(f"   Timeline: {milestone.get('timeline_info', 'Unknown')}")
            prompt_parts.append("")
        
        prompt_parts.extend([
            "Please provide:",
            "1. Updated recommendations for each milestone",
            "2. Next steps and actions to take",
            "3. Timeline adjustments if needed",
            "4. Relevant resources or guidance"
        ])
        
        return "\n".join(prompt_parts)
    
    async def _check_life_event_milestones(self, life_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check for important life event milestones"""
        
        milestones = []
        
        try:
            for event in life_events:
                content = event.get("content", "").lower()
                timestamp = event.get("timestamp")
                
                if not timestamp:
                    continue
                
                # Parse timestamp
                try:
                    event_date = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    days_since = (datetime.now() - event_date).days
                except:
                    continue
                
                # Pregnancy milestones
                if "pregnant" in content or "pregnancy" in content:
                    pregnancy_milestones = self._get_pregnancy_milestones(days_since)
                    milestones.extend(pregnancy_milestones)
                
                # Learning milestones
                elif "learn" in content or "course" in content:
                    if days_since == 7 or days_since == 30:  # 1 week or 1 month
                        milestones.append({
                            "type": "learning_check",
                            "description": f"Learning progress check - {days_since} days since starting",
                            "timeline_info": f"Day {days_since} of learning journey",
                            "recommendations": ["Review progress", "Adjust learning plan", "Set new goals"]
                        })
                
                # Health milestones
                elif "health" in content or "doctor" in content:
                    if days_since in [7, 30, 90]:  # Regular health check intervals
                        milestones.append({
                            "type": "health_followup",
                            "description": f"Health follow-up check - {days_since} days since last update",
                            "timeline_info": f"Day {days_since} health timeline",
                            "recommendations": ["Schedule follow-up", "Monitor progress", "Update health plan"]
                        })
            
        except Exception as e:
            logger.warning(f"Error checking life event milestones: {e}")
        
        return milestones
    
    def _get_pregnancy_milestones(self, days_since: int) -> List[Dict[str, Any]]:
        """Get pregnancy-specific milestones"""
        
        milestones = []
        
        # Key pregnancy milestones
        pregnancy_timeline = [
            (7, "Week 1 - Initial planning"),
            (14, "Week 2 - Doctor appointment scheduling"),
            (28, "Week 4 - First prenatal visit"),
            (56, "Week 8 - First trimester progress"),
            (84, "Week 12 - End of first trimester"),
            (140, "Week 20 - Anatomy scan"),
            (196, "Week 28 - Third trimester begins"),
            (252, "Week 36 - Final preparations"),
            (280, "Week 40 - Due date")
        ]
        
        for milestone_days, description in pregnancy_timeline:
            if abs(days_since - milestone_days) <= 1:  # Within 1 day of milestone
                milestones.append({
                    "type": "pregnancy_milestone",
                    "description": f"Pregnancy milestone: {description}",
                    "timeline_info": f"Day {days_since} of pregnancy journey",
                    "recommendations": self._get_pregnancy_recommendations(milestone_days)
                })
        
        return milestones
    
    def _get_pregnancy_recommendations(self, milestone_days: int) -> List[str]:
        """Get pregnancy recommendations for specific milestone"""
        
        if milestone_days <= 14:
            return ["Start prenatal vitamins", "Schedule first appointment", "Review diet and lifestyle"]
        elif milestone_days <= 84:
            return ["Continue prenatal care", "Monitor symptoms", "Consider sharing news"]
        elif milestone_days <= 196:
            return ["Attend regular checkups", "Plan for baby essentials", "Consider childbirth classes"]
        else:
            return ["Prepare for delivery", "Finalize baby preparations", "Pack hospital bag"]
    
    def _update_cycle_performance(self, duration: float, success: bool):
        """Update cycle performance metrics"""
        
        self.cycle_performance["total_cycles"] += 1
        
        if success:
            # Update average duration
            total_successful = self.cycle_performance["total_cycles"] - self.cycle_performance["failed_cycles"]
            current_avg = self.cycle_performance["average_duration"]
            
            if total_successful == 1:
                self.cycle_performance["average_duration"] = duration
            else:
                self.cycle_performance["average_duration"] = (
                    (current_avg * (total_successful - 1) + duration) / total_successful
                )
        else:
            self.cycle_performance["failed_cycles"] += 1
        
        # Update success rate
        self.cycle_performance["success_rate"] = (
            (self.cycle_performance["total_cycles"] - self.cycle_performance["failed_cycles"]) /
            self.cycle_performance["total_cycles"]
        )
    
    async def _broadcast_thinking_results(self, result: Dict[str, Any]):
        """Broadcast thinking results via WebSocket"""
        
        if not self.websocket_manager:
            return
        
        try:
            await self.websocket_manager.broadcast_thinking_update({
                "type": "autonomous_thinking_cycle",
                "result": result,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.warning(f"Error broadcasting thinking results: {e}")
    
    async def _broadcast_weekly_insights(self, insights: Dict[str, Any]):
        """Broadcast weekly insights via WebSocket"""
        
        if not self.websocket_manager:
            return
        
        try:
            await self.websocket_manager.broadcast_insight_update({
                "type": "weekly_insights",
                "insights": insights,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.warning(f"Error broadcasting weekly insights: {e}")
    
    async def _broadcast_milestone_update(self, milestone_result: Dict[str, Any]):
        """Broadcast milestone updates via WebSocket"""
        
        if not self.websocket_manager:
            return
        
        try:
            await self.websocket_manager.broadcast_milestone_update({
                "type": "life_event_milestone",
                "milestone_result": milestone_result,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.warning(f"Error broadcasting milestone update: {e}")
    
    async def _process_immediate_insights(self, result: Dict[str, Any]):
        """Process insights that need immediate attention"""
        
        try:
            insights_count = result.get("insights_generated", 0)
            
            if insights_count > 3:  # High insight generation
                logger.info(f"🔥 High insight activity detected: {insights_count} insights generated")
                
                # Could trigger additional processing or notifications
                if self.websocket_manager:
                    await self.websocket_manager.broadcast_alert({
                        "type": "high_insight_activity",
                        "insights_count": insights_count,
                        "message": "High autonomous intelligence activity detected",
                        "timestamp": datetime.now().isoformat()
                    })
            
        except Exception as e:
            logger.warning(f"Error processing immediate insights: {e}")
    
    async def _store_weekly_insights(self, insight_result: Dict[str, Any], weekly_data: Dict[str, Any]):
        """Store weekly insights in memory system"""
        
        if not self.memory_system:
            return
        
        try:
            # Store the insight
            insight_content = f"Weekly insight: {insight_result.get('response', '')}"
            await self.memory_system.store_memory(
                "system",
                insight_content,
                "procedural"
            )
            
            # Store weekly summary
            summary_content = f"Weekly summary: {len(weekly_data.get('conversations', []))} conversations, {len(weekly_data.get('life_events', []))} life events"
            await self.memory_system.store_memory(
                "system",
                summary_content,
                "semantic"
            )
            
        except Exception as e:
            logger.warning(f"Error storing weekly insights: {e}")
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get current scheduler status"""
        
        return {
            "running": self.running,
            "thinking_cycles_completed": self.thinking_cycles_completed,
            "insights_generated": self.insights_generated,
            "last_thinking_cycle": self.last_thinking_cycle.isoformat() if self.last_thinking_cycle else None,
            "last_insight_generation": self.last_insight_generation.isoformat() if self.last_insight_generation else None,
            "performance_metrics": self.cycle_performance,
            "scheduled_jobs": len(self.scheduler.get_jobs()) if self.scheduler else 0,
            "timestamp": datetime.now().isoformat()
        }