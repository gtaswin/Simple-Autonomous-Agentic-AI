"""
Event-driven architecture for autonomous agent coordination
"""
import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import json


@dataclass
class Event:
    """Event data structure"""
    id: str
    type: str
    data: Dict[str, Any]
    timestamp: datetime
    source: str
    user_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


class EventBus:
    """Central event bus for autonomous agent coordination"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.event_history: List[Event] = []
        self.max_history = 1000  # Keep last 1000 events for debugging
        self.running = True
        
    async def publish(self, event_type: str, data: Dict[str, Any], 
                     source: str = "system", user_id: Optional[str] = None) -> str:
        """Publish an event to all subscribers"""
        event = Event(
            id=str(uuid.uuid4()),
            type=event_type,
            data=data,
            timestamp=datetime.now(),
            source=source,
            user_id=user_id
        )
        
        # Store in history (keep memory bounded)
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history = self.event_history[-self.max_history:]
        
        # Notify all subscribers asynchronously
        subscribers = self.subscribers.get(event_type, [])
        if subscribers:
            try:
                await asyncio.gather(*[
                    self._safe_call_subscriber(subscriber, event)
                    for subscriber in subscribers
                ], return_exceptions=True)
            except Exception as e:
                print(f"Error publishing event {event_type}: {e}")
        
        return event.id
    
    async def _safe_call_subscriber(self, subscriber: Callable, event: Event):
        """Safely call a subscriber with error handling"""
        try:
            if asyncio.iscoroutinefunction(subscriber):
                await subscriber(event)
            else:
                subscriber(event)
        except Exception as e:
            print(f"Error in event subscriber {subscriber.__name__}: {e}")
    
    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to an event type"""
        self.subscribers[event_type].append(callback)
        return lambda: self.unsubscribe(event_type, callback)
    
    def unsubscribe(self, event_type: str, callback: Callable):
        """Unsubscribe from an event type"""
        if callback in self.subscribers[event_type]:
            self.subscribers[event_type].remove(callback)
    
    def get_event_history(self, event_type: Optional[str] = None, 
                         user_id: Optional[str] = None, limit: int = 100) -> List[Event]:
        """Get event history with optional filtering"""
        events = self.event_history
        
        if event_type:
            events = [e for e in events if e.type == event_type]
        
        if user_id:
            events = [e for e in events if e.user_id == user_id]
        
        return events[-limit:]
    
    def get_subscriber_count(self, event_type: str) -> int:
        """Get number of subscribers for an event type"""
        return len(self.subscribers.get(event_type, []))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics"""
        event_counts = defaultdict(int)
        for event in self.event_history:
            event_counts[event.type] += 1
        
        return {
            "total_events": len(self.event_history),
            "event_types": dict(event_counts),
            "subscribers": {
                event_type: len(subs) 
                for event_type, subs in self.subscribers.items()
            },
            "active": self.running
        }


# Global event bus instance
global_event_bus = EventBus()


# Event type constants for autonomous agent
class EventTypes:
    """Standard event types for autonomous agent"""
    
    # User interactions
    USER_MESSAGE = "user_message"
    USER_GOAL_SET = "user_goal_set"
    USER_PREFERENCE_UPDATED = "user_preference_updated"
    
    # Agent reasoning
    THOUGHT_GENERATED = "thought_generated"
    DECISION_MADE = "decision_made"
    INSIGHT_DISCOVERED = "insight_discovered"
    PATTERN_DETECTED = "pattern_detected"
    
    # Memory events
    MEMORY_STORED = "memory_stored"
    MEMORY_RETRIEVED = "memory_retrieved"
    MEMORY_CONSOLIDATED = "memory_consolidated"
    
    # Agent collaboration
    AGENT_TASK_STARTED = "agent_task_started"
    AGENT_TASK_COMPLETED = "agent_task_completed"
    AGENT_COMMUNICATION = "agent_communication"
    
    # System events
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_ERROR = "system_error"
    SYSTEM_HEALTH_CHECK = "system_health_check"
    
    # Tool usage
    TOOL_EXECUTED = "tool_executed"
    TOOL_ERROR = "tool_error"


# Convenience functions
async def publish_event(event_type: str, data: Dict[str, Any], 
                       source: str = "system", user_id: Optional[str] = None) -> str:
    """Convenience function to publish events"""
    return await global_event_bus.publish(event_type, data, source, user_id)


def subscribe_to_event(event_type: str, callback: Callable):
    """Convenience function to subscribe to events"""
    return global_event_bus.subscribe(event_type, callback)