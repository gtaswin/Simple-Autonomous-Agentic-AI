"""
Redis Memory Layer for Working Memory
Handles fast, temporary memory storage with activity-based TTL
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict

# Import safe JSON serialization
from utils.serialization import safe_json_dumps

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("❌ CRITICAL ERROR: Redis is required for working memory!")
    print("❌ Install with: pip install redis")
    exit(1)

@dataclass
class WorkingMemoryItem:
    """Working memory item with metadata"""
    content: str
    user_id: str
    session_id: str
    memory_type: str
    timestamp: datetime
    importance: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkingMemoryItem':
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

class RedisMemoryLayer:
    """Redis-based working memory with activity-based TTL"""
    
    def __init__(self, config):
        self.config = config
        self.redis = None
        self.logger = logging.getLogger(__name__)
        
        # Redis configuration (consolidated)
        redis_config = config.get("databases.redis", {})
        if not redis_config:
            raise ValueError("❌ CONFIGURATION ERROR: 'databases.redis' not found in settings.yaml")
        
        self.host = redis_config.get("host")
        if not self.host:
            raise ValueError("❌ CONFIGURATION ERROR: 'databases.redis.host' not found in settings.yaml")
        
        self.port = redis_config.get("port")
        if not self.port:
            raise ValueError("❌ CONFIGURATION ERROR: 'databases.redis.port' not found in settings.yaml")
        
        self.db = redis_config.get("db")
        if self.db is None:
            raise ValueError("❌ CONFIGURATION ERROR: 'databases.redis.db' not found in settings.yaml")
        
        self.password = redis_config.get("password")
        
        # Working memory configuration (from consolidated config)
        self.max_items = redis_config.get("max_working_items")
        if not self.max_items:
            raise ValueError("❌ CONFIGURATION ERROR: 'databases.redis.max_working_items' not found in settings.yaml")
        
        self.default_ttl = redis_config.get("default_ttl")
        if not self.default_ttl:
            raise ValueError("❌ CONFIGURATION ERROR: 'databases.redis.default_ttl' not found in settings.yaml")
        
        self.activity_extension = 86400  # Extend TTL by 1 day on access
        
    async def connect(self):
        """Initialize Redis connection"""
        try:
            self.redis = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True,
                retry_on_timeout=True,
                socket_keepalive=True
            )
            
            # Test connection
            await self.redis.ping()
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Redis connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Close Redis connection"""
        if self.redis:
            await self.redis.close()
            self.logger.info("Redis connection closed")
    
    async def store_working_memory(self, content: str, user_id: str, session_id: str, memory_type: str = "working") -> bool:
        """Store item in working memory with activity-based TTL"""
        try:
            # Redis is required for autonomous agent memory
            if not self.redis:
                raise RuntimeError("❌ CRITICAL ERROR: Redis is required for autonomous agent memory! Ensure Redis is running and connected.")
                
            # Create memory item
            item = WorkingMemoryItem(
                content=content,
                user_id=user_id,
                session_id=session_id,
                memory_type=memory_type,
                timestamp=datetime.now()
            )
            
            # Redis key for user's working memory
            key = f"working_memory:{user_id}"
            
            # Store as JSON in Redis list using safe serialization
            item_json = safe_json_dumps(item.to_dict())
            
            # Add to beginning of list (most recent first)
            await self.redis.lpush(key, item_json)
            
            # Trim to keep only max_items
            await self.redis.ltrim(key, 0, self.max_items - 1)
            
            # Set/extend TTL with activity-based extension
            await self.redis.expire(key, self.default_ttl)
            
            # Memory stored successfully
            return True
            
        except Exception as e:
            print(f"❌ Redis: Failed to store working memory: {e}")
            self.logger.error(f"Failed to store working memory: {e}")
            return False
    
    async def get_working_memory(self, user_id: str, limit: Optional[int] = None) -> List[WorkingMemoryItem]:
        """Retrieve working memory items for user"""
        try:
            key = f"working_memory:{user_id}"
            
            # Extend TTL on access (activity-based)
            await self.redis.expire(key, self.default_ttl)
            
            # Get items from Redis list
            limit = limit or self.max_items
            items_json = await self.redis.lrange(key, 0, limit - 1)
            
            # Parse JSON items
            items = []
            for item_json in items_json:
                try:
                    item_data = json.loads(item_json)
                    item = WorkingMemoryItem.from_dict(item_data)
                    items.append(item)
                except Exception as e:
                    self.logger.warning(f"Failed to parse working memory item: {e}")
                    continue
            
            # Memory retrieved successfully
            return items
            
        except Exception as e:
            print(f"❌ Redis: Failed to retrieve working memory: {e}")
            self.logger.error(f"Failed to retrieve working memory: {e}")
            return []
    
    async def clear_working_memory(self, user_id: str) -> bool:
        """Clear working memory for user"""
        try:
            key = f"working_memory:{user_id}"
            await self.redis.delete(key)
            self.logger.info(f"Cleared working memory for {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clear working memory: {e}")
            return False
    
    async def get_memory_stats(self, user_id: str) -> Dict[str, Any]:
        """Get working memory statistics"""
        try:
            key = f"working_memory:{user_id}"
            
            # Get list length and TTL
            length = await self.redis.llen(key)
            ttl = await self.redis.ttl(key)
            
            # Get oldest and newest timestamps
            items = await self.get_working_memory(user_id)
            
            stats = {
                "total_items": length,
                "ttl_seconds": ttl,
                "ttl_days": round(ttl / 86400, 1) if ttl > 0 else 0,
                "oldest_item": items[-1].timestamp.isoformat() if items else None,
                "newest_item": items[0].timestamp.isoformat() if items else None,
                "memory_span_hours": 0
            }
            
            if len(items) >= 2:
                span = items[0].timestamp - items[-1].timestamp
                stats["memory_span_hours"] = round(span.total_seconds() / 3600, 1)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get memory stats: {e}")
            return {}
    
    # ========================================
    # SINGLE SESSION CHAT CONVERSATION METHODS
    # ========================================
    
    async def store_chat_message(self, user_id: str, message: str, sender: str = "user", metadata: Dict[str, Any] = None, session_id: Optional[str] = None) -> bool:
        """Store chat message in single session conversation (session_id ignored for single-user system)"""
        try:
            # Redis is required for autonomous agent memory
            if not self.redis:
                raise RuntimeError("❌ CRITICAL ERROR: Redis is required for autonomous agent memory! Ensure Redis is running and connected.")
                
            # Create chat message item
            chat_item = {
                "message": message,
                "sender": sender,  # "user" or "assistant"
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            # Single chat conversation key
            key = f"chat_conversation:{user_id}"
            
            # Store as JSON in Redis list (chronological order) using safe serialization
            item_json = safe_json_dumps(chat_item)
            await self.redis.rpush(key, item_json)
            
            # Set TTL to prevent indefinite storage (extend on activity)
            await self.redis.expire(key, 86400 * 7)  # 7 days TTL
            
            # Chat message stored
            return True
            
        except Exception as e:
            print(f"❌ Redis: Failed to store chat message: {e}")
            self.logger.error(f"Failed to store chat message: {e}")
            return False
    
    async def get_chat_history(self, user_id: str, limit: Optional[int] = None, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get chat conversation history for user (session_id ignored for single-user system)"""
        try:
            # Redis is required for autonomous agent memory
            if not self.redis:
                raise RuntimeError("❌ CRITICAL ERROR: Redis is required for autonomous agent memory! Ensure Redis is running and connected.")
                
            key = f"chat_conversation:{user_id}"
            
            # Extend TTL on access (activity-based)
            await self.redis.expire(key, 86400 * 7)
            
            # Get messages from Redis list
            if limit:
                # Get last N messages
                items_json = await self.redis.lrange(key, -limit, -1)
            else:
                # Get all messages
                items_json = await self.redis.lrange(key, 0, -1)
            
            # Parse JSON items
            messages = []
            for item_json in items_json:
                try:
                    message_data = json.loads(item_json)
                    messages.append(message_data)
                except Exception as e:
                    self.logger.warning(f"Failed to parse chat message: {e}")
                    continue
            
            # Chat history retrieved
            return messages
            
        except Exception as e:
            print(f"❌ Redis: Failed to retrieve chat history: {e}")
            self.logger.error(f"Failed to retrieve chat history: {e}")
            return []
    
    async def clear_chat_conversation(self, user_id: str) -> bool:
        """Clear entire chat conversation for user (Reset Chat)"""
        try:
            key = f"chat_conversation:{user_id}"
            await self.redis.delete(key)
            self.logger.info(f"Cleared chat conversation for {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clear chat conversation: {e}")
            return False
    
    async def get_chat_stats(self, user_id: str) -> Dict[str, Any]:
        """Get chat conversation statistics"""
        try:
            key = f"chat_conversation:{user_id}"
            
            # Get conversation length and TTL
            length = await self.redis.llen(key)
            ttl = await self.redis.ttl(key)
            
            # Get first and last messages
            messages = await self.get_chat_history(user_id)
            
            stats = {
                "total_messages": length,
                "user_messages": len([m for m in messages if m.get("sender") == "user"]),
                "assistant_messages": len([m for m in messages if m.get("sender") == "assistant"]),
                "ttl_seconds": ttl,
                "ttl_days": round(ttl / 86400, 1) if ttl > 0 else 0,
                "first_message": messages[0]["timestamp"] if messages else None,
                "last_message": messages[-1]["timestamp"] if messages else None,
                "conversation_span_hours": 0
            }
            
            if len(messages) >= 2:
                from datetime import datetime
                first_time = datetime.fromisoformat(messages[0]["timestamp"])
                last_time = datetime.fromisoformat(messages[-1]["timestamp"])
                span = last_time - first_time
                stats["conversation_span_hours"] = round(span.total_seconds() / 3600, 1)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get chat stats: {e}")
            return {}

    async def cleanup_expired_memory(self) -> int:
        """Clean up expired working memory keys"""
        try:
            # Find all working memory keys
            keys = await self.redis.keys("working_memory:*")
            
            cleaned = 0
            for key in keys:
                ttl = await self.redis.ttl(key)
                if ttl == -2:  # Key expired/doesn't exist
                    await self.redis.delete(key)
                    cleaned += 1
                elif ttl == -1:  # Key exists but no TTL set
                    # Reset TTL for keys without expiration
                    await self.redis.expire(key, self.default_ttl)
            
            if cleaned > 0:
                self.logger.info(f"Cleaned up {cleaned} expired working memory keys")
            
            return cleaned
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup expired memory: {e}")
            return 0
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Redis health and return status"""
        try:
            # Test basic operations
            test_key = "health_check_test"
            await self.redis.set(test_key, "test", ex=1)
            value = await self.redis.get(test_key)
            await self.redis.delete(test_key)
            
            # Get Redis info
            info = await self.redis.info()
            
            return {
                "status": "healthy",
                "connection": "active",
                "redis_version": info.get("redis_version"),
                "used_memory_human": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed"),
                "test_operation": "passed" if value == "test" else "failed"
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "connection": "failed",
                "error": str(e)
            }