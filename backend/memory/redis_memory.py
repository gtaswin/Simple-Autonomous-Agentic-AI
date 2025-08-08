"""
Redis Memory Layer for Working Memory
Handles fast, temporary memory storage with activity-based TTL
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

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
        
        # Memory control configuration (unified system)
        self.max_items = config.get_memory_limit("working_memory")
        self.working_memory_ttl = config.get_memory_ttl("working_memory")
        self.activity_extension = config.get_memory_ttl("activity_extension")
        self.session_limit = config.get_memory_limit("session")
        
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
    
    
    # ========================================
    # NEW: 3-TIER MEMORY SYSTEM SUPPORT
    # ========================================
    
    async def get_working_memory_by_key(self, key: str, limit: int = 7) -> List[Dict]:
        """Get working memory items by specific key (for per-agent per-user)"""
        try:
            if not self.redis:
                return []
            
            # Get items from Redis list (newest first)
            items_data = await self.redis.lrange(key, 0, limit - 1)
            
            items = []
            for item_data in items_data:
                try:
                    item = json.loads(item_data)
                    items.append(item)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse working memory item: {e}")
                    continue
            
            return items
            
        except Exception as e:
            self.logger.error(f"Failed to get working memory by key {key}: {e}")
            return []
    
    async def store_working_memory_by_key(self, key: str, content: str, metadata: Dict[str, Any] = None) -> str:
        """Store working memory item by specific key (for per-agent per-user)"""
        try:
            if not self.redis:
                raise RuntimeError("Redis not connected")
            
            # Create memory item
            memory_item = {
                "content": content,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat()
            }
            
            # Store in Redis list (newest at front)
            item_json = safe_json_dumps(memory_item)
            await self.redis.lpush(key, item_json)
            
            # Maintain configured item limit (trim excess)
            await self.redis.ltrim(key, 0, self.max_items - 1)
            
            # Set TTL for automatic cleanup (7 days)
            await self.redis.expire(key, self.working_memory_ttl)
            
            return f"{key}:item_{int(datetime.now().timestamp())}"
            
        except Exception as e:
            self.logger.error(f"Failed to store working memory by key {key}: {e}")
            raise
    
    async def clear_working_memory_by_key(self, key: str) -> bool:
        """Clear working memory by specific key"""
        try:
            if not self.redis:
                return False
            
            result = await self.redis.delete(key)
            return result > 0
            
        except Exception as e:
            self.logger.error(f"Failed to clear working memory by key {key}: {e}")
            return False
    
    async def store_session_conversation(self, key: str, conversation: Dict[str, Any], max_conversations: int = None) -> str:
        """Store complete conversation in session memory"""
        try:
            if not self.redis:
                raise RuntimeError("Redis not connected")
            
            # Store conversation in Redis list
            conversation_json = safe_json_dumps(conversation)
            await self.redis.lpush(key, conversation_json)
            
            # Maintain conversation limit
            limit = max_conversations or self.session_limit
            await self.redis.ltrim(key, 0, limit - 1)
            
            # No TTL for session memory (persists until manual cleanup)
            
            return f"{key}:conversation_{int(datetime.now().timestamp())}"
            
        except Exception as e:
            self.logger.error(f"Failed to store session conversation: {e}")
            raise
    
    async def get_session_conversations(self, key: str, limit: int = None) -> List[Dict]:
        """Get session conversations by key in chronological order (oldest first)"""
        try:
            if not self.redis:
                return []
            
            # Get conversations from Redis list (newest first from lpush)
            actual_limit = limit or self.session_limit
            conversations_data = await self.redis.lrange(key, 0, actual_limit - 1)
            
            conversations = []
            for conv_data in conversations_data:
                try:
                    conversation = json.loads(conv_data)
                    conversations.append(conversation)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse session conversation: {e}")
                    continue
            
            # Reverse to get chronological order (oldest first)
            return conversations[::-1]
            
        except Exception as e:
            self.logger.error(f"Failed to get session conversations: {e}")
            return []
    
    async def clear_session_memory(self, key: str) -> bool:
        """Clear session memory by key"""
        try:
            if not self.redis:
                return False
            
            result = await self.redis.delete(key)
            return result > 0
            
        except Exception as e:
            self.logger.error(f"Failed to clear session memory by key {key}: {e}")
            return False
    
    async def get_chat_history(self, user_id: str, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Get chat history for API endpoint compatibility"""
        try:
            # Use session conversation method to get chat history
            session_key = f"session_memory:{user_id}"
            conversations = await self.get_session_conversations(session_key, limit=limit + offset)
            
            # Apply offset and limit
            if offset > 0:
                conversations = conversations[offset:]
            if limit and len(conversations) > limit:
                conversations = conversations[:limit]
            
            return conversations
            
        except Exception as e:
            self.logger.error(f"Failed to get chat history for {user_id}: {e}")
            return []
    
    async def get_autonomous_insights(self, user_id: str) -> Dict[str, Any]:
        """Get autonomous insights for API endpoint compatibility"""
        try:
            # Get insights from autonomous_insights pattern
            insights_pattern = f"autonomous_insights:{user_id}:*"
            
            # Get all insight keys for the user
            insight_keys = await self.redis.keys(insights_pattern)
            
            insights = {}
            for key in insight_keys:
                # Extract insight type from key
                key_str = key.decode() if isinstance(key, bytes) else key
                insight_type = key_str.split(':')[-1]  # Get last part after ':'
                
                # Get insight data as hash (insights are stored as Redis hashes)
                insight_data = await self.redis.hgetall(key)
                if insight_data:
                    # Convert Redis hash to dict with proper JSON parsing
                    parsed_insight = {}
                    for field, value in insight_data.items():
                        field_str = field.decode() if isinstance(field, bytes) else field
                        value_str = value.decode() if isinstance(value, bytes) else value
                        
                        # Try to parse JSON values
                        try:
                            import json
                            parsed_insight[field_str] = json.loads(value_str)
                        except (json.JSONDecodeError, ValueError):
                            parsed_insight[field_str] = value_str
                    
                    insights[insight_type] = parsed_insight
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Failed to get autonomous insights for {user_id}: {e}")
            return {}
    
    async def clear_autonomous_insights(self, user_id: str) -> Dict[str, Any]:
        """Clear autonomous insights for API endpoint compatibility"""
        try:
            # Get all insight keys for the user
            insights_pattern = f"autonomous_insights:{user_id}:*"
            insight_keys = await self.redis.keys(insights_pattern)
            
            # Delete all insights
            if insight_keys:
                deleted_count = await self.redis.delete(*insight_keys)
                return {
                    "deleted_insights": deleted_count,
                    "insight_types_cleared": [key.decode().split(':')[-1] if isinstance(key, bytes) else key.split(':')[-1] for key in insight_keys]
                }
            else:
                return {"deleted_insights": 0, "insight_types_cleared": []}
                
        except Exception as e:
            self.logger.error(f"Failed to clear autonomous insights for {user_id}: {e}")
            return {"deleted_insights": 0, "insight_types_cleared": [], "error": str(e)}