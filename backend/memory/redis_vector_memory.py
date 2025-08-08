"""
Redis Vector Memory Layer
Handles short-term memory with TTL and vector search capabilities using Redis Stack.
"""

import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import redis
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class RedisVectorMemoryLayer:
    """
    Handles short-term memory storage with automatic TTL expiration.
    Uses Redis Stack's vector search capabilities for semantic memory retrieval.
    """
    
    def __init__(
        self,
        redis_client: redis.Redis,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_dimensions: int = 384,
        similarity_threshold: float = 0.7,
        default_ttl: int = 86400,  # 24 hours
        index_name: str = "shortterm_memory_idx",
        config=None  # Add config parameter
    ):
        self.redis_client = redis_client
        self.embedding_model = SentenceTransformer(embedding_model)
        self.vector_dimensions = vector_dimensions
        self.similarity_threshold = similarity_threshold
        self.default_ttl = default_ttl
        self.index_name = index_name
        self.config = config  # Store config reference
        self._index_created = False
        self._use_vector_search = False  # Will be set during connection
        
    async def connect(self) -> bool:
        """Initialize connection and create vector index if needed."""
        try:
            # Test Redis connection
            await self._ensure_async_redis()
            self.redis_client.ping()
            
            # Check if Redis Stack search is available
            try:
                # Try to get module info to verify Redis Stack
                modules = self.redis_client.execute_command("MODULE", "LIST")
                search_available = any('search' in str(module).lower() for module in modules)
                if search_available:
                    logger.info("Redis Stack with search capabilities detected")
                    self._use_vector_search = True
                else:
                    logger.info("Redis Stack search module not detected - using fallback search")
                    self._use_vector_search = False
            except Exception as e:
                logger.info(f"Cannot verify Redis Stack modules ({e}) - using fallback search")
                self._use_vector_search = False
            
            # Create vector index if it doesn't exist
            await self._ensure_vector_index()
            
            logger.info("Redis Vector Memory Layer connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect Redis Vector Memory Layer: {e}")
            return False
    
    async def _ensure_async_redis(self):
        """Ensure we have async Redis client capabilities."""
        if not hasattr(self.redis_client, 'ft'):
            logger.warning("Redis client doesn't support RedisSearch. Vector search may not work.")
    
    async def _ensure_vector_index(self):
        """Create vector search index if it doesn't exist."""
        if self._index_created:
            return
            
        try:
            # Check if index already exists
            self.redis_client.ft(self.index_name).info()
            logger.info(f"Vector index '{self.index_name}' already exists")
            self._index_created = True
            return
            
        except redis.ResponseError:
            # Index doesn't exist, create it
            pass
        
        try:
            # Try multiple import paths for different Redis versions
            search_fields_imported = False
            index_def_imported = False
            
            # Import search fields
            try:
                from redis.commands.search.field import TextField, VectorField, NumericField
                search_fields_imported = True
            except ImportError:
                try:
                    from redis.commands.search import TextField, VectorField, NumericField
                    search_fields_imported = True
                except ImportError:
                    pass
            
            # Import index definition
            try:
                from redis.commands.search.indexDefinition import IndexDefinition, IndexType
                index_def_imported = True
            except ImportError:
                try:
                    from redis.commands.search import IndexDefinition, IndexType
                    index_def_imported = True
                except ImportError:
                    try:
                        from redis.commands.search.definition import IndexDefinition, IndexType
                        index_def_imported = True
                    except ImportError:
                        pass
            
            if not search_fields_imported or not index_def_imported:
                logger.info("Redis search modules not available - using text-based fallback search")
                return
            
            # Define schema
            schema = [
                TextField("content"),
                TextField("user_name"), 
                TextField("memory_type"),
                TextField("metadata"),
                NumericField("created_at"),
                NumericField("importance_score"),
                VectorField(
                    "embedding",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self.vector_dimensions,
                        "DISTANCE_METRIC": "COSINE"
                    }
                )
            ]
            
            # Create index
            index_definition = IndexDefinition(
                prefix=[f"shortterm_memory:"],
                index_type=IndexType.HASH
            )
            
            self.redis_client.ft(self.index_name).create_index(
                schema, 
                definition=index_definition
            )
            
            logger.info(f"Created vector index '{self.index_name}'")
            self._index_created = True
            
        except Exception as e:
            logger.error(f"Failed to create vector index: {e}")
            # Fallback: continue without vector search
            self._index_created = False
    
    async def store_short_term_memory(
        self,
        user_name: str,
        content: str,
        memory_type: str = "short_term",
        importance_score: float = 0.5,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store content in short-term memory with automatic TTL expiration.
        
        Args:
            user_name: User name
            content: Text content to store
            memory_type: Type of memory (default: "short_term")
            importance_score: Importance score (0.0-1.0)
            ttl: Time to live in seconds (default: 24 hours)
            metadata: Additional metadata dictionary
        
        Returns:
            Memory ID of stored item
        """
        try:
            memory_id = str(uuid.uuid4())
            ttl = ttl or self.default_ttl
            
            # Generate embedding
            embedding = self.embedding_model.encode(content)
            embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
            
            # Prepare memory data
            memory_data = {
                "memory_id": memory_id,
                "user_name": user_name,
                "content": content,
                "memory_type": memory_type,
                "importance_score": importance_score,
                "created_at": datetime.now().timestamp(),
                "metadata": json.dumps(metadata or {}),
                "embedding": embedding_bytes
            }
            
            # Store with TTL
            key = f"shortterm_memory:{memory_id}"
            self.redis_client.hset(key, mapping=memory_data)
            self.redis_client.expire(key, ttl)
            
            # Also store user index for quick cleanup
            user_key = f"user_shortterm:{user_name}"
            self.redis_client.sadd(user_key, memory_id)
            self.redis_client.expire(user_key, ttl + 3600)  # Keep user index slightly longer
            
            logger.debug(f"Stored short-term memory {memory_id} for user {user_name} with TTL {ttl}s")
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to store short-term memory: {e}")
            raise
    
    async def update_memory(
        self,
        memory_id: str,
        user_name: str,
        new_content: str,
        new_importance: float,
        refresh_ttl: bool = True,
        new_ttl: Optional[int] = None
    ) -> bool:
        """
        Update existing memory with new content and importance.
        
        Args:
            memory_id: ID of memory to update
            user_name: User name
            new_content: New content
            new_importance: New importance score
            refresh_ttl: Whether to refresh TTL
        
        Returns:
            True if update successful, False otherwise
        """
        try:
            memory_key = f"shortterm_memory:{memory_id}"
            
            # Check if memory exists
            memory_data = await self.redis_client.hgetall(memory_key)
            if not memory_data:
                logger.warning(f"Memory {memory_id} not found for update")
                return False
            
            # Generate new embedding
            new_embedding = self.embedding_model.encode([new_content])[0].astype(np.float32).tobytes()
            
            # Determine TTL for update
            if refresh_ttl:
                # Use provided TTL or default
                final_ttl = new_ttl or self.default_ttl
            else:
                # Keep existing TTL
                final_ttl = await self.redis_client.ttl(memory_key)
                if final_ttl <= 0:
                    final_ttl = self.default_ttl
            
            # Ensure final_ttl is an integer (safety check)
            if not isinstance(final_ttl, int):
                logger.warning(f"TTL value {final_ttl} is not int, using default")
                final_ttl = self.default_ttl
            
            # Update memory data
            updated_data = {
                "content": new_content,
                "importance": new_importance,
                "updated_at": datetime.now().isoformat(),
                "vector": new_embedding
            }
            
            # Update in Redis
            await self.redis_client.hset(memory_key, mapping=updated_data)
            
            # Refresh TTL if requested
            if refresh_ttl:
                await self.redis_client.expire(memory_key, final_ttl)
            
            logger.info(f"Updated memory {memory_id} with importance {new_importance}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update memory {memory_id}: {e}")
            return False
    
    async def search_short_term_memory(
        self,
        user_name: str,
        query: str,
        limit: int = 5,
        min_importance: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search short-term memory using vector similarity.
        
        Args:
            user_name: User name
            query: Search query
            limit: Maximum number of results
            min_importance: Minimum importance score
        
        Returns:
            List of matching memories with similarity scores
        """
        try:
            if not self._index_created:
                # Fallback to text-based search
                return await self._fallback_text_search(user_name, query, limit, min_importance)
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query)
            query_bytes = np.array(query_embedding, dtype=np.float32).tobytes()
            
            # Build vector search query
            from redis.commands.search.query import Query
            
            query_str = f"@user_name:{user_name} @importance_score:[{min_importance} +inf]"
            redis_query = Query(query_str).return_fields(
                "content", "memory_type", "importance_score", "created_at", "metadata"
            ).sort_by("created_at", asc=False).paging(0, limit)
            
            # Add vector search (KNN)
            redis_query = redis_query.add_filter(
                f"@embedding:[VECTOR_RANGE {self.similarity_threshold} $query_vec]"
            )
            
            # Execute search
            results = self.redis_client.ft(self.index_name).search(
                redis_query,
                query_params={"query_vec": query_bytes}
            )
            
            # Process results
            memories = []
            for doc in results.docs:
                memory = {
                    "memory_id": doc.id.split(":")[-1],
                    "content": doc.content,
                    "memory_type": doc.memory_type,
                    "importance_score": float(doc.importance_score),
                    "created_at": float(doc.created_at),
                    "metadata": json.loads(doc.metadata),
                    "similarity_score": getattr(doc, '__score', 1.0)
                }
                memories.append(memory)
            
            logger.debug(f"Found {len(memories)} short-term memories for user {user_name}")
            return memories
            
        except Exception as e:
            logger.error(f"Vector search failed, using fallback: {e}")
            return await self._fallback_text_search(user_name, query, limit, min_importance)
    
    async def _fallback_text_search(
        self, 
        user_name: str, 
        query: str, 
        limit: int, 
        min_importance: float
    ) -> List[Dict[str, Any]]:
        """Fallback text-based search when vector search is unavailable."""
        try:
            # Get user's memory IDs
            user_key = f"user_shortterm:{user_name}"
            memory_ids = self.redis_client.smembers(user_key)
            
            memories = []
            query_lower = query.lower()
            
            for memory_id in memory_ids:
                memory_key = f"shortterm_memory:{memory_id.decode()}"
                memory_data = self.redis_client.hgetall(memory_key)
                
                if not memory_data:
                    continue
                
                importance = float(memory_data[b'importance_score'])
                if importance < min_importance:
                    continue
                
                content = memory_data[b'content'].decode()
                # Simple text matching
                similarity = 0.5 if query_lower in content.lower() else 0.3
                
                memory = {
                    "memory_id": memory_data[b'memory_id'].decode(),
                    "content": content,
                    "memory_type": memory_data[b'memory_type'].decode(),
                    "importance_score": importance,
                    "created_at": float(memory_data[b'created_at']),
                    "metadata": json.loads(memory_data[b'metadata'].decode()),
                    "similarity_score": similarity
                }
                memories.append(memory)
            
            # Sort by importance and creation time
            memories.sort(key=lambda x: (x['importance_score'], x['created_at']), reverse=True)
            return memories[:limit]
            
        except Exception as e:
            logger.error(f"Fallback search failed: {e}")
            return []
    
    async def get_user_short_term_memories(
        self, 
        user_name: str, 
        limit: int = 20,
        min_importance: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Get all short-term memories for a user."""
        try:
            user_key = f"user_shortterm:{user_name}"
            memory_ids = self.redis_client.smembers(user_key)
            
            memories = []
            for memory_id in memory_ids:
                memory_key = f"shortterm_memory:{memory_id.decode()}"
                memory_data = self.redis_client.hgetall(memory_key)
                
                if not memory_data:
                    continue
                
                importance = float(memory_data[b'importance_score'])
                if importance < min_importance:
                    continue
                
                memory = {
                    "memory_id": memory_data[b'memory_id'].decode(),
                    "content": memory_data[b'content'].decode(),
                    "memory_type": memory_data[b'memory_type'].decode(),
                    "importance_score": importance,
                    "created_at": float(memory_data[b'created_at']),
                    "metadata": json.loads(memory_data[b'metadata'].decode())
                }
                memories.append(memory)
            
            # Sort by creation time (newest first)
            memories.sort(key=lambda x: x['created_at'], reverse=True)
            return memories[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get user short-term memories: {e}")
            return []
    
    async def cleanup_user_short_term_memory(self, user_name: str) -> int:
        """
        Clean up all short-term memories for a user.
        Note: TTL will handle automatic expiration, but this provides immediate cleanup.
        
        Returns:
            Number of memories deleted
        """
        try:
            user_key = f"user_shortterm:{user_name}"
            memory_ids = self.redis_client.smembers(user_key)
            
            deleted_count = 0
            for memory_id in memory_ids:
                memory_key = f"shortterm_memory:{memory_id.decode()}"
                if self.redis_client.delete(memory_key):
                    deleted_count += 1
            
            # Clean up user index
            self.redis_client.delete(user_key)
            
            logger.info(f"Cleaned up {deleted_count} short-term memories for user {user_name}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup user short-term memory: {e}")
            return 0
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get short-term memory statistics."""
        try:
            # Count total short-term memories
            pattern = "shortterm_memory:*"
            total_memories = len(self.redis_client.keys(pattern))
            
            # Get memory info
            memory_info = self.redis_client.info('memory')
            
            stats = {
                "total_short_term_memories": total_memories,
                "redis_memory_used": memory_info.get('used_memory_human', 'unknown'),
                "vector_index_exists": self._index_created,
                "index_name": self.index_name,
                "default_ttl_hours": self.default_ttl / 3600
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {"error": str(e)}