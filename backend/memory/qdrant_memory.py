"""
Qdrant Memory Layer for Long-term Memory Storage
Handles persistent memory storage with vector embeddings
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    print("❌ CRITICAL ERROR: Qdrant is required for long-term memory!")
    print("❌ Install with: pip install qdrant-client")
    exit(1)

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("❌ CRITICAL ERROR: sentence-transformers is required for embeddings!")
    print("❌ Install with: pip install sentence-transformers")
    exit(1)

from .autonomous_memory import MemoryTrace, MemoryType, MemoryPriority

@dataclass
class QdrantMemoryItem:
    """Qdrant memory item with vector embeddings"""
    id: str
    content: str
    memory_type: str
    user_name: str
    timestamp: datetime
    importance: float
    tags: List[str]
    concepts: List[str]
    entities: List[str]
    embedding: List[float]
    
    def to_point(self) -> PointStruct:
        """Convert to Qdrant point structure"""
        return PointStruct(
            id=self.id,
            vector=self.embedding,
            payload={
                "content": self.content,
                "memory_type": self.memory_type,
                "user_name": self.user_name,
                "timestamp": self.timestamp.timestamp(),  # Store as numeric timestamp for filtering
                "timestamp_iso": self.timestamp.isoformat(),  # Keep ISO format for readability
                "importance": self.importance,
                "tags": self.tags,
                "concepts": self.concepts,
                "entities": self.entities
            }
        )
    
    @classmethod
    def from_point(cls, point: dict) -> 'QdrantMemoryItem':
        """Create from Qdrant point"""
        payload = point.get("payload", {})
        # Handle both old ISO format and new timestamp format for backward compatibility
        if "timestamp_iso" in payload:
            timestamp = datetime.fromisoformat(payload["timestamp_iso"])
        elif isinstance(payload["timestamp"], str):
            timestamp = datetime.fromisoformat(payload["timestamp"])
        else:
            timestamp = datetime.fromtimestamp(payload["timestamp"])
            
        return cls(
            id=point["id"],
            content=payload["content"],
            memory_type=payload["memory_type"],
            user_name=payload["user_name"],
            timestamp=timestamp,
            importance=payload["importance"],
            tags=payload.get("tags", []),
            concepts=payload.get("concepts", []),
            entities=payload.get("entities", []),
            embedding=point.get("vector", [])
        )

class QdrantMemoryLayer:
    """Qdrant-based long-term memory with vector embeddings"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Qdrant configuration (simplified)
        qdrant_config = config.get("databases.qdrant", {})
        if not qdrant_config:
            raise ValueError("❌ CONFIGURATION ERROR: 'databases.qdrant' not found in settings.yaml")
        
        # Essential fields only
        self.host = qdrant_config["host"]
        self.port = qdrant_config["port"]
        self.collection_name = qdrant_config["collection_name"]
        self.vector_size = qdrant_config["vector_size"]
        self.similarity_threshold = qdrant_config.get("similarity_threshold", 0.7)
        
        # Initialize clients
        self.client = None
        self.embedding_model = None
        
    async def connect(self):
        """Initialize Qdrant connection and embedding model"""
        try:
            # Initialize Qdrant client (HTTP connection only)
            self.client = QdrantClient(
                host=self.host,
                port=self.port
            )
            self.logger.info(f"✅ Qdrant connected via HTTP on port {self.port}")
            
            # Test connection
            collections = self.client.get_collections()
            
            # Initialize embedding model from consolidated configuration
            embedder_name = self.config.get("transformers.models.embedder")
            if not embedder_name:
                raise ValueError("❌ CONFIGURATION ERROR: 'transformers.models.embedder' not found in settings.yaml")
            self.embedding_model = SentenceTransformer(embedder_name)
            
            # Create collection if it doesn't exist
            await self._ensure_collection_exists()
            
            # Qdrant connected successfully
            return True
            
        except Exception as e:
            print(f"❌ Qdrant: Connection failed: {e}")
            self.logger.error(f"❌ Qdrant connection failed: {e}")
            return False
    
    async def _ensure_collection_exists(self):
        """Create collection if it doesn't exist"""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                self.logger.info(f"Created Qdrant collection: {self.collection_name}")
        except Exception as e:
            self.logger.error(f"Failed to ensure collection exists: {e}")
            raise
    
    def _create_embedding(self, text: str) -> List[float]:
        """Create embedding for text"""
        try:
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            self.logger.error(f"Failed to create embedding: {e}")
            return [0.0] * self.vector_size  # Fallback to zero vector
    
    async def store_memory(self, memory: MemoryTrace) -> bool:
        """Store memory in Qdrant with vector embedding"""
        try:
            # Create embedding
            embedding = self._create_embedding(memory.content)
            
            # Create Qdrant memory item
            qdrant_item = QdrantMemoryItem(
                id=memory.id,
                content=memory.content,
                memory_type=memory.memory_type.value,
                user_name=memory.user_name,
                timestamp=memory.created_at,
                importance=memory.importance_score,
                tags=list(memory.tags),
                concepts=list(memory.concepts),
                entities=list(memory.entities),
                embedding=embedding
            )
            
            # Store in Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[qdrant_item.to_point()]
            )
            
            self.logger.debug(f"Stored memory in Qdrant: {memory.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store memory in Qdrant: {e}")
            return False
    
    async def search_memories(
        self, 
        query: str, 
        memory_types: List[MemoryType] = None,
        user_name: str = "John",
        limit: int = 10,
        min_similarity: float = None
    ) -> List[QdrantMemoryItem]:
        """Search memories by semantic similarity"""
        try:
            # Create query embedding
            query_embedding = self._create_embedding(query)
            
            # Build filter conditions
            filter_conditions = [
                FieldCondition(
                    key="user_name",
                    match=models.MatchValue(value=user_name)
                )
            ]
            
            if memory_types:
                filter_conditions.append(
                    FieldCondition(
                        key="memory_type",
                        match=models.MatchAny(any=[mt.value for mt in memory_types])
                    )
                )
            
            # Filter out expired memories (only return non-expired or critical memories)
            current_time = datetime.now().timestamp()  # Use timestamp for numeric comparison
            
            # Add expiration filter - memories with future expiration dates
            # Note: We'll filter at application level since Qdrant stores ISO strings
            # TODO: Store timestamps as numbers for better Qdrant filtering
            
            # Search with vector similarity
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=Filter(must=filter_conditions),
                limit=limit,
                score_threshold=min_similarity or self.similarity_threshold
            )
            
            # Convert to QdrantMemoryItem objects and filter expired memories
            memories = []
            current_time_dt = datetime.now()
            
            for result in search_results:
                try:
                    # Check expiration at application level
                    expires_at_str = result.payload.get('expires_at')
                    
                    # Skip if expired (unless it's a critical memory with far-future date)
                    if expires_at_str and expires_at_str != "3000-01-01T00:00:00":
                        try:
                            expires_at = datetime.fromisoformat(expires_at_str.replace('Z', ''))
                            if current_time_dt > expires_at:
                                continue  # Skip expired memory
                        except (ValueError, AttributeError):
                            # If we can't parse the date, keep the memory (safer approach)
                            pass
                    
                    memory_item = QdrantMemoryItem.from_point({
                        "id": result.id,
                        "vector": result.vector,
                        "payload": result.payload
                    })
                    memories.append(memory_item)
                except Exception as e:
                    self.logger.warning(f"Failed to parse search result: {e}")
                    continue
            
            self.logger.debug(f"Found {len(memories)} memories for query: {query[:50]}...")
            return memories
            
        except Exception as e:
            self.logger.error(f"Failed to search memories: {e}")
            return []
    
    async def get_memories_by_type(
        self, 
        memory_type: MemoryType, 
        user_name: str = "John",
        limit: int = 100
    ) -> List[QdrantMemoryItem]:
        """Get all memories of a specific type"""
        try:
            filter_conditions = [
                FieldCondition(
                    key="user_name",
                    match=models.MatchValue(value=user_name)
                ),
                FieldCondition(
                    key="memory_type",
                    match=models.MatchValue(value=memory_type.value)
                )
            ]
            
            # Scroll through all memories of type
            memories = []
            offset = None
            
            while len(memories) < limit:
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=Filter(must=filter_conditions),
                    limit=min(50, limit - len(memories)),
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                if not scroll_result[0]:  # No more results
                    break
                
                for point in scroll_result[0]:
                    try:
                        memory_item = QdrantMemoryItem.from_point({
                            "id": point.id,
                            "vector": [],
                            "payload": point.payload
                        })
                        memories.append(memory_item)
                    except Exception as e:
                        self.logger.warning(f"Failed to parse memory: {e}")
                        continue
                
                offset = scroll_result[1]  # Next offset
                
                if not offset:  # No more pages
                    break
            
            return memories[:limit]
            
        except Exception as e:
            self.logger.error(f"Failed to get memories by type: {e}")
            return []
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete memory from Qdrant"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[memory_id]
                )
            )
            self.logger.debug(f"Deleted memory from Qdrant: {memory_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete memory: {e}")
            return False
    
    async def get_memory_stats(self, user_name: str = "John") -> Dict[str, Any]:
        """Get Qdrant memory statistics"""
        try:
            # Get collection info
            collection_info = self.client.get_collection(self.collection_name)
            
            # Count memories by type for user
            type_counts = {}
            for memory_type in MemoryType:
                memories = await self.get_memories_by_type(memory_type, user_name, limit=1000)
                type_counts[memory_type.value] = len(memories)
            
            return {
                "total_points": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance.value,
                "user_memory_counts": type_counts,
                "collection_status": collection_info.status.value
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get memory stats: {e}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Qdrant health and return status"""
        try:
            # Test basic operations
            collections = self.client.get_collections()
            collection_info = self.client.get_collection(self.collection_name)
            
            return {
                "status": "healthy",
                "connection": "active",
                "collections_count": len(collections.collections),
                "target_collection": self.collection_name,
                "points_count": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size,
                "embedding_model": self.config.get("transformers.models.embedder", "unknown")
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "connection": "failed",
                "error": str(e)
            }
    
    async def get_recent_memories(self, user_name: str, since_date: datetime) -> List[QdrantMemoryItem]:
        """Get memories since a specific date"""
        try:
            from qdrant_client.models import Filter, FieldCondition, Range, DatetimeRange
            
            # Convert datetime to timestamp for filtering
            since_timestamp = since_date.timestamp()
            
            # Create filter for user and date range
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="user_name",
                        match={"value": user_name}
                    ),
                    FieldCondition(
                        key="timestamp",
                        range=Range(gte=since_timestamp)
                    )
                ]
            )
            
            # Search for memories within date range
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=[0.0] * self.vector_size,  # Dummy vector
                query_filter=search_filter,
                limit=1000,  # Get up to 1000 recent memories
                with_payload=True
            )
            
            recent_memories = []
            for result in results:
                payload = result.payload
                memory_item = QdrantMemoryItem(
                    id=str(result.id),
                    content=payload.get("content", ""),
                    memory_type=payload.get("memory_type", "long_term"),
                    user_name=payload.get("user_name", user_name),
                    timestamp=datetime.fromtimestamp(payload.get("timestamp", 0)) if isinstance(payload.get("timestamp"), (int, float)) else datetime.fromisoformat(payload.get("timestamp", datetime.now().isoformat())),
                    embedding=result.vector if hasattr(result, 'vector') else [],
                    tags=payload.get("tags", []),
                    concepts=payload.get("concepts", []),
                    entities=payload.get("entities", []),
                    importance=payload.get("importance", 0.5)
                )
                recent_memories.append(memory_item)
            
            # Sort by timestamp (most recent first)
            recent_memories.sort(key=lambda x: x.timestamp, reverse=True)
            
            return recent_memories
            
        except Exception as e:
            self.logger.error(f"Error getting recent memories from Qdrant: {e}")
            return []
    
    # No cleanup method needed - Qdrant is permanent storage
    # Legacy cleanup methods removed (get_all_user_memories, get_all_memories, delete_memory)  
    # Redis Stack handles TTL natively, no manual cleanup needed