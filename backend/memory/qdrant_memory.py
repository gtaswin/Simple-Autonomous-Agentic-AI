"""
Qdrant Memory Layer for Long-term Memory Storage
Handles persistent memory storage with vector embeddings
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict

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

from .memory_types import MemoryTrace, MemoryType, MemoryPriority

@dataclass
class QdrantMemoryItem:
    """Qdrant memory item with vector embeddings"""
    id: str
    content: str
    memory_type: str
    user_id: str
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
                "user_id": self.user_id,
                "timestamp": self.timestamp.isoformat(),
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
        return cls(
            id=point["id"],
            content=payload["content"],
            memory_type=payload["memory_type"],
            user_id=payload["user_id"],
            timestamp=datetime.fromisoformat(payload["timestamp"]),
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
        
        # Qdrant configuration (consolidated)
        qdrant_config = config.get("databases.qdrant", {})
        if not qdrant_config:
            raise ValueError("❌ CONFIGURATION ERROR: 'databases.qdrant' not found in settings.yaml")
        
        self.host = qdrant_config.get("host")
        if not self.host:
            raise ValueError("❌ CONFIGURATION ERROR: 'databases.qdrant.host' not found in settings.yaml")
        
        self.port = qdrant_config.get("port")
        if not self.port:
            raise ValueError("❌ CONFIGURATION ERROR: 'databases.qdrant.port' not found in settings.yaml")
        
        # gRPC configuration
        self.grpc_port = qdrant_config.get("grpc_port", 6334)
        self.prefer_grpc = qdrant_config.get("prefer_grpc", False)
        
        self.collection_name = qdrant_config.get("collection_name")
        if not self.collection_name:
            raise ValueError("❌ CONFIGURATION ERROR: 'databases.qdrant.collection_name' not found in settings.yaml")
        
        # Use consolidated timeout configuration
        self.timeout = self.config.get("timeouts.database_operations")
        if not self.timeout:
            raise ValueError("❌ CONFIGURATION ERROR: 'timeouts.database_operations' not found in settings.yaml")
        
        # Initialize clients
        self.client = None
        self.embedding_model = None
        
        # Memory configuration (from consolidated config)
        self.vector_size = qdrant_config.get("vector_size")
        if not self.vector_size:
            raise ValueError("❌ CONFIGURATION ERROR: 'databases.qdrant.vector_size' not found in settings.yaml")
        
        self.similarity_threshold = qdrant_config.get("similarity_threshold")
        if not self.similarity_threshold:
            raise ValueError("❌ CONFIGURATION ERROR: 'databases.qdrant.similarity_threshold' not found in settings.yaml")
        
    async def connect(self):
        """Initialize Qdrant connection and embedding model"""
        try:
            # Initialize Qdrant client with gRPC support
            if self.prefer_grpc:
                try:
                    # Try gRPC connection first
                    self.client = QdrantClient(
                        host=self.host,
                        grpc_port=self.grpc_port,
                        prefer_grpc=True,
                        timeout=self.timeout
                    )
                    self.logger.info(f"✅ Qdrant connected via gRPC on port {self.grpc_port}")
                except Exception as grpc_error:
                    # Fall back to HTTP if gRPC fails
                    self.logger.warning(f"⚠️ gRPC connection failed, falling back to HTTP: {grpc_error}")
                    self.client = QdrantClient(
                        host=self.host,
                        port=self.port,
                        timeout=self.timeout
                    )
                    self.logger.info(f"✅ Qdrant connected via HTTP on port {self.port}")
            else:
                # Use HTTP connection
                self.client = QdrantClient(
                    host=self.host,
                    port=self.port,
                    timeout=self.timeout
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
                user_id=memory.user_id,
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
        user_id: str = "default",
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
                    key="user_id",
                    match=models.MatchValue(value=user_id)
                )
            ]
            
            if memory_types:
                filter_conditions.append(
                    FieldCondition(
                        key="memory_type",
                        match=models.MatchAny(any=[mt.value for mt in memory_types])
                    )
                )
            
            # Search with vector similarity
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=Filter(must=filter_conditions),
                limit=limit,
                score_threshold=min_similarity or self.similarity_threshold
            )
            
            # Convert to QdrantMemoryItem objects
            memories = []
            for result in search_results:
                try:
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
        user_id: str = "default",
        limit: int = 100
    ) -> List[QdrantMemoryItem]:
        """Get all memories of a specific type"""
        try:
            filter_conditions = [
                FieldCondition(
                    key="user_id",
                    match=models.MatchValue(value=user_id)
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
    
    async def get_memory_stats(self, user_id: str = "default") -> Dict[str, Any]:
        """Get Qdrant memory statistics"""
        try:
            # Get collection info
            collection_info = self.client.get_collection(self.collection_name)
            
            # Count memories by type for user
            type_counts = {}
            for memory_type in MemoryType:
                memories = await self.get_memories_by_type(memory_type, user_id, limit=1000)
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
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "connection": "failed",
                "error": str(e)
            }