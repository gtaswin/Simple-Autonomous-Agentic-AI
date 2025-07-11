"""
Centralized Transformers Service for Local AI Processing

This service provides fast, local AI processing for classification, analysis, and 
content extraction tasks to reduce LLM API calls and improve performance.
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from functools import lru_cache
import logging
from dataclasses import dataclass
from datetime import datetime

try:
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers not available. Install with: pip install transformers torch")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️ Sentence-transformers not available. Install with: pip install sentence-transformers")


@dataclass
class TransformerResult:
    """Standardized result from transformer operations"""
    label: str
    confidence: float
    processing_time: float
    additional_data: Dict[str, Any] = None


class TransformersService:
    """Centralized hub for all local transformer operations"""
    
    def __init__(self, config=None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model cache
        self._models = {}
        self._device = 0 if torch.cuda.is_available() else -1
        
        # Performance metrics
        self.metrics = {
            "total_calls": 0,
            "total_time": 0.0,
            "cache_hits": 0,
            "model_loads": 0
        }
        
        if TRANSFORMERS_AVAILABLE:
            self._initialize_models()
        else:
            self.logger.warning("Transformers not available - falling back to keyword methods")
    
    def _initialize_models(self):
        """Initialize core models at startup using configuration"""
        try:
            print("🤖 Loading transformer models from configuration...")
            
            # Get model configurations from settings
            models_config = self._get_models_config()
            performance_config = self._get_performance_config()
            
            # Configure device
            device = self._configure_device(performance_config.get("device", "auto"))
            
            # Configure cache size
            cache_size = performance_config.get("cache_size", 1000)
            # Update the lru_cache maxsize for methods that use it
            
            # Memory classification model
            memory_model = models_config.get("memory_classifier")
            if not memory_model:
                raise ValueError("❌ CONFIGURATION ERROR: 'transformers.models.memory_classifier' not found in settings.yaml")
            self._models['memory_classifier'] = pipeline(
                "zero-shot-classification",
                model=memory_model,
                device=device
            )
            print(f"  ✅ Memory classifier: {memory_model}")
            
            # Intent classification
            intent_model = models_config.get("intent_classifier")
            if not intent_model:
                raise ValueError("❌ CONFIGURATION ERROR: 'transformers.models.intent_classifier' not found in settings.yaml")
            self._models['intent_classifier'] = pipeline(
                "zero-shot-classification", 
                model=intent_model,
                device=device
            )
            print(f"  ✅ Intent classifier: {intent_model}")
            
            # Sentiment analysis
            sentiment_model = models_config.get("sentiment_analyzer")
            if not sentiment_model:
                raise ValueError("❌ CONFIGURATION ERROR: 'transformers.models.sentiment_analyzer' not found in settings.yaml")
            self._models['sentiment'] = pipeline(
                "sentiment-analysis",
                model=sentiment_model,
                device=device
            )
            print(f"  ✅ Sentiment analyzer: {sentiment_model}")
            
            # Named Entity Recognition
            ner_model = models_config.get("entity_extractor")
            if not ner_model:
                raise ValueError("❌ CONFIGURATION ERROR: 'transformers.models.entity_extractor' not found in settings.yaml")
            self._models['ner'] = pipeline(
                "ner",
                model=ner_model,
                device=device,
                aggregation_strategy="simple"
            )
            print(f"  ✅ Entity extractor: {ner_model}")
            
            # Summarization
            summarizer_model = models_config.get("summarizer")
            if not summarizer_model:
                raise ValueError("❌ CONFIGURATION ERROR: 'transformers.models.summarizer' not found in settings.yaml")
            self._models['summarizer'] = pipeline(
                "summarization",
                model=summarizer_model,
                device=device
            )
            print(f"  ✅ Summarizer: {summarizer_model}")
            
            # Embeddings model
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                embedder_model = models_config.get("embedder")
                if not embedder_model:
                    raise ValueError("❌ CONFIGURATION ERROR: 'transformers.models.embedder' not found in settings.yaml")
                self._models['embedder'] = SentenceTransformer(embedder_model)
                print(f"  ✅ Embedder: {embedder_model}")
            
            print("✅ Transformer models loaded successfully")
            self.metrics["model_loads"] = len(self._models)
            
        except Exception as e:
            self.logger.error(f"Failed to load transformer models: {e}")
            self._models = {}
    
    def _get_models_config(self) -> Dict[str, str]:
        """Get model configurations from settings"""
        if self.config and hasattr(self.config, 'get'):
            return self.config.get("transformers.models", {})
        return {}
    
    def _get_performance_config(self) -> Dict[str, Any]:
        """Get performance configurations from settings"""
        if self.config and hasattr(self.config, 'get'):
            return self.config.get("transformers.performance", {})
        return {}
    
    def _get_fallback_config(self) -> Dict[str, Any]:
        """Get fallback configurations from settings"""
        if self.config and hasattr(self.config, 'get'):
            return self.config.get("transformers.fallback", {})
        return {}
    
    def _configure_device(self, device_config: str) -> int:
        """Configure device based on settings"""
        if device_config == "auto":
            return 0 if torch.cuda.is_available() else -1
        elif device_config == "cpu":
            return -1
        elif device_config == "cuda":
            return 0 if torch.cuda.is_available() else -1
        elif device_config.isdigit():
            return int(device_config)
        else:
            return 0 if torch.cuda.is_available() else -1
    
    @lru_cache(maxsize=1000)
    def classify_memory_type(self, message: str) -> TransformerResult:
        """Classify message for memory storage type"""
        start_time = datetime.now()
        
        if not self._models.get('memory_classifier'):
            return self._fallback_memory_classification(message)
        
        try:
            memory_categories = [
                'personal information',     # Name, job, location, etc.
                'goal setting',            # Future plans, aspirations
                'preference statement',    # Likes, dislikes, choices
                'experience sharing',      # Past events, stories
                'skill learning',          # Learning requests, how-to
                'general conversation'     # Greetings, casual chat
            ]
            
            result = self._models['memory_classifier'](message, memory_categories)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_metrics(processing_time, cache_hit=False)
            
            return TransformerResult(
                label=result['labels'][0].replace(' ', '_'),
                confidence=result['scores'][0],
                processing_time=processing_time,
                additional_data={
                    'memory_type': self._map_to_memory_type(result['labels'][0]),
                    'importance_score': min(result['scores'][0] * 1.2, 1.0),  # Boost confidence for importance
                    'all_scores': dict(zip(result['labels'], result['scores']))
                }
            )
            
        except Exception as e:
            self.logger.error(f"Memory classification failed: {e}")
            return self._fallback_memory_classification(message)
    
    @lru_cache(maxsize=500)
    def classify_intent(self, message: str) -> TransformerResult:
        """Classify user intent for agent routing"""
        start_time = datetime.now()
        
        if not self._models.get('intent_classifier'):
            return self._fallback_intent_classification(message)
        
        try:
            intent_categories = [
                'simple question',         # Basic information requests
                'complex research',        # Multi-step research tasks
                'personal assistance',     # Memory, preferences, goals
                'creative task',          # Writing, brainstorming
                'technical query',        # Programming, troubleshooting
                'planning request'        # Strategic planning, goal setting
            ]
            
            result = self._models['intent_classifier'](message, intent_categories)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_metrics(processing_time, cache_hit=False)
            
            return TransformerResult(
                label=result['labels'][0].replace(' ', '_'),
                confidence=result['scores'][0],
                processing_time=processing_time,
                additional_data={
                    'routing_decision': self._map_to_agent_strategy(result['labels'][0]),
                    'all_scores': dict(zip(result['labels'], result['scores']))
                }
            )
            
        except Exception as e:
            self.logger.error(f"Intent classification failed: {e}")
            return self._fallback_intent_classification(message)
    
    def analyze_sentiment(self, text: str) -> TransformerResult:
        """Analyze sentiment of text"""
        start_time = datetime.now()
        
        if not self._models.get('sentiment'):
            return self._fallback_sentiment_analysis(text)
        
        try:
            result = self._models['sentiment'](text)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_metrics(processing_time, cache_hit=False)
            
            return TransformerResult(
                label=result[0]['label'],
                confidence=result[0]['score'],
                processing_time=processing_time,
                additional_data={
                    'sentiment_impact': self._calculate_sentiment_impact(result[0])
                }
            )
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {e}")
            return self._fallback_sentiment_analysis(text)
    
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract named entities from text"""
        start_time = datetime.now()
        
        if not self._models.get('ner'):
            return self._fallback_entity_extraction(text)
        
        try:
            entities = self._models['ner'](text)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_metrics(processing_time, cache_hit=False)
            
            # Group entities by type
            grouped_entities = {}
            for entity in entities:
                entity_type = entity['entity_group']
                if entity_type not in grouped_entities:
                    grouped_entities[entity_type] = []
                grouped_entities[entity_type].append({
                    'text': entity['word'],
                    'confidence': entity['score'],
                    'start': entity.get('start', 0),
                    'end': entity.get('end', 0)
                })
            
            return {
                'entities': grouped_entities,
                'processing_time': processing_time,
                'total_entities': len(entities)
            }
            
        except Exception as e:
            self.logger.error(f"Entity extraction failed: {e}")
            return self._fallback_entity_extraction(text)
    
    def generate_summary(self, text: str, max_length: int = 150) -> Dict[str, Any]:
        """Generate summary of text"""
        start_time = datetime.now()
        
        if not self._models.get('summarizer'):
            return self._fallback_summarization(text)
        
        try:
            # Skip summarization for very short text
            if len(text.split()) < 50:
                return {
                    'summary': text[:max_length],
                    'processing_time': 0.001,
                    'compression_ratio': 1.0
                }
            
            result = self._models['summarizer'](
                text, 
                max_length=max_length, 
                min_length=max(30, max_length // 3),
                do_sample=False
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_metrics(processing_time, cache_hit=False)
            
            summary_text = result[0]['summary_text']
            
            return {
                'summary': summary_text,
                'processing_time': processing_time,
                'compression_ratio': len(summary_text) / len(text),
                'original_length': len(text),
                'summary_length': len(summary_text)
            }
            
        except Exception as e:
            self.logger.error(f"Summarization failed: {e}")
            return self._fallback_summarization(text)
    
    def get_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Generate embeddings for semantic search"""
        if not self._models.get('embedder'):
            return None
        
        try:
            embeddings = self._models['embedder'].encode(texts)
            return embeddings.tolist()
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            return None
    
    # Helper methods
    def _map_to_memory_type(self, category: str) -> str:
        """Map classification to memory types"""
        mapping = {
            'personal information': 'semantic',
            'goal setting': 'prospective',
            'preference statement': 'semantic',
            'experience sharing': 'episodic',
            'skill learning': 'procedural',
            'general conversation': 'working'
        }
        return mapping.get(category, 'working')
    
    def _map_to_agent_strategy(self, intent: str) -> str:
        """Map intent to agent strategy"""
        mapping = {
            'simple question': 'direct_response',
            'complex research': 'expert_team_collaboration',
            'personal assistance': 'memory_focused',
            'creative task': 'expert_team_collaboration',
            'technical query': 'expert_team_collaboration',
            'planning request': 'expert_team_collaboration'
        }
        return mapping.get(intent, 'direct_response')
    
    def _calculate_sentiment_impact(self, sentiment_result: Dict) -> float:
        """Calculate sentiment impact score"""
        label = sentiment_result['label'].lower()
        score = sentiment_result['score']
        
        if 'positive' in label:
            return score
        elif 'negative' in label:
            return -score
        else:
            return 0.0
    
    def _update_metrics(self, processing_time: float, cache_hit: bool = False):
        """Update performance metrics"""
        self.metrics['total_calls'] += 1
        self.metrics['total_time'] += processing_time
        if cache_hit:
            self.metrics['cache_hits'] += 1
    
    # Fallback methods for when transformers are not available
    def _fallback_memory_classification(self, message: str) -> TransformerResult:
        """Keyword-based fallback for memory classification"""
        message_lower = message.lower()
        
        # Use configuration-based fallback
        fallback_config = self._get_fallback_config()
        confidence_threshold = fallback_config.get("confidence_threshold")
        if confidence_threshold is None:
            raise ValueError("❌ CONFIGURATION ERROR: 'transformers.fallback.confidence_threshold' not found in settings.yaml")
        
        if any(word in message_lower for word in ['goal', 'want', 'plan', 'future']):
            memory_type = 'prospective'
        elif any(word in message_lower for word in ['name', 'job', 'live', 'age']):
            memory_type = 'semantic'
        elif any(word in message_lower for word in ['yesterday', 'today', 'happened']):
            memory_type = 'episodic'
        elif any(word in message_lower for word in ['how', 'learn', 'teach']):
            memory_type = 'procedural'
        else:
            memory_type = 'working'
        
        return TransformerResult(
            label=memory_type,
            confidence=confidence_threshold,
            processing_time=0.001,
            additional_data={
                'memory_type': memory_type, 
                'importance_score': 0.5,
                'fallback_source': 'keyword_classification'
            }
        )
    
    def _fallback_intent_classification(self, message: str) -> TransformerResult:
        """Keyword-based fallback for intent classification"""
        message_lower = message.lower()
        
        # Use configuration-based fallback
        fallback_config = self._get_fallback_config()
        confidence_threshold = fallback_config.get("confidence_threshold")
        if confidence_threshold is None:
            raise ValueError("❌ CONFIGURATION ERROR: 'transformers.fallback.confidence_threshold' not found in settings.yaml")
        
        if any(word in message_lower for word in ['research', 'find', 'search', 'complex']):
            intent = 'complex_research'
        elif any(word in message_lower for word in ['remember', 'prefer', 'goal']):
            intent = 'personal_assistance'
        elif any(word in message_lower for word in ['create', 'write', 'generate']):
            intent = 'creative_task'
        elif any(word in message_lower for word in ['code', 'program', 'technical']):
            intent = 'technical_query'
        else:
            intent = 'simple_question'
        
        return TransformerResult(
            label=intent,
            confidence=confidence_threshold,
            processing_time=0.001,
            additional_data={
                'routing_decision': self._map_to_agent_strategy(intent),
                'fallback_source': 'keyword_classification'
            }
        )
    
    def _fallback_sentiment_analysis(self, text: str) -> TransformerResult:
        """Keyword-based fallback for sentiment analysis"""
        positive_words = ['good', 'great', 'excellent', 'happy', 'love', 'amazing']
        negative_words = ['bad', 'terrible', 'hate', 'awful', 'horrible', 'sad']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = 'POSITIVE'
            confidence = 0.6
        elif negative_count > positive_count:
            sentiment = 'NEGATIVE'
            confidence = 0.6
        else:
            sentiment = 'NEUTRAL'
            confidence = 0.5
        
        return TransformerResult(
            label=sentiment,
            confidence=confidence,
            processing_time=0.001,
            additional_data={'sentiment_impact': confidence if sentiment == 'POSITIVE' else -confidence if sentiment == 'NEGATIVE' else 0}
        )
    
    def _fallback_entity_extraction(self, text: str) -> Dict[str, Any]:
        """Simple fallback entity extraction"""
        return {
            'entities': {},
            'processing_time': 0.001,
            'total_entities': 0
        }
    
    def _fallback_summarization(self, text: str) -> Dict[str, Any]:
        """Simple truncation fallback"""
        summary = text[:150] + "..." if len(text) > 150 else text
        return {
            'summary': summary,
            'processing_time': 0.001,
            'compression_ratio': len(summary) / len(text),
            'original_length': len(text),
            'summary_length': len(summary)
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        avg_time = self.metrics['total_time'] / max(1, self.metrics['total_calls'])
        cache_hit_rate = self.metrics['cache_hits'] / max(1, self.metrics['total_calls'])
        
        return {
            **self.metrics,
            'average_processing_time': avg_time,
            'cache_hit_rate': cache_hit_rate,
            'models_loaded': len(self._models),
            'transformers_available': TRANSFORMERS_AVAILABLE
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for transformers service"""
        try:
            # Test a simple classification
            test_result = self.classify_memory_type("Hello")
            
            return {
                'status': 'healthy',
                'transformers_available': TRANSFORMERS_AVAILABLE,
                'models_loaded': len(self._models),
                'test_classification_time': test_result.processing_time,
                'metrics': self.get_metrics()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'transformers_available': TRANSFORMERS_AVAILABLE,
                'models_loaded': len(self._models)
            }


# Global instance
_transformers_service = None

def get_transformers_service(config=None) -> TransformersService:
    """Get global transformers service instance"""
    global _transformers_service
    if _transformers_service is None:
        _transformers_service = TransformersService(config)
    return _transformers_service