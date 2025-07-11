"""
JSON Serialization utilities for WebSocket communication
Handles datetime objects and other non-JSON-serializable types
"""
import json
from datetime import datetime, date
from typing import Any, Dict, List
from uuid import UUID
from dataclasses import is_dataclass, asdict
from pydantic import BaseModel


class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles Python objects properly"""
    
    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, date):
            return obj.isoformat()
        elif isinstance(obj, UUID):
            return str(obj)
        elif is_dataclass(obj):
            return asdict(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        else:
            # Let the base class handle the error
            return super().default(obj)


def safe_json_dumps(data: Any, **kwargs) -> str:
    """
    Safely serialize data to JSON string.
    Handles datetime objects and other non-serializable types.
    """
    try:
        return json.dumps(data, cls=JSONEncoder, ensure_ascii=False, **kwargs)
    except Exception as e:
        # Fallback: convert to string representation
        try:
            cleaned_data = _clean_for_json(data)
            return json.dumps(cleaned_data, cls=JSONEncoder, ensure_ascii=False, **kwargs)
        except Exception as fallback_e:
            # Last resort: return error message
            return json.dumps({
                "error": "Serialization failed",
                "original_error": str(e),
                "fallback_error": str(fallback_e),
                "data_type": type(data).__name__
            })


def _clean_for_json(obj: Any) -> Any:
    """Recursively clean object for JSON serialization"""
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, UUID):
        return str(obj)
    # Handle numpy types to fix Pydantic serialization warnings
    elif hasattr(obj, 'dtype') and hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    elif hasattr(obj, 'tolist'):  # numpy array
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _clean_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_clean_for_json(item) for item in obj]
    elif is_dataclass(obj):
        return _clean_for_json(asdict(obj))
    elif isinstance(obj, BaseModel): # Handle Pydantic models
        return _clean_for_json(obj.model_dump())
    elif hasattr(obj, '__dict__'):
        return _clean_for_json(obj.__dict__)
    elif hasattr(obj, 'to_dict'):
        return _clean_for_json(obj.to_dict())
    else:
        # Convert to string as last resort
        return str(obj)


def prepare_websocket_message(message_type: str, data: Any, user_id: str = None) -> Dict[str, Any]:
    """
    Prepare a standardized WebSocket message with proper serialization.
    
    Args:
        message_type: Type of message (e.g., "thought", "decision", "status")
        data: Message payload
        user_id: Optional user ID
        
    Returns:
        Dictionary ready for JSON serialization
    """
    message = {
        "type": message_type,
        "data": _clean_for_json(data),
        "timestamp": datetime.now().isoformat()
    }
    
    if user_id:
        message["user_id"] = user_id
        
    return message


def safe_serialize(data: Any) -> Any:
    """
    Safely serialize data for storage or transmission.
    Similar to _clean_for_json but returns the cleaned object instead of JSON string.
    """
    return _clean_for_json(data)