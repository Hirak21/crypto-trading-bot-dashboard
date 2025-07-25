"""
Serialization utilities for trading models.

This module provides advanced serialization and deserialization capabilities
for trading objects, including batch operations and format conversions.
"""

import json
import pickle
import csv
from typing import Any, Dict, List, Union, Type, IO
from datetime import datetime
from pathlib import Path
import logging

from .trading import TradingSignal, Position, Trade, MarketData, OrderBook


class SerializationError(Exception):
    """Custom exception for serialization errors."""
    pass


class TradingDataSerializer:
    """Advanced serializer for trading data objects."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Supported object types
        self.supported_types = {
            'TradingSignal': TradingSignal,
            'Position': Position,
            'Trade': Trade,
            'MarketData': MarketData,
            'OrderBook': OrderBook
        }
    
    def to_json(self, obj: Any, indent: int = 2) -> str:
        """Serialize object to JSON string."""
        try:
            if hasattr(obj, 'to_dict'):
                data = obj.to_dict()
                data['_type'] = obj.__class__.__name__
                return json.dumps(data, indent=indent, default=self._json_serializer)
            else:
                raise SerializationError(f"Object {type(obj)} does not support JSON serialization")
        
        except Exception as e:
            raise SerializationError(f"JSON serialization failed: {e}")
    
    def from_json(self, json_str: str) -> Any:
        """Deserialize object from JSON string."""
        try:
            data = json.loads(json_str)
            
            if '_type' not in data:
                raise SerializationError("JSON data missing type information")
            
            obj_type_name = data.pop('_type')
            
            if obj_type_name not in self.supported_types:
                raise SerializationError(f"Unsupported object type: {obj_type_name}")
            
            obj_type = self.supported_types[obj_type_name]
            
            if hasattr(obj_type, 'from_dict'):
                return obj_type.from_dict(data)
            else:
                raise SerializationError(f"Object type {obj_type_name} does not support deserialization")
        
        except json.JSONDecodeError as e:
            raise SerializationError(f"Invalid JSON format: {e}")
        except Exception as e:
            raise SerializationError(f"JSON deserialization failed: {e}")
    
    def to_binary(self, obj: Any) -> bytes:
        """Serialize object to binary format using pickle."""
        try:
            return pickle.dumps(obj)
        except Exception as e:
            raise SerializationError(f"Binary serialization failed: {e}")
    
    def from_binary(self, data: bytes) -> Any:
        """Deserialize object from binary format."""
        try:
            return pickle.loads(data)
        except Exception as e:
            raise SerializationError(f"Binary deserialization failed: {e}")
    
    def to_csv_row(self, obj: Any) -> Dict[str, Any]:
        """Convert object to CSV row format."""
        try:
            if hasattr(obj, 'to_dict'):
                data = obj.to_dict()
                # Flatten nested dictionaries
                flattened = self._flatten_dict(data)
                return flattened
            else:
                raise SerializationError(f"Object {type(obj)} does not support CSV serialization")
        
        except Exception as e:
            raise SerializationError(f"CSV row conversion failed: {e}")
    
    def batch_to_json(self, objects: List[Any], indent: int = 2) -> str:
        """Serialize list of objects to JSON array."""
        try:
            serialized_objects = []
            
            for obj in objects:
                if hasattr(obj, 'to_dict'):
                    data = obj.to_dict()
                    data['_type'] = obj.__class__.__name__
                    serialized_objects.append(data)
                else:
                    raise SerializationError(f"Object {type(obj)} does not support JSON serialization")
            
            return json.dumps(serialized_objects, indent=indent, default=self._json_serializer)
        
        except Exception as e:
            raise SerializationError(f"Batch JSON serialization failed: {e}")
    
    def batch_from_json(self, json_str: str) -> List[Any]:
        """Deserialize list of objects from JSON array."""
        try:
            data_list = json.loads(json_str)
            
            if not isinstance(data_list, list):
                raise SerializationError("JSON data must be an array for batch deserialization")
            
            objects = []
            
            for i, data in enumerate(data_list):
                try:
                    if '_type' not in data:
                        raise SerializationError(f"Item {i} missing type information")
                    
                    obj_type_name = data.pop('_type')
                    
                    if obj_type_name not in self.supported_types:
                        raise SerializationError(f"Item {i} has unsupported type: {obj_type_name}")
                    
                    obj_type = self.supported_types[obj_type_name]
                    
                    if hasattr(obj_type, 'from_dict'):
                        objects.append(obj_type.from_dict(data))
                    else:
                        raise SerializationError(f"Type {obj_type_name} does not support deserialization")
                
                except Exception as e:
                    self.logger.warning(f"Failed to deserialize item {i}: {e}")
                    continue
            
            return objects
        
        except json.JSONDecodeError as e:
            raise SerializationError(f"Invalid JSON format: {e}")
        except Exception as e:
            raise SerializationError(f"Batch JSON deserialization failed: {e}")
    
    def to_csv_file(self, objects: List[Any], file_path: Union[str, Path]) -> None:
        """Export objects to CSV file."""
        try:
            if not objects:
                raise SerializationError("No objects to export")
            
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert all objects to CSV rows
            csv_rows = []
            for obj in objects:
                csv_rows.append(self.to_csv_row(obj))
            
            # Get all unique field names
            fieldnames = set()
            for row in csv_rows:
                fieldnames.update(row.keys())
            
            fieldnames = sorted(list(fieldnames))
            
            # Write CSV file
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)
            
            self.logger.info(f"Exported {len(objects)} objects to {file_path}")
        
        except Exception as e:
            raise SerializationError(f"CSV export failed: {e}")
    
    def to_json_file(self, objects: Union[Any, List[Any]], file_path: Union[str, Path], 
                     indent: int = 2) -> None:
        """Export objects to JSON file."""
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if isinstance(objects, list):
                json_str = self.batch_to_json(objects, indent)
            else:
                json_str = self.to_json(objects, indent)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(json_str)
            
            count = len(objects) if isinstance(objects, list) else 1
            self.logger.info(f"Exported {count} objects to {file_path}")
        
        except Exception as e:
            raise SerializationError(f"JSON file export failed: {e}")
    
    def from_json_file(self, file_path: Union[str, Path]) -> Union[Any, List[Any]]:
        """Import objects from JSON file."""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise SerializationError(f"File not found: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                json_str = f.read()
            
            # Try to determine if it's a single object or array
            data = json.loads(json_str)
            
            if isinstance(data, list):
                return self.batch_from_json(json_str)
            else:
                return self.from_json(json_str)
        
        except Exception as e:
            raise SerializationError(f"JSON file import failed: {e}")
    
    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for special types."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested dictionary for CSV export."""
        items = []
        
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert lists to comma-separated strings
                items.append((new_key, ','.join(map(str, v))))
            else:
                items.append((new_key, v))
        
        return dict(items)


# Global serializer instance
serializer = TradingDataSerializer()


# Convenience functions
def to_json(obj: Any, indent: int = 2) -> str:
    """Serialize object to JSON string."""
    return serializer.to_json(obj, indent)


def from_json(json_str: str) -> Any:
    """Deserialize object from JSON string."""
    return serializer.from_json(json_str)


def to_binary(obj: Any) -> bytes:
    """Serialize object to binary format."""
    return serializer.to_binary(obj)


def from_binary(data: bytes) -> Any:
    """Deserialize object from binary format."""
    return serializer.from_binary(data)


def batch_to_json(objects: List[Any], indent: int = 2) -> str:
    """Serialize list of objects to JSON array."""
    return serializer.batch_to_json(objects, indent)


def batch_from_json(json_str: str) -> List[Any]:
    """Deserialize list of objects from JSON array."""
    return serializer.batch_from_json(json_str)


def export_to_csv(objects: List[Any], file_path: Union[str, Path]) -> None:
    """Export objects to CSV file."""
    serializer.to_csv_file(objects, file_path)


def export_to_json(objects: Union[Any, List[Any]], file_path: Union[str, Path], 
                   indent: int = 2) -> None:
    """Export objects to JSON file."""
    serializer.to_json_file(objects, file_path, indent)


def import_from_json(file_path: Union[str, Path]) -> Union[Any, List[Any]]:
    """Import objects from JSON file."""
    return serializer.from_json_file(file_path)