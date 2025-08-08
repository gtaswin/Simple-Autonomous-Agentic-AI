#!/usr/bin/env python3
"""
Phase 5: TypeScript Interface Generator
Generates TypeScript interfaces from Pydantic schemas for frontend integration.
"""

import os
import json
import importlib
import inspect
from datetime import datetime
from typing import Dict, List, Any, Optional, get_type_hints
from pathlib import Path

# Import Pydantic and our schemas
from pydantic import BaseModel
from core.output_schemas import *


class TypeScriptGenerator:
    """Generates TypeScript interfaces from Pydantic models"""
    
    def __init__(self):
        self.type_mappings = {
            'str': 'string',
            'int': 'number',
            'float': 'number',
            'bool': 'boolean',
            'datetime': 'string',  # ISO format
            'Any': 'any',
            'Dict': 'Record<string, any>',
            'List': 'Array',
        }
        
        self.generated_interfaces = set()
        self.output_lines = []
    
    def generate_interfaces(self) -> str:
        """Generate all TypeScript interfaces"""
        
        # Header
        self.output_lines.extend([
            "/**",
            " * Phase 5: Auto-generated TypeScript Interfaces",
            " * Generated from Pydantic schemas for structured agent outputs",
            f" * Generated at: {datetime.now().isoformat()}",
            " * DO NOT EDIT MANUALLY - This file is auto-generated",
            " */",
            "",
            "// ================================================================================",
            "// ENUMS",
            "// ================================================================================",
            ""
        ])
        
        # Generate enums first
        self._generate_enums()
        
        self.output_lines.extend([
            "",
            "// ================================================================================", 
            "// INTERFACES",
            "// ================================================================================",
            ""
        ])
        
        # Generate interfaces
        self._generate_all_interfaces()
        
        # Footer with utility types
        self.output_lines.extend([
            "",
            "// ================================================================================",
            "// UTILITY TYPES",
            "// ================================================================================",
            "",
            "export type AgentOutputUnion = MemoryReaderOutput | KnowledgeAgentOutput | OrganizerAgentOutput | MemoryWriterOutput;",
            "",
            "export type WorkflowOutputUnion = WorkflowExecutionOutput | AutonomousOperationOutput;",
            "",
            "export interface ApiResponse<T = any> {",
            "  response: string;",
            "  metadata: Record<string, any>;",
            "  timestamp: string;",
            "  user_name?: string;",
            "  structured_output?: T;",
            "}",
            "",
            "export interface StructuredApiResponse extends ApiResponse {",
            "  structured_output: AgentOutputUnion | WorkflowOutputUnion;",
            "}",
            ""
        ])
        
        return "\n".join(self.output_lines)
    
    def _generate_enums(self):
        """Generate TypeScript enums from Python enums"""
        
        enums_to_generate = [
            ('AgentType', AgentType),
            ('ProcessingModel', ProcessingModel), 
            ('OperationStatus', OperationStatus),
            ('MemoryType', MemoryType),
            ('SearchResultSource', SearchResultSource),
            ('SynthesisQuality', SynthesisQuality),
            ('FactType', FactType),
            ('WorkflowPattern', WorkflowPattern),
            ('AutonomousTrigger', AutonomousTrigger),
            ('AutonomousOperationType', AutonomousOperationType)
        ]
        
        for enum_name, enum_class in enums_to_generate:
            self.output_lines.append(f"export enum {enum_name} {{")
            
            for member in enum_class:
                # Convert snake_case to UPPER_CASE for enum members
                ts_member_name = member.name
                ts_member_value = member.value
                self.output_lines.append(f"  {ts_member_name} = '{ts_member_value}',")
            
            self.output_lines.append("}")
            self.output_lines.append("")
    
    def _generate_all_interfaces(self):
        """Generate all interfaces from Pydantic models"""
        
        models_to_generate = [
            BaseAgentOutput,
            MemoryItem,
            MemoryReaderOutput,
            KnowledgeSearchResult,
            KnowledgeAgentOutput,
            ContextQualityMetrics,
            OrganizerAgentOutput,
            ExtractedFact,
            MemoryStorageStats,
            MemoryWriterOutput,
            ParallelExecutionMetrics,
            WorkflowExecutionOutput,
            AutonomousInsight,
            AutonomousOperationOutput
        ]
        
        for model_class in models_to_generate:
            self._generate_interface(model_class)
    
    def _generate_interface(self, model_class: BaseModel):
        """Generate TypeScript interface from Pydantic model"""
        
        class_name = model_class.__name__
        
        if class_name in self.generated_interfaces:
            return
        
        self.generated_interfaces.add(class_name)
        
        # Interface declaration
        extends_clause = ""
        if hasattr(model_class, '__bases__') and model_class.__bases__:
            for base in model_class.__bases__:
                if base != BaseModel and hasattr(base, '__name__'):
                    base_name = base.__name__
                    if base_name != 'BaseModel':
                        extends_clause = f" extends {base_name}"
                        break
        
        self.output_lines.append(f"export interface {class_name}{extends_clause} {{")
        
        # Get field information
        if hasattr(model_class, '__fields__'):
            fields = model_class.__fields__
        else:
            # For newer Pydantic versions
            fields = getattr(model_class, 'model_fields', {})
        
        # Generate fields
        for field_name, field_info in fields.items():
            ts_type = self._convert_python_type_to_typescript(field_info)
            optional = self._is_field_optional(field_info)
            
            # Add JSDoc comment if field has description
            description = self._get_field_description(field_info)
            if description:
                self.output_lines.append(f"  /** {description} */")
            
            optional_suffix = "?" if optional else ""
            self.output_lines.append(f"  {field_name}{optional_suffix}: {ts_type};")
        
        self.output_lines.append("}")
        self.output_lines.append("")
    
    def _convert_python_type_to_typescript(self, field_info) -> str:
        """Convert Python type annotations to TypeScript types"""
        
        # Handle different Pydantic versions
        if hasattr(field_info, 'type_'):
            python_type = field_info.type_
        elif hasattr(field_info, 'annotation'):
            python_type = field_info.annotation
        else:
            return 'any'
        
        # Handle type string representation
        type_str = str(python_type)
        
        # Direct mappings
        for py_type, ts_type in self.type_mappings.items():
            if py_type in type_str:
                if py_type == 'List':
                    # Extract inner type for arrays
                    if '[' in type_str and ']' in type_str:
                        inner_type = type_str[type_str.find('[')+1:type_str.rfind(']')]
                        inner_ts_type = self._map_simple_type(inner_type)
                        return f"{inner_ts_type}[]"
                    return "any[]"
                elif py_type == 'Dict':
                    # Extract value type for objects
                    if '[str, ' in type_str:
                        value_type = type_str[type_str.find('[str, ')+5:type_str.rfind(']')]
                        value_ts_type = self._map_simple_type(value_type)
                        return f"Record<string, {value_ts_type}>"
                    return "Record<string, any>"
                else:
                    return ts_type
        
        # Handle Union types (Optional)
        if 'Union' in type_str or 'Optional' in type_str:
            if 'NoneType' in type_str:
                # Extract non-None type
                clean_type = type_str.replace('Union[', '').replace('Optional[', '').replace(', NoneType]', '').replace(']', '')
                ts_type = self._map_simple_type(clean_type)
                return f"{ts_type} | null"
        
        # Handle custom model types (other Pydantic models)
        if hasattr(python_type, '__name__'):
            return python_type.__name__
        
        # Handle enum types
        if 'Enum' in type_str:
            enum_name = type_str.split('.')[-1].replace("'", "").replace(">", "")
            return enum_name
        
        return 'any'
    
    def _map_simple_type(self, type_str: str) -> str:
        """Map simple type strings to TypeScript"""
        type_str = type_str.strip().replace("'", "")
        
        simple_mappings = {
            'str': 'string',
            'int': 'number', 
            'float': 'number',
            'bool': 'boolean',
            'datetime': 'string',
            'Any': 'any'
        }
        
        return simple_mappings.get(type_str, type_str)
    
    def _is_field_optional(self, field_info) -> bool:
        """Check if field is optional"""
        if hasattr(field_info, 'default') and field_info.default is not None:
            return True
        if hasattr(field_info, 'is_required'):
            return not field_info.is_required
        return False
    
    def _get_field_description(self, field_info) -> Optional[str]:
        """Get field description from Pydantic field info"""
        if hasattr(field_info, 'field_info') and hasattr(field_info.field_info, 'description'):
            return field_info.field_info.description
        if hasattr(field_info, 'description'):
            return field_info.description
        return None


def main():
    """Main function to generate TypeScript interfaces"""
    
    print("🚀 Generating TypeScript interfaces from Pydantic schemas...")
    
    # Create generator
    generator = TypeScriptGenerator()
    
    # Generate interfaces
    typescript_content = generator.generate_interfaces()
    
    # Determine output path
    backend_dir = Path(__file__).parent.parent
    frontend_dir = backend_dir.parent / "frontend"
    output_dir = frontend_dir / "src" / "types"
    output_file = output_dir / "agent-outputs.ts"
    
    # Create directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write TypeScript file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(typescript_content)
    
    print(f"✅ TypeScript interfaces generated: {output_file}")
    print(f"📊 Generated {len(generator.generated_interfaces)} interfaces")
    print(f"📝 Total lines: {len(generator.output_lines)}")
    
    # Also generate a summary file
    summary_file = output_dir / "generation-summary.json"
    summary = {
        "generated_at": datetime.now().isoformat(),
        "interfaces_count": len(generator.generated_interfaces),
        "total_lines": len(generator.output_lines),
        "generated_interfaces": list(generator.generated_interfaces),
        "source_file": str(output_file.relative_to(frontend_dir))
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📋 Generation summary: {summary_file}")


if __name__ == "__main__":
    main()