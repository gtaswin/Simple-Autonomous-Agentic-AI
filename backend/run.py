#!/usr/bin/env python3
"""
Run script for the Autonomous Agentic AI System
Handles Python path and uvicorn startup with 3-agent architecture
"""

import os
import sys

# Add backend directory to Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

if __name__ == "__main__":
    import uvicorn
    import logging
    
    print("🚀 Starting Autonomous Agentic AI System...")
    print("Architecture: Memory + Research + Intelligence Agents")
    
    # Set environment variable for Python path
    os.environ["PYTHONPATH"] = backend_dir
    
    # Configure logging to work with our application loggers
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',  # Simple format for our workflow logs
        force=True  # Override any existing configuration
    )
    
    try:
        uvicorn.run(
            "api.autonomous_main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
            log_config=None,  # Don't use uvicorn's default log config
            use_colors=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Autonomous AI System...")
    except Exception as e:
        print(f"❌ Error starting Autonomous AI System: {e}")
        sys.exit(1)