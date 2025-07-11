import asyncio
import json
import os
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import tempfile
import subprocess

# Tool imports
try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None

# E2B support removed - using local execution only

try:
    from playwright.async_api import async_playwright
except ImportError:
    async_playwright = None

import pandas as pd
import requests
from langchain.tools import Tool
from langchain_community.tools import TavilySearchResults


@dataclass
class ToolResult:
    """Standardized tool execution result"""
    success: bool
    result: Any
    error: Optional[str]
    execution_time: float
    tool_name: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "result": str(self.result),
            "error": self.error,
            "execution_time": self.execution_time,
            "tool_name": self.tool_name,
            "metadata": self.metadata,
            "timestamp": datetime.now().isoformat()
        }


class AdvancedTools:
    """2025 standard agent tools with modern integrations"""
    
    def __init__(self, config=None):
        self.config = config
        self.tools = {}
        self.execution_history: List[ToolResult] = []
        
        # API Keys from settings.yaml only
        self.tavily_api_key = self._get_tool_config("tavily", "api_key") or "demo"
        
        # Initialize tools
        self._initialize_tools()
        
        # Tool usage metrics
        self.metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "tool_usage_count": {}
        }
    
    def _get_tool_config(self, tool_name: str, key: str) -> Optional[str]:
        """Get tool configuration from settings"""
        if not self.config:
            return None
        
        # Handle both AssistantConfig objects and raw dicts
        if hasattr(self.config, 'get'):
            return self.config.get(f"tools.{tool_name}.{key}")
        elif hasattr(self.config, 'raw_config'):
            tools_config = self.config.raw_config.get("tools", {})
            tool_config = tools_config.get(tool_name, {})
            return tool_config.get(key)
        else:
            # Raw dict
            tools_config = self.config.get("tools", {})
            tool_config = tools_config.get(tool_name, {})
            return tool_config.get(key)
    
    def _initialize_tools(self):
        """Initialize all available tools"""
        
        # Web Research Tools
        self.tools["deep_search"] = Tool(
            name="deep_search",
            description="Deep web research with citations and multiple sources",
            func=self._deep_web_search
        )
        
        self.tools["quick_search"] = Tool(
            name="quick_search", 
            description="Quick web search for immediate answers",
            func=self._quick_web_search
        )
        
        # Code Execution Tools
        self.tools["python_sandbox"] = Tool(
            name="python_sandbox",
            description="Execute Python code safely in isolated environment",
            func=self._execute_python_code
        )
        
        self.tools["shell_command"] = Tool(
            name="shell_command",
            description="Execute safe shell commands (limited scope)",
            func=self._execute_shell_command
        )
        
        # Browser Automation Tools
        self.tools["web_browser"] = Tool(
            name="web_browser",
            description="Browse and interact with websites using automation",
            func=self._browser_action
        )
        
        self.tools["screenshot"] = Tool(
            name="screenshot",
            description="Take screenshot of a webpage",
            func=self._take_screenshot
        )
        
        # Data Analysis Tools
        self.tools["data_analyzer"] = Tool(
            name="data_analyzer",
            description="Analyze CSV, JSON, or Excel data files",
            func=self._analyze_data
        )
        
        self.tools["chart_generator"] = Tool(
            name="chart_generator", 
            description="Generate charts and visualizations from data",
            func=self._generate_chart
        )
        
        # File Management Tools
        self.tools["file_manager"] = Tool(
            name="file_manager",
            description="Manage files and directories safely",
            func=self._manage_files
        )
        
        self.tools["text_processor"] = Tool(
            name="text_processor",
            description="Process and analyze text documents",
            func=self._process_text
        )
        
        # Communication Tools
        self.tools["email_composer"] = Tool(
            name="email_composer",
            description="Compose professional emails (draft only)",
            func=self._compose_email
        )
        
        # Calendar and Scheduling Tools
        self.tools["calendar_manager"] = Tool(
            name="calendar_manager",
            description="Manage calendar events and scheduling",
            func=self._manage_calendar
        )
        
        # Research and Learning Tools
        self.tools["document_summarizer"] = Tool(
            name="document_summarizer",
            description="Summarize long documents and extract key insights",
            func=self._summarize_document
        )
        
        self.tools["knowledge_extractor"] = Tool(
            name="knowledge_extractor",
            description="Extract structured knowledge from unstructured text",
            func=self._extract_knowledge
        )
    
    async def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool with error handling and metrics"""
        
        start_time = datetime.now()
        
        if tool_name not in self.tools:
            return ToolResult(
                success=False,
                result=None,
                error=f"Tool '{tool_name}' not found",
                execution_time=0.0,
                tool_name=tool_name,
                metadata={}
            )
        
        try:
            tool = self.tools[tool_name]
            
            # Execute tool
            if asyncio.iscoroutinefunction(tool.func):
                result = await tool.func(**kwargs)
            else:
                result = tool.func(**kwargs)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            tool_result = ToolResult(
                success=True,
                result=result,
                error=None,
                execution_time=execution_time,
                tool_name=tool_name,
                metadata=kwargs
            )
            
            # Update metrics
            self._update_metrics(tool_result)
            
            # Store in history
            self.execution_history.append(tool_result)
            
            return tool_result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            tool_result = ToolResult(
                success=False,
                result=None,
                error=str(e),
                execution_time=execution_time,
                tool_name=tool_name,
                metadata=kwargs
            )
            
            # Update metrics
            self._update_metrics(tool_result)
            
            # Store in history
            self.execution_history.append(tool_result)
            
            return tool_result
    
    async def _deep_web_search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Deep web research with multiple sources and citations"""
        
        results = {
            "query": query,
            "sources": [],
            "summary": "",
            "key_insights": [],
            "citations": []
        }
        
        try:
            if TavilyClient and self.tavily_api_key != "demo":
                # Use Tavily for deep research
                client = TavilyClient(api_key=self.tavily_api_key)
                
                search_result = client.search(
                    query=query,
                    search_depth="advanced",
                    max_results=max_results,
                    include_domains=[],
                    exclude_domains=[]
                )
                
                results["sources"] = search_result.get("results", [])
                results["summary"] = search_result.get("answer", "")
                
                # Extract citations
                for i, source in enumerate(results["sources"]):
                    results["citations"].append({
                        "number": i + 1,
                        "title": source.get("title", ""),
                        "url": source.get("url", ""),
                        "snippet": source.get("content", "")[:200] + "..."
                    })
                
                # Generate key insights
                if results["sources"]:
                    insights = []
                    for source in results["sources"][:3]:
                        content = source.get("content", "")
                        if len(content) > 100:
                            insights.append(content[:150] + "...")
                    results["key_insights"] = insights
                
            else:
                # Fallback to basic search
                results = await self._fallback_search(query, max_results)
                
        except Exception as e:
            results["error"] = str(e)
            results = await self._fallback_search(query, max_results)
        
        return results
    
    async def _quick_web_search(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        """Quick web search for immediate answers"""
        
        try:
            if TavilyClient and self.tavily_api_key != "demo":
                client = TavilyClient(api_key=self.tavily_api_key)
                
                search_result = client.search(
                    query=query,
                    search_depth="basic",
                    max_results=max_results
                )
                
                return {
                    "query": query,
                    "answer": search_result.get("answer", ""),
                    "sources": search_result.get("results", [])[:max_results],
                    "search_time": datetime.now().isoformat()
                }
            else:
                return await self._fallback_search(query, max_results)
                
        except Exception as e:
            return {
                "query": query,
                "error": str(e),
                "answer": "Search temporarily unavailable",
                "sources": []
            }
    
    async def _fallback_search(self, query: str, max_results: int) -> Dict[str, Any]:
        """Fallback search when primary tools aren't available"""
        
        # Simple simulation of search results
        return {
            "query": query,
            "answer": f"Search results for: {query}",
            "sources": [
                {
                    "title": f"Result 1 for {query}",
                    "url": "https://example.com/1",
                    "content": f"Mock search result content for {query}. This would contain relevant information."
                },
                {
                    "title": f"Result 2 for {query}",
                    "url": "https://example.com/2", 
                    "content": f"Additional search result content for {query}. More detailed information here."
                }
            ][:max_results],
            "fallback": True
        }
    
    async def _execute_python_code(self, code: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute Python code using local execution with safety restrictions"""
        
        try:
            # Always use local execution (E2B support removed)
            return await self._local_python_execution(code, timeout)
                
        except Exception as e:
            return {
                "stdout": "",
                "stderr": str(e),
                "exit_code": 1,
                "execution_time": 0.0,
                "success": False,
                "error": "Code execution failed"
            }
    
    async def _local_python_execution(self, code: str, timeout: int) -> Dict[str, Any]:
        """Local Python execution with safety restrictions"""
        
        # Safety check - block dangerous operations
        dangerous_imports = [
            "os", "subprocess", "sys", "shutil", "glob", "socket", 
            "urllib", "requests", "http", "ftplib", "smtplib"
        ]
        
        code_lines = code.split('\n')
        for line in code_lines:
            line_stripped = line.strip()
            if line_stripped.startswith('import ') or line_stripped.startswith('from '):
                for dangerous in dangerous_imports:
                    if dangerous in line:
                        return {
                            "stdout": "",
                            "stderr": f"Import '{dangerous}' not allowed in sandbox mode",
                            "exit_code": 1,
                            "execution_time": 0.0,
                            "success": False
                        }
        
        # Execute in temporary file
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Run with timeout
            start_time = datetime.now()
            result = subprocess.run(
                ["python", temp_file],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Clean up
            os.unlink(temp_file)
            
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
                "execution_time": execution_time,
                "success": result.returncode == 0
            }
            
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": "Code execution timed out",
                "exit_code": 124,
                "execution_time": timeout,
                "success": False
            }
        except Exception as e:
            return {
                "stdout": "",
                "stderr": str(e),
                "exit_code": 1,
                "execution_time": 0.0,
                "success": False
            }
    
    async def _execute_shell_command(self, command: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute safe shell commands with restrictions"""
        
        # Whitelist of safe commands
        safe_commands = [
            "ls", "pwd", "date", "whoami", "echo", "cat", "head", "tail",
            "grep", "find", "wc", "sort", "uniq", "diff", "file"
        ]
        
        command_parts = command.split()
        if not command_parts or command_parts[0] not in safe_commands:
            return {
                "stdout": "",
                "stderr": f"Command '{command_parts[0] if command_parts else command}' not allowed",
                "exit_code": 1,
                "success": False
            }
        
        try:
            start_time = datetime.now()
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
                "execution_time": execution_time,
                "success": result.returncode == 0
            }
            
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": "Command timed out",
                "exit_code": 124,
                "success": False
            }
        except Exception as e:
            return {
                "stdout": "",
                "stderr": str(e),
                "exit_code": 1,
                "success": False
            }
    
    async def _browser_action(self, action: str, url: str = None, **kwargs) -> Dict[str, Any]:
        """Browser automation for web interactions"""
        
        if not async_playwright:
            return {
                "success": False,
                "error": "Playwright not available",
                "action": action
            }
        
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                
                result = {"action": action, "success": True}
                
                if action == "navigate" and url:
                    await page.goto(url)
                    result["title"] = await page.title()
                    result["url"] = page.url
                    
                elif action == "get_text":
                    if url:
                        await page.goto(url)
                    selector = kwargs.get("selector", "body")
                    text = await page.text_content(selector)
                    result["text"] = text
                    
                elif action == "click":
                    if url:
                        await page.goto(url)
                    selector = kwargs.get("selector")
                    if selector:
                        await page.click(selector)
                        result["clicked"] = selector
                    
                elif action == "fill_form":
                    if url:
                        await page.goto(url)
                    form_data = kwargs.get("form_data", {})
                    for selector, value in form_data.items():
                        await page.fill(selector, value)
                    result["form_filled"] = True
                
                await browser.close()
                return result
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "action": action
            }
    
    async def _take_screenshot(self, url: str, selector: str = None) -> Dict[str, Any]:
        """Take screenshot of webpage"""
        
        if not async_playwright:
            return {
                "success": False,
                "error": "Playwright not available"
            }
        
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(url)
                
                # Wait for page load
                await page.wait_for_load_state("networkidle")
                
                # Take screenshot
                screenshot_path = f"/tmp/screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                
                if selector:
                    element = page.locator(selector)
                    await element.screenshot(path=screenshot_path)
                else:
                    await page.screenshot(path=screenshot_path)
                
                await browser.close()
                
                return {
                    "success": True,
                    "screenshot_path": screenshot_path,
                    "url": url,
                    "selector": selector
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _analyze_data(self, data_source: str, analysis_type: str = "summary") -> Dict[str, Any]:
        """Analyze data from various sources"""
        
        try:
            # Load data based on source type
            if data_source.endswith('.csv'):
                df = pd.read_csv(data_source)
            elif data_source.endswith('.json'):
                df = pd.read_json(data_source)
            elif data_source.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(data_source)
            else:
                return {
                    "success": False,
                    "error": "Unsupported data format"
                }
            
            analysis = {
                "success": True,
                "data_source": data_source,
                "analysis_type": analysis_type,
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "data_types": df.dtypes.to_dict()
            }
            
            if analysis_type == "summary":
                analysis["summary"] = df.describe().to_dict()
                analysis["missing_values"] = df.isnull().sum().to_dict()
                
            elif analysis_type == "correlation":
                numeric_df = df.select_dtypes(include=['number'])
                if not numeric_df.empty:
                    analysis["correlation"] = numeric_df.corr().to_dict()
                
            elif analysis_type == "distribution":
                analysis["value_counts"] = {}
                for col in df.columns:
                    if df[col].dtype == 'object':
                        analysis["value_counts"][col] = df[col].value_counts().head().to_dict()
            
            return analysis
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "data_source": data_source
            }
    
    async def _generate_chart(self, data_source: str, chart_type: str = "bar", **kwargs) -> Dict[str, Any]:
        """Generate charts and visualizations"""
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Load data
            if data_source.endswith('.csv'):
                df = pd.read_csv(data_source)
            elif data_source.endswith('.json'):
                df = pd.read_json(data_source)
            else:
                return {
                    "success": False,
                    "error": "Unsupported data format for charting"
                }
            
            plt.figure(figsize=(10, 6))
            
            if chart_type == "bar":
                x_col = kwargs.get("x_column", df.columns[0])
                y_col = kwargs.get("y_column", df.columns[1] if len(df.columns) > 1 else df.columns[0])
                plt.bar(df[x_col], df[y_col])
                
            elif chart_type == "line":
                x_col = kwargs.get("x_column", df.columns[0])
                y_col = kwargs.get("y_column", df.columns[1] if len(df.columns) > 1 else df.columns[0])
                plt.plot(df[x_col], df[y_col])
                
            elif chart_type == "scatter":
                x_col = kwargs.get("x_column", df.columns[0])
                y_col = kwargs.get("y_column", df.columns[1] if len(df.columns) > 1 else df.columns[0])
                plt.scatter(df[x_col], df[y_col])
                
            elif chart_type == "histogram":
                col = kwargs.get("column", df.columns[0])
                plt.hist(df[col], bins=kwargs.get("bins", 20))
            
            plt.title(kwargs.get("title", f"{chart_type.title()} Chart"))
            plt.xlabel(kwargs.get("xlabel", "X"))
            plt.ylabel(kwargs.get("ylabel", "Y"))
            
            # Save chart
            chart_path = f"/tmp/chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(chart_path)
            plt.close()
            
            return {
                "success": True,
                "chart_path": chart_path,
                "chart_type": chart_type,
                "data_source": data_source
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _manage_files(self, action: str, path: str = None, **kwargs) -> Dict[str, Any]:
        """Manage files and directories safely"""
        
        # Restrict to safe directories
        safe_dirs = ["/tmp", "/var/tmp", "./data", "./uploads"]
        
        if path and not any(path.startswith(safe_dir) for safe_dir in safe_dirs):
            return {
                "success": False,
                "error": "Access to this directory not allowed"
            }
        
        try:
            if action == "list":
                if os.path.isdir(path):
                    files = os.listdir(path)
                    return {
                        "success": True,
                        "files": files,
                        "path": path
                    }
                
            elif action == "create_dir":
                os.makedirs(path, exist_ok=True)
                return {
                    "success": True,
                    "created": path
                }
                
            elif action == "read":
                if os.path.isfile(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    return {
                        "success": True,
                        "content": content[:10000],  # Limit content size
                        "path": path
                    }
                    
            elif action == "write":
                content = kwargs.get("content", "")
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return {
                    "success": True,
                    "written": path,
                    "size": len(content)
                }
            
            return {
                "success": False,
                "error": f"Unknown action: {action}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _process_text(self, text: str, operation: str = "analyze") -> Dict[str, Any]:
        """Process and analyze text documents"""
        
        try:
            result = {
                "success": True,
                "operation": operation,
                "original_length": len(text)
            }
            
            if operation == "analyze":
                words = text.split()
                sentences = text.split('.')
                
                result.update({
                    "word_count": len(words),
                    "sentence_count": len(sentences),
                    "character_count": len(text),
                    "average_words_per_sentence": len(words) / max(1, len(sentences))
                })
                
            elif operation == "summarize":
                # Simple extractive summarization
                sentences = text.split('.')
                # Take first and last few sentences as summary
                summary_sentences = sentences[:2] + sentences[-2:]
                result["summary"] = '. '.join(summary_sentences)
                
            elif operation == "extract_keywords":
                words = text.lower().split()
                # Simple keyword extraction based on frequency
                word_freq = {}
                for word in words:
                    if len(word) > 4:  # Only longer words
                        word_freq[word] = word_freq.get(word, 0) + 1
                
                keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
                result["keywords"] = [word for word, freq in keywords]
                
            elif operation == "sentiment":
                # Simple sentiment analysis
                positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic"]
                negative_words = ["bad", "terrible", "awful", "horrible", "disappointing", "poor"]
                
                text_lower = text.lower()
                positive_count = sum(1 for word in positive_words if word in text_lower)
                negative_count = sum(1 for word in negative_words if word in text_lower)
                
                if positive_count > negative_count:
                    sentiment = "positive"
                elif negative_count > positive_count:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"
                
                result.update({
                    "sentiment": sentiment,
                    "positive_indicators": positive_count,
                    "negative_indicators": negative_count
                })
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _compose_email(self, to: str, subject: str, content: str, tone: str = "professional") -> Dict[str, Any]:
        """Compose professional emails (draft only)"""
        
        try:
            # Email templates based on tone
            templates = {
                "professional": {
                    "greeting": f"Dear {to.split('@')[0].title()},",
                    "closing": "Best regards,"
                },
                "friendly": {
                    "greeting": f"Hi {to.split('@')[0].title()},",
                    "closing": "Best,"
                },
                "formal": {
                    "greeting": f"Dear Sir/Madam,",
                    "closing": "Sincerely,"
                }
            }
            
            template = templates.get(tone, templates["professional"])
            
            email_draft = f"""To: {to}
Subject: {subject}

{template['greeting']}

{content}

{template['closing']}
[Your Name]
"""
            
            return {
                "success": True,
                "email_draft": email_draft,
                "to": to,
                "subject": subject,
                "tone": tone,
                "note": "This is a draft only. Review before sending."
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _manage_calendar(self, action: str, **kwargs) -> Dict[str, Any]:
        """Manage calendar events and scheduling"""
        
        try:
            if action == "create_event":
                event = {
                    "title": kwargs.get("title", "New Event"),
                    "date": kwargs.get("date", datetime.now().isoformat()),
                    "duration": kwargs.get("duration", "1 hour"),
                    "description": kwargs.get("description", ""),
                    "location": kwargs.get("location", ""),
                    "attendees": kwargs.get("attendees", [])
                }
                
                return {
                    "success": True,
                    "action": "create_event",
                    "event": event,
                    "note": "Event created in local calendar (demo mode)"
                }
                
            elif action == "find_free_time":
                date = kwargs.get("date", datetime.now().date())
                duration = kwargs.get("duration", 60)  # minutes
                
                # Mock free time slots
                free_slots = [
                    "09:00 - 10:00",
                    "11:00 - 12:00", 
                    "14:00 - 15:00",
                    "16:00 - 17:00"
                ]
                
                return {
                    "success": True,
                    "action": "find_free_time",
                    "date": str(date),
                    "duration_minutes": duration,
                    "free_slots": free_slots
                }
                
            elif action == "list_events":
                date = kwargs.get("date", datetime.now().date())
                
                # Mock events
                events = [
                    {
                        "title": "Team Meeting",
                        "time": "10:00 - 11:00",
                        "location": "Conference Room A"
                    },
                    {
                        "title": "Project Review",
                        "time": "15:00 - 16:00", 
                        "location": "Online"
                    }
                ]
                
                return {
                    "success": True,
                    "action": "list_events",
                    "date": str(date),
                    "events": events
                }
            
            return {
                "success": False,
                "error": f"Unknown calendar action: {action}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _summarize_document(self, document: str, max_length: int = 500) -> Dict[str, Any]:
        """Summarize long documents and extract key insights"""
        
        try:
            # Split into paragraphs
            paragraphs = [p.strip() for p in document.split('\n\n') if p.strip()]
            
            if not paragraphs:
                return {
                    "success": False,
                    "error": "No content to summarize"
                }
            
            # Simple extractive summarization
            # Take first paragraph, middle paragraph, and last paragraph
            key_paragraphs = []
            
            if len(paragraphs) >= 1:
                key_paragraphs.append(paragraphs[0])  # Introduction
            
            if len(paragraphs) >= 3:
                middle_idx = len(paragraphs) // 2
                key_paragraphs.append(paragraphs[middle_idx])  # Middle content
            
            if len(paragraphs) >= 2:
                key_paragraphs.append(paragraphs[-1])  # Conclusion
            
            summary = ' '.join(key_paragraphs)
            
            # Truncate if too long
            if len(summary) > max_length:
                summary = summary[:max_length] + "..."
            
            # Extract key insights (simple approach)
            insights = []
            for paragraph in paragraphs:
                if any(word in paragraph.lower() for word in ["important", "key", "significant", "critical", "essential"]):
                    insights.append(paragraph[:200] + "..." if len(paragraph) > 200 else paragraph)
                    if len(insights) >= 3:
                        break
            
            return {
                "success": True,
                "original_length": len(document),
                "summary_length": len(summary),
                "summary": summary,
                "key_insights": insights,
                "paragraph_count": len(paragraphs)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _extract_knowledge(self, text: str, format_type: str = "structured") -> Dict[str, Any]:
        """Extract structured knowledge from unstructured text"""
        
        try:
            result = {
                "success": True,
                "format_type": format_type,
                "extracted_knowledge": {}
            }
            
            if format_type == "structured":
                # Extract entities, dates, numbers, etc.
                import re
                
                # Extract dates
                date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b'
                dates = re.findall(date_pattern, text)
                
                # Extract numbers
                number_pattern = r'\b\d+\.?\d*\b'
                numbers = re.findall(number_pattern, text)
                
                # Extract emails
                email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                emails = re.findall(email_pattern, text)
                
                # Extract phone numbers
                phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
                phones = re.findall(phone_pattern, text)
                
                # Extract URLs
                url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
                urls = re.findall(url_pattern, text)
                
                result["extracted_knowledge"] = {
                    "dates": list(set(dates)),
                    "numbers": list(set(numbers))[:10],  # Limit to first 10
                    "emails": list(set(emails)),
                    "phone_numbers": list(set(phones)),
                    "urls": list(set(urls))
                }
                
            elif format_type == "concepts":
                # Extract key concepts and terms
                words = text.split()
                
                # Find capitalized words (potential proper nouns)
                proper_nouns = [word.strip('.,!?') for word in words if word[0].isupper() and len(word) > 2]
                
                # Find repeated important terms
                word_freq = {}
                for word in words:
                    clean_word = word.lower().strip('.,!?')
                    if len(clean_word) > 4:
                        word_freq[clean_word] = word_freq.get(clean_word, 0) + 1
                
                key_terms = [word for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:15]]
                
                result["extracted_knowledge"] = {
                    "proper_nouns": list(set(proper_nouns))[:20],
                    "key_terms": key_terms,
                    "total_unique_words": len(set(words))
                }
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _update_metrics(self, tool_result: ToolResult):
        """Update tool usage metrics"""
        
        self.metrics["total_executions"] += 1
        
        if tool_result.success:
            self.metrics["successful_executions"] += 1
        else:
            self.metrics["failed_executions"] += 1
        
        # Update average execution time
        total = self.metrics["total_executions"]
        current_avg = self.metrics["average_execution_time"]
        self.metrics["average_execution_time"] = (
            (current_avg * (total - 1) + tool_result.execution_time) / total
        )
        
        # Update tool usage count
        tool_name = tool_result.tool_name
        if tool_name not in self.metrics["tool_usage_count"]:
            self.metrics["tool_usage_count"][tool_name] = 0
        self.metrics["tool_usage_count"][tool_name] += 1
    
    def get_available_tools(self) -> List[Dict[str, str]]:
        """Get list of available tools with descriptions"""
        
        return [
            {
                "name": name,
                "description": tool.description
            }
            for name, tool in self.tools.items()
        ]
    
    def get_tool_metrics(self) -> Dict[str, Any]:
        """Get tool usage metrics"""
        
        return {
            **self.metrics,
            "success_rate": (self.metrics["successful_executions"] / max(1, self.metrics["total_executions"])) * 100,
            "available_tools": len(self.tools),
            "recent_executions": len([r for r in self.execution_history if 
                                    (datetime.now() - datetime.fromisoformat(r.metadata.get("timestamp", "2000-01-01"))).total_seconds() < 3600])
        }
    
    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent tool execution history"""
        
        return [result.to_dict() for result in self.execution_history[-limit:]]
    
    def cleanup_old_history(self, max_age_hours: int = 24):
        """Clean up old execution history"""
        
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        self.execution_history = [
            result for result in self.execution_history
            if datetime.fromisoformat(result.metadata.get("timestamp", "2000-01-01")) > cutoff_time
        ]