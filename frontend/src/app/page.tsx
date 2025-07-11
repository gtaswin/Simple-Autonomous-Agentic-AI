'use client'

import { useState, useEffect, useRef } from 'react'
import { Send, Bot, User, Settings, Menu, X, Zap, Brain, MessageSquare, Sparkles, MoreVertical, Activity, Database, Users, ChevronDown, ChevronUp, Lightbulb, Search, RefreshCw, Eye, TrendingUp, Clock, Globe, Cpu, Network } from 'lucide-react'

interface Message {
  id: string
  content: string
  sender: 'user' | 'assistant'
  timestamp: Date
  typing?: boolean
  streaming?: boolean
  fromHistory?: boolean
}

interface AgentState {
  mode: string
  current_focus: string
  cognitive_load: number
  recent_thoughts: any[]
  active_goals: any[]
  thoughts_today: number
  decisions_today: number
}

interface AgentStatus {
  status: string
  agent_count: number
  agents: Record<string, any>
  collaboration_active: boolean
  timestamp: number
}

interface ThinkingStream {
  thoughts: any[]
  status: string
  thought_count: number
  active_reasoning: boolean
}

interface ExpertTeamStatus {
  expert_team: any
  timestamp: string
}

interface AutonomousInsights {
  user_id: string
  autonomous_insights: any[]
  count: number
  timestamp: string
}

interface MemoryInsights {
  insights: any
  timestamp: string
}

interface ProviderStatus {
  status: string
  hierarchy: string[]
  providers: Record<string, any>
  model_routing?: Record<string, any>
}

interface SystemMetrics {
  metrics: {
    reasoning: { total_cycles: number; active_cycles: number }
    collaboration: { total_collaborations: number; active_collaborations: number }
    decisions: { autonomous_decisions: number; cost_today: number }
    memory: { total_memories: number; stored_today: number }
  }
}

interface AgentThoughts {
  thoughts: Array<{
    id: string
    content: string
    agent_name: string
    timestamp: string
    thought_type: string
    importance: number
  }>
}

/**
 * OPTIMIZED API POLLING STRATEGY (2025-01-05)
 * 
 * To reduce frequent API calls and server load:
 * 1. Extended polling intervals (1-5 minutes instead of 5-30 seconds)
 * 2. Smart caching with timestamps - prevents redundant calls within time windows
 * 3. On-demand loading - fetch data only when users expand sections
 * 4. Force refresh option - manual refresh button bypasses cache
 * 5. WebSocket for real-time data - reduces need for polling
 * 6. Conditional loading - only refresh visible sections during periodic updates
 */
export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'connecting'>('disconnected')
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [agentState, setAgentState] = useState<AgentState | null>(null)
  const [agentStatus, setAgentStatus] = useState<AgentStatus | null>(null)
  const [thinkingStream, setThinkingStream] = useState<ThinkingStream | null>(null)
  const [providerStatus, setProviderStatus] = useState<ProviderStatus | null>(null)
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics | null>(null)
  const [memoryInsights, setMemoryInsights] = useState<MemoryInsights | null>(null)
  const [agentThoughts, setAgentThoughts] = useState<AgentThoughts | null>(null)
  const [expertTeamStatus, setExpertTeamStatus] = useState<ExpertTeamStatus | null>(null)
  const [autonomousInsights, setAutonomousInsights] = useState<AutonomousInsights | null>(null)
  const [expandedSections, setExpandedSections] = useState({
    status: true,
    thinking: true,
    insights: false,
    memory: false,
    agents: false,
    providers: false
  })
  const [streamingMessage, setStreamingMessage] = useState<string>('')
  const [wsConnections, setWsConnections] = useState<{
    thinking?: WebSocket
    agent?: WebSocket
  }>({})
  const [chatHistory, setChatHistory] = useState<any[]>([])
  const [chatStats, setChatStats] = useState<any>(null)
  const [historyLoaded, setHistoryLoaded] = useState(false)
  const [lastUpdated, setLastUpdated] = useState<Record<string, number>>({})
  const [isRefreshing, setIsRefreshing] = useState(false)
  
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const fetchAgentState = async (force = false) => {
    const key = 'agentState'
    const now = Date.now()
    const lastFetch = lastUpdated[key] || 0
    
    // Skip if fetched within last 30 seconds unless forced
    if (!force && (now - lastFetch) < 30000) {
      return
    }
    
    try {
      const response = await fetch('http://localhost:8000/agent?include=state')
      if (response.ok) {
        const data = await response.json()
        setAgentState(data.agent_state)
        setLastUpdated(prev => ({ ...prev, [key]: now }))
      }
    } catch (error) {
      console.error('Failed to fetch agent state:', error)
    }
  }

  const fetchAgentStatus = async (force = false) => {
    const key = 'agentStatus'
    const now = Date.now()
    const lastFetch = lastUpdated[key] || 0
    
    // Skip if fetched within last 60 seconds unless forced
    if (!force && (now - lastFetch) < 60000) {
      return
    }
    
    try {
      const response = await fetch('http://localhost:8000/agents/status')
      if (response.ok) {
        const data = await response.json()
        setAgentStatus(data)
        setLastUpdated(prev => ({ ...prev, [key]: now }))
      }
    } catch (error) {
      console.error('Failed to fetch agent status:', error)
    }
  }

  const fetchThinkingStream = async () => {
    try {
      const response = await fetch('http://localhost:8000/thinking/status')
      if (response.ok) {
        const data = await response.json()
        setThinkingStream(data)
      }
    } catch (error) {
      console.error('Failed to fetch thinking status:', error)
    }
  }

  const fetchProviderStatus = async (force = false) => {
    const key = 'providerStatus'
    const now = Date.now()
    const lastFetch = lastUpdated[key] || 0
    
    // Skip if fetched within last 5 minutes unless forced (providers rarely change)
    if (!force && (now - lastFetch) < 300000) {
      return
    }
    
    try {
      const response = await fetch('http://localhost:8000/providers/status')
      if (response.ok) {
        const data = await response.json()
        setProviderStatus(data)
        setLastUpdated(prev => ({ ...prev, [key]: now }))
      }
    } catch (error) {
      console.error('Failed to fetch provider status:', error)
    }
  }

  const checkWebSocketHealth = async () => {
    try {
      const response = await fetch('http://localhost:8000/websocket/health')
      if (response.ok) {
        const data = await response.json()
        console.log('🩺 WebSocket Health:', data)
        return data
      }
    } catch (error) {
      console.error('Failed to check WebSocket health:', error)
    }
  }

  const fetchSystemMetrics = async () => {
    try {
      const response = await fetch('http://localhost:8000/metrics/system')
      if (response.ok) {
        const data = await response.json()
        setSystemMetrics(data)
      }
    } catch (error) {
      console.error('Failed to fetch system metrics:', error)
    }
  }

  const fetchMemoryInsights = async () => {
    try {
      const response = await fetch('http://localhost:8000/memory?type=insights')
      if (response.ok) {
        const data = await response.json()
        setMemoryInsights(data)
        
        // If no memories yet, also check system metrics for total count
        if (data.insights?.total_memories === 0) {
          console.log('📊 No memories found, checking system metrics...')
        }
      }
    } catch (error) {
      console.error('Failed to fetch memory insights:', error)
    }
  }

  const fetchAgentThoughts = async () => {
    try {
      const response = await fetch('http://localhost:8000/agent?include=thoughts&limit=10')
      if (response.ok) {
        const data = await response.json()
        // The consolidated endpoint returns {thoughts: [...]} instead of {thoughts: [...]} directly
        setAgentThoughts(data.thoughts || data)
      }
    } catch (error) {
      console.error('Failed to fetch agent thoughts:', error)
    }
  }

  const fetchExpertTeamStatus = async () => {
    try {
      const response = await fetch('http://localhost:8000/expert-team/status')
      if (response.ok) {
        const data = await response.json()
        setExpertTeamStatus(data)
      }
    } catch (error) {
      console.error('Failed to fetch expert team status:', error)
    }
  }

  const fetchAutonomousInsights = async () => {
    try {
      const response = await fetch('http://localhost:8000/autonomous/insights')
      if (response.ok) {
        const data = await response.json()
        setAutonomousInsights(data)
      }
    } catch (error) {
      console.error('Failed to fetch autonomous insights:', error)
    }
  }

  const loadChatHistory = async () => {
    if (historyLoaded) return // Don't reload if already loaded
    
    try {
      const response = await fetch('http://localhost:8000/chat/history?user_id=admin&include_stats=true')
      if (response.ok) {
        const data = await response.json()
        const historyMessages = data.messages || []
        setChatStats(data.conversation_stats || {})
        
        // Convert history messages to the same format as current messages
        const convertedHistory = historyMessages.map((msg: any, index: number) => ({
          id: `history-${index}`,
          content: msg.message,
          sender: msg.sender,
          timestamp: new Date(msg.timestamp),
          fromHistory: true
        }))
        
        // Replace current messages with history + any new messages
        setMessages(prev => {
          const newMessages = prev.filter(m => !m.fromHistory)
          return [...convertedHistory, ...newMessages]
        })
        
        setHistoryLoaded(true)
        console.log('🔄 Chat history auto-loaded:', historyMessages.length, 'messages')
      }
    } catch (error) {
      console.error('Failed to fetch chat history:', error)
    }
  }

  const resetChatConversation = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/chat/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: 'admin' })
      })
      
      if (response.ok) {
        console.log('Chat conversation reset successfully')
        setChatHistory([])
        setChatStats(null)
        setMessages([]) // Clear all messages
        setHistoryLoaded(false) // Allow history to be reloaded
      }
    } catch (error) {
      console.error('Failed to reset chat conversation:', error)
    }
  }

  const toggleSection = (section: keyof typeof expandedSections) => {
    const isExpanding = !expandedSections[section]
    setExpandedSections(prev => ({ ...prev, [section]: !prev[section] }))
    
    // Refresh data when expanding sections (on-demand loading with force=true)
    if (isExpanding) {
      switch (section) {
        case 'status':
          fetchAgentState(true) // Force refresh when user opens section
          break
        case 'agents':
          fetchAgentStatus(true)
          fetchAgentThoughts()
          break
        case 'memory':
          fetchMemoryInsights()
          break
        case 'insights':
          fetchAutonomousInsights()
          break
        case 'providers':
          fetchProviderStatus(true)
          break
        case 'thinking':
          fetchThinkingStream()
          break
      }
    }
  }

  const setupWebSocketConnections = () => {
    let reconnectAttempts = 0
    const maxReconnectAttempts = 5
    
    // Thinking WebSocket with improved retry logic
    const connectThinkingWebSocket = () => {
      const thinkingWs = new WebSocket('ws://localhost:8000/thinking/stream?user_id=admin')
      
      thinkingWs.onopen = () => {
        console.log('🧠 Connected to thinking stream')
        setConnectionStatus('connected')
        reconnectAttempts = 0 // Reset on successful connection
      }
      
      thinkingWs.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          
          if (data.type === 'initial_thoughts') {
            setThinkingStream({
              thoughts: data.thoughts || [],
              status: data.status || 'active',
              thought_count: data.thoughts?.length || 0,
              active_reasoning: data.status === 'continuous_thinking_active'
            })
          } else if (data.type === 'new_thought') {
            setThinkingStream(prev => {
              if (!prev) return null
              return {
                ...prev,
                thoughts: prev.thoughts ? [data.data, ...prev.thoughts.slice(0, 9)] : [data.data],
                thought_count: (prev.thought_count || 0) + 1
              }
            })
          } else if (data.type === 'status_update') {
            setThinkingStream(prev => {
              if (!prev) return null
              return {
                ...prev,
                active_reasoning: data.data?.active_reasoning || false,
                status: data.data?.active_reasoning ? 'active' : 'inactive'
              }
            })
          } else if (data.type === 'heartbeat') {
            // Handle heartbeat - just acknowledge receipt
            console.log('💓 Received heartbeat from server')
          }
        } catch (error) {
          console.error('Error parsing thinking WebSocket message:', error)
        }
      }
      
      thinkingWs.onclose = (event) => {
        console.log(`🔌 Thinking stream disconnected (code: ${event.code}, reason: ${event.reason})`)
        setConnectionStatus('disconnected')
        
        // Exponential backoff reconnection
        if (reconnectAttempts < maxReconnectAttempts) {
          const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000) // Max 30 seconds
          console.log(`🔄 Reconnecting thinking stream in ${delay}ms (attempt ${reconnectAttempts + 1})`)
          reconnectAttempts++
          setTimeout(connectThinkingWebSocket, delay)
        } else {
          console.log('❌ Max reconnection attempts reached for thinking stream')
        }
      }
      
      thinkingWs.onerror = (error) => {
        console.error('🔌 Thinking WebSocket error:', error)
        setConnectionStatus('disconnected')
      }
      
      return thinkingWs
    }
    
    const thinkingWs = connectThinkingWebSocket()
    
    // Agent WebSocket with improved retry logic
    const connectAgentWebSocket = () => {
      const agentWs = new WebSocket('ws://localhost:8000/agent-stream?user_id=admin')
      
      agentWs.onopen = () => {
        console.log('🤖 Connected to agent stream')
      }
      
      agentWs.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          
          if (data.type === 'connection') {
            console.log('Agent stream connected:', data.data)
          } else if (data.type === 'agent_state') {
            setAgentState(data.data)
          } else if (data.type === 'thoughts') {
            setAgentThoughts({ thoughts: data.data })
          } else if (data.type === 'new_insight') {
            // Update autonomous insights with new insight
            setAutonomousInsights(prev => {
              if (!prev) return null
              return {
                ...prev,
                autonomous_insights: prev.autonomous_insights ? [data.data, ...prev.autonomous_insights.slice(0, 9)] : [data.data],
                count: (prev.count || 0) + 1
              }
            })
          } else if (data.type === 'heartbeat') {
            console.log('💓 Agent stream heartbeat received')
          }
        } catch (error) {
          console.error('Error parsing agent WebSocket message:', error)
        }
      }
      
      agentWs.onclose = (event) => {
        console.log(`🔌 Agent stream disconnected (code: ${event.code})`)
        // Reconnect with exponential backoff
        const delay = Math.min(7000 * Math.pow(1.5, reconnectAttempts), 30000)
        setTimeout(connectAgentWebSocket, delay)
      }
      
      agentWs.onerror = (error) => {
        console.error('🔌 Agent WebSocket error:', error)
      }
      
      return agentWs
    }
    
    const agentWs = connectAgentWebSocket()

    setWsConnections({ thinking: thinkingWs, agent: agentWs })
    
    return { thinkingWs, agentWs }
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  useEffect(() => {
    // Load chat history first, then add welcome message if no history exists
    const initializeChat = async () => {
      await loadChatHistory()
      
      // Only add welcome message if no history was loaded
      setMessages(prev => {
        if (prev.length === 0) {
          return [{
            id: '1',
            content: "Hello! I'm your autonomous AI agent with multi-agent collaboration. I think continuously in the background, learn from our conversations, and make decisions independently. I'm always analyzing patterns and preparing insights for you. What would you like to explore together?",
            sender: 'assistant',
            timestamp: new Date()
          }]
        }
        return prev
      })
    }
    
    initializeChat()

    // Initial data fetch - only load essential data and expanded sections
    if (expandedSections.status) fetchAgentState()
    if (expandedSections.thinking) fetchThinkingStream()
    
    // Always load critical system data
    fetchSystemMetrics()
    fetchExpertTeamStatus()
    
    // Load other data only if sections are expanded
    if (expandedSections.agents) {
      fetchAgentStatus()
      fetchAgentThoughts()
    }
    if (expandedSections.memory) fetchMemoryInsights()
    if (expandedSections.insights) fetchAutonomousInsights()
    if (expandedSections.providers) fetchProviderStatus()

    // Set up WebSocket connections
    const { thinkingWs, agentWs } = setupWebSocketConnections()

    // Set up periodic updates for non-real-time data (much less frequent)
    const stateInterval = setInterval(fetchAgentState, 60000)      // 1 minute
    const statusInterval = setInterval(fetchAgentStatus, 120000)   // 2 minutes  
    const providerInterval = setInterval(fetchProviderStatus, 300000) // 5 minutes (rarely changes)
    const metricsInterval = setInterval(fetchSystemMetrics, 90000)  // 1.5 minutes
    const memoryInterval = setInterval(fetchMemoryInsights, 180000) // 3 minutes
    const thoughtsInterval = setInterval(fetchAgentThoughts, 45000) // 45 seconds
    const expertInterval = setInterval(fetchExpertTeamStatus, 120000) // 2 minutes
    const insightsInterval = setInterval(fetchAutonomousInsights, 240000) // 4 minutes
    const healthInterval = setInterval(checkWebSocketHealth, 120000) // 2 minutes - WebSocket health

    return () => {
      clearInterval(stateInterval)
      clearInterval(statusInterval)
      clearInterval(providerInterval)
      clearInterval(metricsInterval)
      clearInterval(memoryInterval)
      clearInterval(thoughtsInterval)
      clearInterval(expertInterval)
      clearInterval(insightsInterval)
      clearInterval(healthInterval)
      thinkingWs.close()
      agentWs.close()
    }
  }, [])

  const sendMessage = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      content: input,
      sender: 'user',
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsLoading(true)

    // Add streaming indicator
    const streamingMessage: Message = {
      id: 'streaming',
      content: '',
      sender: 'assistant',
      timestamp: new Date(),
      streaming: true
    }
    setMessages(prev => [...prev, streamingMessage])

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: input,
          user_id: 'admin',
          context: {}
        }),
      })

      if (!response.ok) {
        throw new Error('Network response was not ok')
      }

      const data = await response.json()
      
      // Remove streaming indicator
      setMessages(prev => prev.filter(msg => msg.id !== 'streaming'))

      // Simulate streaming effect
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: '',
        sender: 'assistant',
        timestamp: new Date()
      }

      setMessages(prev => [...prev, assistantMessage])

      // Stream the response word by word
      const words = data.response.split(' ')
      let currentContent = ''
      
      for (let i = 0; i < words.length; i++) {
        currentContent += (i > 0 ? ' ' : '') + words[i]
        
        setMessages(prev => 
          prev.map(msg => 
            msg.id === assistantMessage.id 
              ? { ...msg, content: currentContent }
              : msg
          )
        )
        
        // Add delay for streaming effect
        await new Promise(resolve => setTimeout(resolve, 50))
      }
      
      // Refresh agent state after message
      fetchAgentState()
      fetchAgentThoughts()
    } catch (error) {
      console.error('Error sending message:', error)
      
      // Remove streaming indicator
      setMessages(prev => prev.filter(msg => msg.id !== 'streaming'))
      
      // Add error message
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: "I apologize, but I'm having trouble connecting right now. Please try again in a moment.",
        sender: 'assistant',
        timestamp: new Date()
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }

  const clearChat = async () => {
    // Reset backend chat storage first
    await resetChatConversation()
    
    setMessages([
      {
        id: '1',
        content: "Hello! I'm your autonomous AI agent with multi-agent collaboration. I think continuously in the background, learn from our conversations, and make decisions independently. I'm always analyzing patterns and preparing insights for you. What would you like to explore together?",
        sender: 'assistant',
        timestamp: new Date()
      }
    ])
    
    setHistoryLoaded(false) // Reset history state
    
    // Refresh all data
    fetchAgentState()
    fetchAgentStatus()
    fetchThinkingStream()
    fetchMemoryInsights()
    fetchAgentThoughts()
    fetchAutonomousInsights()
    fetchSystemMetrics()
    fetchProviderStatus()
    fetchExpertTeamStatus()
  }

  const refreshAllData = async () => {
    setIsRefreshing(true)
    
    try {
      // Force refresh all data when manually triggered (ignore timestamps)
      const promises = []
      
      if (expandedSections.status) promises.push(fetchAgentState(true))
      if (expandedSections.agents) {
        promises.push(fetchAgentStatus(true))
        promises.push(fetchAgentThoughts())
      }
      if (expandedSections.thinking) promises.push(fetchThinkingStream())
      if (expandedSections.memory) promises.push(fetchMemoryInsights())
      if (expandedSections.insights) promises.push(fetchAutonomousInsights())
      if (expandedSections.providers) promises.push(fetchProviderStatus(true))
      
      // Always fetch critical system data
      promises.push(fetchSystemMetrics())
      promises.push(fetchExpertTeamStatus())
      
      // Wait for all requests to complete
      await Promise.allSettled(promises)
      
      console.log('🔄 Manual refresh completed - all data updated')
    } finally {
      setIsRefreshing(false)
    }
  }

  return (
    <div className="flex h-screen bg-gray-950 text-white overflow-hidden">
      {/* Enhanced Left Panel */}
      <div className={`${sidebarOpen ? 'translate-x-0' : '-translate-x-full'} lg:translate-x-0 fixed lg:static inset-y-0 left-0 z-50 w-96 bg-gray-900 border-r border-gray-800 transition-transform duration-300 ease-in-out flex flex-col h-screen`}>
        {/* Mobile Close Button */}
        <div className="lg:hidden flex justify-end p-4">
          <button 
            onClick={() => setSidebarOpen(false)}
            className="p-2 hover:bg-gray-800 rounded-lg"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
        
        {/* Scrollable Content */}
        <div className="flex-1 overflow-y-auto sidebar-scroll">
          {/* System Status */}
          <div className="p-4">
          <div className="bg-gray-800 rounded-lg">
            <button 
              onClick={() => toggleSection('status')}
              className="w-full flex items-center justify-between p-4 hover:bg-gray-700/50 rounded-lg transition-colors"
            >
              <div className="flex items-center gap-2">
                <Activity className="w-4 h-4 text-blue-400" />
                <span className="text-sm font-medium">System Status</span>
              </div>
              {expandedSections.status ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>
            
            {expandedSections.status && (
              <div className="px-4 pb-4 space-y-3">
                <div className="space-y-2 mb-3">
                  <div className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full animate-pulse ${connectionStatus === 'connected' ? 'bg-green-500' : 'bg-red-500'}`} />
                    <span className="text-xs text-gray-400">WebSocket: {connectionStatus}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full ${thinkingStream?.active_reasoning ? 'bg-purple-500 animate-pulse' : 'bg-gray-500'}`} />
                    <span className="text-xs text-gray-400">Agent Thinking: {thinkingStream?.active_reasoning ? 'active' : 'idle'}</span>
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="bg-gray-700/50 rounded p-2">
                    <div className="text-gray-400">Messages</div>
                    <div className="text-white font-medium">{messages.length - 1}</div>
                  </div>
                  <div className="bg-gray-700/50 rounded p-2">
                    <div className="text-gray-400">Thoughts</div>
                    <div className="text-blue-400 font-medium">{agentState?.thoughts_today || 0}</div>
                  </div>
                  <div className="bg-gray-700/50 rounded p-2">
                    <div className="text-gray-400">Load</div>
                    <div className="text-purple-400 font-medium">{agentState ? Math.round(agentState.cognitive_load * 100) : 0}%</div>
                  </div>
                  <div className="bg-gray-700/50 rounded p-2">
                    <div className="text-gray-400">Agents</div>
                    <div className="text-green-400 font-medium">{agentStatus ? Object.keys(agentStatus.agents || {}).filter(name => name !== 'system_overview').length : 0}</div>
                  </div>
                </div>
                
                <div className="text-xs">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Mode</span>
                    <span className="text-yellow-400">{agentState?.mode || 'loading'}</span>
                  </div>
                  <div className="flex justify-between mt-1">
                    <span className="text-gray-400">Focus</span>
                    <span className="text-indigo-400 truncate max-w-32">{agentState?.current_focus || 'waiting'}</span>
                  </div>
                </div>
                
                {systemMetrics && (
                  <div className="text-xs space-y-1 pt-2 border-t border-gray-700">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Decisions Today</span>
                      <span className="text-green-400">{systemMetrics.metrics.decisions.autonomous_decisions}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Collaborations</span>
                      <span className="text-blue-400">{systemMetrics.metrics.collaboration.total_collaborations}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Memories</span>
                      <span className="text-purple-400">{systemMetrics.metrics.memory.total_memories}</span>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
        
        {/* Live Thinking */}
        <div className="p-4">
          <div className="bg-gray-800 rounded-lg">
            <button 
              onClick={() => toggleSection('thinking')}
              className="w-full flex items-center justify-between p-4 hover:bg-gray-700/50 rounded-lg transition-colors"
            >
              <div className="flex items-center gap-2">
                <Brain className="w-4 h-4 text-purple-400" />
                <span className="text-sm font-medium">Live Thinking</span>
                {thinkingStream?.active_reasoning && (
                  <div className="w-2 h-2 bg-purple-400 rounded-full animate-pulse" />
                )}
              </div>
              {expandedSections.thinking ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>
            
            {expandedSections.thinking && (
              <div className="px-4 pb-4 space-y-2">
                {thinkingStream?.active_reasoning && (
                  <div className="p-3 rounded bg-purple-500/10 border border-purple-500/20">
                    <div className="flex items-center gap-2 mb-1">
                      <div className="w-2 h-2 bg-purple-400 rounded-full animate-pulse" />
                      <span className="text-xs text-purple-400">{thinkingStream.status}</span>
                    </div>
                    <p className="text-xs text-gray-300">
                      {agentState?.current_focus || 'Processing...'}
                    </p>
                    <span className="text-xs text-gray-500">
                      {thinkingStream.thought_count} thoughts
                    </span>
                  </div>
                )}
                
                {thinkingStream?.thoughts && thinkingStream.thoughts.length > 0 && (
                  <div className="space-y-2 max-h-48 overflow-y-auto sidebar-scroll">
                    {thinkingStream.thoughts.slice(0, 5).map((thought, index) => (
                      <div key={index} className="p-2 rounded bg-gray-700/30 text-xs">
                        <div className="flex items-center gap-2 mb-1">
                          <div className="w-1.5 h-1.5 bg-blue-400 rounded-full" />
                          <span className="text-blue-400">{thought.type || 'Thinking'}</span>
                          <span className="text-gray-500 ml-auto">
                            {new Date(thought.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                          </span>
                        </div>
                        <p className="text-gray-300">{thought.content}</p>
                      </div>
                    ))}
                  </div>
                )}

                {(!thinkingStream?.thoughts || thinkingStream.thoughts.length === 0) && (
                  <div className="p-3 rounded bg-gray-700/30 text-center">
                    <p className="text-xs text-gray-400">No recent thoughts available</p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
        
        {/* Agent Insights */}
        <div className="p-4">
          <div className="bg-gray-800 rounded-lg">
            <button 
              onClick={() => toggleSection('insights')}
              className="w-full flex items-center justify-between p-4 hover:bg-gray-700/50 rounded-lg transition-colors"
            >
              <div className="flex items-center gap-2">
                <Lightbulb className="w-4 h-4 text-yellow-400" />
                <span className="text-sm font-medium">Insights</span>
                {autonomousInsights && autonomousInsights.count > 0 && (
                  <span className="bg-yellow-500/20 text-yellow-400 text-xs px-2 py-0.5 rounded">
                    {autonomousInsights.count}
                  </span>
                )}
              </div>
              {expandedSections.insights ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>
            
            {expandedSections.insights && (
              <div className="px-4 pb-4">
                {autonomousInsights && autonomousInsights.autonomous_insights.length > 0 ? (
                  <div className="space-y-2 max-h-32 overflow-y-auto sidebar-scroll">
                    {autonomousInsights.autonomous_insights.slice(0, 3).map((insight, index) => (
                      <div key={index} className="p-2 rounded bg-yellow-500/10 border border-yellow-500/20 text-xs">
                        <div className="flex items-center gap-2 mb-1">
                          <Sparkles className="w-3 h-3 text-yellow-400" />
                          <span className="text-yellow-400">Insight</span>
                        </div>
                        <p className="text-gray-300">{insight.content || insight.description || 'New insight discovered'}</p>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="p-3 rounded bg-gray-700/30 text-center">
                    <Lightbulb className="w-4 h-4 text-gray-500 mx-auto mb-1" />
                    <p className="text-xs text-gray-400">No insights yet</p>
                    <p className="text-xs text-gray-500 mt-1">Start chatting to generate AI insights</p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
        
        {/* Memory Analytics */}
        <div className="p-4">
          <div className="bg-gray-800 rounded-lg">
            <button 
              onClick={() => toggleSection('memory')}
              className="w-full flex items-center justify-between p-4 hover:bg-gray-700/50 rounded-lg transition-colors"
            >
              <div className="flex items-center gap-2">
                <Database className="w-4 h-4 text-green-400" />
                <span className="text-sm font-medium">Memory</span>
              </div>
              {expandedSections.memory ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>
            
            {expandedSections.memory && (
              <div className="px-4 pb-4">
                {memoryInsights ? (
                  <div className="space-y-2 text-xs">
                    <div className="grid grid-cols-2 gap-2">
                      <div className="bg-gray-700/50 rounded p-2">
                        <div className="text-gray-400">Total</div>
                        <div className="text-green-400 font-medium">{memoryInsights.insights.total_memories || 0}</div>
                      </div>
                      <div className="bg-gray-700/50 rounded p-2">
                        <div className="text-gray-400">This Week</div>
                        <div className="text-blue-400 font-medium">{memoryInsights.insights.recent_activity?.memories_this_week || 0}</div>
                      </div>
                    </div>
                    
                    {memoryInsights.insights.top_concepts && memoryInsights.insights.top_concepts.length > 0 && (
                      <div className="pt-2 border-t border-gray-700">
                        <div className="text-gray-400 mb-1">Top Concepts</div>
                        {memoryInsights.insights.top_concepts.slice(0, 3).map((concept: any, index: number) => (
                          <div key={index} className="flex justify-between">
                            <span className="text-gray-300 truncate">{concept.concept}</span>
                            <span className="text-green-400">{concept.count}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="p-3 rounded bg-gray-700/30 text-center">
                    <Database className="w-4 h-4 text-gray-500 mx-auto mb-1" />
                    <p className="text-xs text-gray-400">No memory data yet</p>
                    <p className="text-xs text-gray-500 mt-1">Chat with AI to build memory insights</p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
        
        {/* Agent Status */}
        <div className="p-4">
          <div className="bg-gray-800 rounded-lg">
            <button 
              onClick={() => toggleSection('agents')}
              className="w-full flex items-center justify-between p-4 hover:bg-gray-700/50 rounded-lg transition-colors"
            >
              <div className="flex items-center gap-2">
                <Users className="w-4 h-4 text-indigo-400" />
                <span className="text-sm font-medium">Agents</span>
                <span className="bg-indigo-500/20 text-indigo-400 text-xs px-2 py-0.5 rounded">
                  {agentStatus ? Object.keys(agentStatus.agents || {}).filter(name => name !== 'system_overview').length : 0}
                </span>
              </div>
              {expandedSections.agents ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>
            
            {expandedSections.agents && (
              <div className="px-4 pb-4 space-y-2 text-xs max-h-40 overflow-y-auto sidebar-scroll">
                {agentStatus && Object.entries(agentStatus.agents || {})
                  .filter(([agentName]) => agentName !== 'system_overview')
                  .map(([agentName, agentData]: [string, any]) => (
                  <div key={agentName} className="p-2 rounded bg-gray-700/30">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-indigo-400 font-medium">
                        {agentName.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      </span>
                      <div className={`w-2 h-2 rounded-full ${agentData.status === 'operational' ? 'bg-green-500' : 'bg-yellow-500'}`} />
                    </div>
                    <div className="text-gray-400">{agentData.role || 'Agent'}</div>
                    {agentData.metrics && (
                      <div className="text-gray-500 text-xs mt-1">
                        {agentData.metrics.success_rate !== undefined && (
                          <span>Success: {Math.round(agentData.metrics.success_rate)}%</span>
                        )}
                        {agentData.recent_collaborations !== undefined && (
                          <span className="ml-2">Recent: {agentData.recent_collaborations}</span>
                        )}
                      </div>
                    )}
                  </div>
                ))}
                
                {agentThoughts && agentThoughts.thoughts && agentThoughts.thoughts.length > 0 && (
                  <div className="pt-2 border-t border-gray-700">
                    <div className="text-gray-400 mb-1">Recent Thoughts</div>
                    {agentThoughts.thoughts.slice(0, 2).map((thought, index) => (
                      <div key={index} className="p-2 rounded bg-gray-700/20">
                        <div className="text-blue-400 mb-1">{thought.agent_name}</div>
                        <div className="text-gray-300">{thought.content.substring(0, 60)}...</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

          {/* Provider Status */}
          <div className="p-4">
            <div className="bg-gray-800 rounded-lg">
              <button 
                onClick={() => toggleSection('providers')}
                className="w-full flex items-center justify-between p-4 hover:bg-gray-700/50 rounded-lg transition-colors"
              >
                <div className="flex items-center gap-2">
                  <Globe className="w-4 h-4 text-cyan-400" />
                  <span className="text-sm font-medium">Providers</span>
                </div>
                {expandedSections.providers ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              </button>
              
              {expandedSections.providers && (
                <div className="px-4 pb-4 space-y-2 text-xs">
                  {providerStatus ? (
                    <>
                      {/* Provider Status */}
                      {Object.entries(providerStatus.providers || {}).map(([provider, details]: [string, any]) => (
                        <div key={provider} className="flex items-center justify-between p-2 rounded bg-gray-700/30">
                          <div className="flex items-center gap-2">
                            <span className="text-cyan-400 capitalize">{provider}</span>
                            {details.has_api_key && (
                              <span className="text-green-400 text-xs">✓ API Key</span>
                            )}
                          </div>
                          <div className="flex items-center gap-2">
                            <div className={`w-2 h-2 rounded-full ${
                              details.status === 'healthy' ? 'bg-green-500' : 
                              details.status === 'not_configured' ? 'bg-yellow-500' : 'bg-red-500'
                            }`} />
                            <span className={`text-xs ${
                              details.status === 'healthy' ? 'text-green-400' : 
                              details.status === 'not_configured' ? 'text-yellow-400' : 'text-red-400'
                            }`}>
                              {details.status}
                            </span>
                          </div>
                        </div>
                      ))}
                      
                      {/* Model Routing */}
                      {providerStatus.model_routing && (
                        <div className="mt-3">
                          <div className="text-gray-400 mb-2">Model Routing:</div>
                          <div className="space-y-1">
                            {Object.entries(providerStatus.model_routing).map(([category, tier]: [string, any]) => (
                              <div key={category} className="flex items-center justify-between p-1 rounded bg-gray-700/20">
                                <span className="text-blue-400 capitalize">{category}</span>
                                <span className="text-gray-300 text-xs">{tier}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </>
                  ) : (
                    <div className="p-3 rounded bg-gray-700/30 text-center">
                      <p className="text-xs text-gray-400">Loading providers...</p>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
          
          {/* Bottom padding for better scrolling */}
          <div className="h-4"></div>
        </div>


        {/* Actions */}
        <div className="p-4 border-t border-gray-800">
          <div className="flex justify-center">
            <button 
              onClick={refreshAllData}
              disabled={isRefreshing}
              className="bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 text-white py-2 px-4 rounded-lg text-sm flex items-center gap-2"
              title="Refresh all visible data"
            >
              <RefreshCw className={`w-4 h-4 ${isRefreshing ? 'animate-spin' : ''}`} />
              Refresh Data
              {isRefreshing && <span className="text-xs">...</span>}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Overlay */}
      {sidebarOpen && (
        <div 
          className="lg:hidden fixed inset-0 bg-black/50 z-40"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Chat Header */}
        <div className="h-16 bg-gray-900 border-b border-gray-800 flex items-center justify-between px-6">
          <div className="flex items-center gap-3">
            <button 
              onClick={() => setSidebarOpen(true)}
              className="lg:hidden p-2 hover:bg-gray-800 rounded-lg"
            >
              <Menu className="w-5 h-5" />
            </button>
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2 px-3 py-2 bg-gray-800 rounded-lg">
                <Bot className="w-5 h-5 text-blue-400" />
                <span className="text-white font-medium">Autonomous AI Agent</span>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <button
                onClick={clearChat}
                className="px-3 py-1.5 rounded-lg text-xs bg-red-600 hover:bg-red-700 text-white flex items-center gap-1"
                title="Clear chat and start fresh conversation"
              >
                <RefreshCw className="w-3 h-3" />
                Reset Chat
              </button>
              {chatStats && chatStats.total_messages > 0 && (
                <span className="bg-gray-700 text-gray-300 px-2 py-1 rounded text-xs flex items-center gap-1">
                  <Clock className="w-3 h-3" />
                  {chatStats.total_messages} messages
                </span>
              )}
            </div>
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full animate-pulse ${
                connectionStatus === 'connected' ? 'bg-green-500' : 
                connectionStatus === 'connecting' ? 'bg-yellow-500' : 'bg-red-500'
              }`} />
              <span className="text-sm text-gray-400 capitalize">{connectionStatus}</span>
            </div>
            {systemMetrics && (
              <div className="flex items-center gap-1 text-xs text-gray-400">
                <Cpu className="w-3 h-3" />
                <span>{systemMetrics.metrics.reasoning.active_cycles} active</span>
              </div>
            )}
          </div>
        </div>

        {/* Chat Area */}
        <div className="flex-1 overflow-y-auto p-4">
          <div className="max-w-4xl mx-auto space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex gap-3 ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                {message.sender === 'assistant' && (
                  <div className={`w-8 h-8 bg-gradient-to-br ${
                    message.fromHistory 
                      ? 'from-orange-500 to-red-600' 
                      : 'from-blue-500 to-purple-600'
                  } rounded-lg flex items-center justify-center flex-shrink-0`}>
                    <Bot className="w-4 h-4 text-white" />
                  </div>
                )}
                
                <div className={`max-w-2xl ${message.sender === 'user' ? 'order-first' : ''}`}>
                  <div
                    className={`px-4 py-3 rounded-lg ${
                      message.sender === 'user'
                        ? message.fromHistory
                          ? 'bg-blue-600/70 text-white border border-blue-500/50'
                          : 'bg-blue-600 text-white'
                        : message.fromHistory
                          ? 'bg-orange-800/20 text-white border border-orange-600/30'
                          : 'bg-gray-800 text-white border border-gray-700'
                    }`}
                  >
                    {message.streaming ? (
                      <div className="flex items-center gap-2">
                        <div className="flex gap-1">
                          <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                          <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                          <div className="w-2 h-2 bg-green-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                        </div>
                        <span className="text-sm text-gray-400">AI is thinking and collaborating...</span>
                      </div>
                    ) : (
                      <div className="leading-relaxed whitespace-pre-wrap">{message.content}</div>
                    )}
                  </div>
                  <div className={`text-xs text-gray-500 mt-1 flex items-center gap-1 ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                    {message.fromHistory && (
                      <span className="text-orange-400 text-xs px-1.5 py-0.5 bg-orange-500/20 rounded">
                        history
                      </span>
                    )}
                    <span>{formatTime(message.timestamp)}</span>
                  </div>
                </div>

                {message.sender === 'user' && (
                  <div className="w-8 h-8 bg-gradient-to-br from-green-500 to-emerald-600 rounded-lg flex items-center justify-center flex-shrink-0">
                    <User className="w-4 h-4 text-white" />
                  </div>
                )}
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Area */}
        <div className="p-4 bg-gray-900 border-t border-gray-800">
          <div className="max-w-4xl mx-auto">
            <form onSubmit={sendMessage} className="flex gap-3">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask your autonomous AI agent anything..."
                className="flex-1 bg-gray-800 text-white border border-gray-700 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                disabled={isLoading}
              />
              <button
                type="submit"
                disabled={isLoading || !input.trim()}
                className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white px-6 py-3 rounded-lg transition-colors flex items-center gap-2"
              >
                <Send className="w-4 h-4" />
                Send
              </button>
            </form>
            <p className="text-xs text-gray-500 mt-2 text-center">
              Autonomous AI with multi-agent collaboration • Live thinking • Real-time insights
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}