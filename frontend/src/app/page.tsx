'use client'

import { useState, useEffect, useRef } from 'react'
import { Send, Bot, User, Brain, RefreshCw } from 'lucide-react'
import ModernTextAnimation from '../components/ui/ModernTextAnimation'
import LeftSidebar from '../components/ui/LeftSidebar'
import AgentStreamPanel from '../components/ui/AgentStreamPanel'

// Simplified interfaces - only what we actually use
interface Message {
  id: string
  content: string
  sender: 'user' | 'assistant'
  timestamp: Date
  streaming?: boolean
}

interface SystemStatus {
  system_health: string
  agents: Record<string, any>
  websocket_connections: any
}


interface AutonomousInsight {
  insight_id: string
  user_name: string
  insight_type: string
  content: string
  created_at: string
  metadata: any
}

interface AutonomousInsights {
  status: string
  user_name: string
  insights: AutonomousInsight[]
  total_insights: number
  timestamp: string
}

interface ChatHistory {
  messages: any[]
  total_messages: number
  has_more: boolean
}

export default function Dashboard() {
  // Essential state only (reduced from 24 to 12 variables)
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isStreamPanelVisible, setIsStreamPanelVisible] = useState(false)
  const streamPanelRef = useRef<any>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null)
  const [chatHistory, setChatHistory] = useState<ChatHistory | null>(null)
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'connecting'>('disconnected')
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [unifiedWs, setUnifiedWs] = useState<WebSocket | null>(null)
  const [streamData, setStreamData] = useState<any>(null)
  const [lastUpdated, setLastUpdated] = useState<number>(0)
  const [isClearing, setIsClearing] = useState(false)
  const [autonomousInsights, setAutonomousInsights] = useState<AutonomousInsights | null>(null)
  const [isLoadingInsights, setIsLoadingInsights] = useState(false)
  
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const lastSubmissionRef = useRef<number>(0)

  // Scroll to bottom when new messages arrive
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Load initial data
  useEffect(() => {
    loadSystemStatus()
    loadChatHistory()
    loadAutonomousInsights()
    setupUnifiedWebSocket()
    
    // Set up periodic updates (simplified)
    const statusInterval = setInterval(loadSystemStatus, 120000)   // 2 minutes
    const autonomousInsightsInterval = setInterval(loadAutonomousInsights, 300000) // 5 minutes
    
    return () => {
      clearInterval(statusInterval)
      clearInterval(autonomousInsightsInterval)
      // Don't close WebSocket on cleanup - it should persist
    }
  }, [])

  // Load system status from /status endpoint
  const loadSystemStatus = async () => {
    try {
      const response = await fetch('http://localhost:8000/status')
      if (response.ok) {
        const data = await response.json()
        setSystemStatus(data)
      }
    } catch (error) {
      console.error('Failed to load system status:', error)
    }
  }


  // Load autonomous insights for current user
  const loadAutonomousInsights = async () => {
    if (isLoadingInsights) return
    
    setIsLoadingInsights(true)
    try {
      // Backend automatically uses configured user from settings.yaml
      const response = await fetch('http://localhost:8000/autonomous/insights')
      
      if (response.ok) {
        const data = await response.json()
        setAutonomousInsights(data)
      } else {
        console.error('Failed to load autonomous insights:', response.status)
      }
    } catch (error) {
      console.error('Failed to load autonomous insights:', error)
    } finally {
      setIsLoadingInsights(false)
    }
  }

  // Clear autonomous insights for current user
  const clearAutonomousInsights = async () => {
    if (!autonomousInsights) return
    
    const userName = autonomousInsights.user_name
    if (!confirm(`Are you sure you want to clear all autonomous insights for ${userName}? This action cannot be undone.`)) {
      return
    }
    
    try {
      // Backend automatically uses configured user from settings.yaml
      const response = await fetch('http://localhost:8000/autonomous/insights', {
        method: 'DELETE',
      })
      
      if (response.ok) {
        setAutonomousInsights(null)
        console.log('✅ Autonomous insights cleared successfully')
      } else {
        console.error('Failed to clear autonomous insights:', response.status)
      }
    } catch (error) {
      console.error('Failed to clear autonomous insights:', error)
    }
  }

  // Generate insights manually
  const generateInsights = async () => {
    if (isLoadingInsights) return
    
    setIsLoadingInsights(true)
    try {
      // Backend automatically uses configured user from settings.yaml
      const response = await fetch('http://localhost:8000/autonomous/trigger', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          operation_type: 'insight_generation',
          trigger_source: 'manual',
          broadcast_updates: true
        }),
      })
      
      if (response.ok) {
        console.log('✅ Insights generation triggered successfully')
        // Wait a moment then refresh insights
        setTimeout(loadAutonomousInsights, 2000)
      } else {
        console.error('Failed to trigger insights generation:', response.status)
      }
    } catch (error) {
      console.error('Failed to trigger insights generation:', error)
    } finally {
      setIsLoadingInsights(false)
    }
  }

  // Load chat history (uses configured user from settings.yaml)
  const loadChatHistory = async () => {
    try {
      const response = await fetch('http://localhost:8000/chat/history?limit=20&offset=0')
      if (response.ok) {
        const data = await response.json()
        setChatHistory(data)
        
        // Convert history to messages format with deduplication
        if (data.messages) {
          const historyMessages: Message[] = data.messages.map((msg: any, index: number) => ({
            id: `history-${msg.timestamp}-${index}`, // Use timestamp + index for unique ID
            content: msg.sender === 'user' ? msg.message : msg.response || msg.message,
            sender: msg.sender,
            timestamp: new Date(msg.timestamp)
          }))
          
          // Only update if messages have actually changed
          const currentMessageIds = messages.map(m => m.id).sort()
          const newMessageIds = historyMessages.map(m => m.id).sort()
          const messagesChanged = JSON.stringify(currentMessageIds) !== JSON.stringify(newMessageIds)
          
          if (messagesChanged) {
            setMessages(historyMessages)
          }
        }
      }
    } catch (error) {
      console.error('Failed to load chat history:', error)
    }
  }

  // Setup unified WebSocket connection (uses configured user from settings.yaml)
  const setupUnifiedWebSocket = () => {
    // Prevent multiple connections - close existing first
    if (unifiedWs) {
      if (unifiedWs.readyState === WebSocket.OPEN) {
        console.log('Closing existing WebSocket connection')
        unifiedWs.close()
      }
      setUnifiedWs(null)
    }
    
    try {
      setConnectionStatus('connecting')
      const ws = new WebSocket('ws://localhost:8000/stream')
      
      ws.onopen = () => {
        console.log('🔄 Unified WebSocket connected')
        setConnectionStatus('connected')
        setUnifiedWs(ws)
      }
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          setStreamData(data)
          setLastUpdated(Date.now())
          
          // Handle different message types
          if (data.type === 'autonomous_insight') {
            console.log('🤖 Autonomous insight received')
            loadAutonomousInsights() // Refresh autonomous insights
          } else if (data.type === 'chat_update') {
            console.log('💬 Chat updated')
            loadChatHistory() // Refresh chat
          } else if (data.type === 'connection_established') {
            console.log('✅ WebSocket established')
          } else if (data.type === 'streaming_ready') {
            console.log('🔄 Agent streaming ready:', data.data.supported_events)
          } else if (data.type === 'agent_stream_batch') {
            console.log('🔄 Agent stream batch received:', data.batch_size, 'updates')
            // Forward to streaming panel
            if (streamPanelRef.current && data.updates) {
              data.updates.forEach((update: any) => {
                streamPanelRef.current.handleStreamUpdate(update)
              })
            }
            // Auto-show stream panel when workflow starts
            if (data.updates?.some((u: any) => u.event_type === 'workflow_start')) {
              setIsStreamPanelVisible(true)
            }
          } else if (data.type === 'chat_response') {
            console.log('💬 Chat response received via WebSocket:', data)
            // Don't reload chat history here - it causes duplication
          } else {
            console.log('🔄 WebSocket message:', data.type, data)
          }
          
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error)
        }
      }
      
      ws.onclose = (event) => {
        console.log('🔌 Unified WebSocket disconnected', event.code, event.reason)
        setConnectionStatus('disconnected')
        if (unifiedWs === ws) {
          setUnifiedWs(null)
        }
        
        // Only reconnect if it wasn't a manual close (code 1000 = normal closure)
        if (event.code !== 1000 && event.code !== 1001) {
          console.log('Attempting to reconnect in 5 seconds...')
          setTimeout(setupUnifiedWebSocket, 5000)
        }
      }
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        setConnectionStatus('disconnected')
      }
      
      // Set the websocket only after all handlers are attached
      setUnifiedWs(ws)
      
    } catch (error) {
      console.error('Failed to setup WebSocket:', error)
      setConnectionStatus('disconnected')
    }
  }


  // Clear memory function
  const clearMemory = async () => {
    if (isClearing) return
    
    // Backend automatically uses configured user from settings.yaml
    const userName = 'configured user'  // Will be determined by backend
    
    if (!confirm(`Are you sure you want to clear all session chat and working memory for the ${userName}? This action cannot be undone.`)) {
      return
    }
    
    setIsClearing(true)
    try {
      const response = await fetch('http://localhost:8000/memory/cleanup', {
        method: 'DELETE',
      })
      
      if (response.ok) {
        const data = await response.json()
        console.log('Memory cleared successfully:', data)
        
        // Clear local chat messages
        setMessages([])
        setChatHistory(null)
        
        // Refresh system status and autonomous insights
        await Promise.all([
          loadSystemStatus(),
          loadAutonomousInsights()
        ])
        
        alert('Memory cleared successfully!')
      } else {
        throw new Error('Failed to clear memory')
      }
    } catch (error) {
      console.error('Failed to clear memory:', error)
      alert('Failed to clear memory. Please try again.')
    } finally {
      setIsClearing(false)
    }
  }

  // Send message
  const sendMessage = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    // Prevent duplicate submissions within 1 second
    const now = Date.now()
    if (now - lastSubmissionRef.current < 1000) {
      console.log('🚫 Preventing duplicate submission')
      return
    }
    lastSubmissionRef.current = now

    const messageId = now.toString()
    console.log(`📤 Sending message ${messageId}: "${input}"`)

    const userMessage: Message = {
      id: messageId,
      content: input,
      sender: 'user',
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    const messageToSend = input
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
          message: messageToSend,
          context: {}
        }),
      })

      if (!response.ok) {
        throw new Error('Network response was not ok')
      }

      const data = await response.json()
      
      // Remove streaming indicator
      setMessages(prev => prev.filter(msg => msg.id !== 'streaming'))

      // Add assistant response
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: data.response || 'No response received',
        sender: 'assistant',
        timestamp: new Date()
      }

      setMessages(prev => [...prev, assistantMessage])
      
      // Refresh insights after chat
      setTimeout(loadAutonomousInsights, 1000)
      
    } catch (error) {
      console.error('Error sending message:', error)
      setMessages(prev => prev.filter(msg => msg.id !== 'streaming'))
      
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: 'Sorry, there was an error processing your message. Please try again.',
        sender: 'assistant',
        timestamp: new Date()
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="h-screen bg-gray-950 text-white flex">
      {/* Sidebar */}
      <LeftSidebar 
        systemStatus={systemStatus}
        connectionStatus={connectionStatus}
        lastUpdated={lastUpdated}
        onClearMemory={clearMemory}
        isClearing={isClearing}
        autonomousInsights={autonomousInsights}
        onLoadAutonomousInsights={loadAutonomousInsights}
        onClearAutonomousInsights={clearAutonomousInsights}
        onGenerateInsights={generateInsights}
        isLoadingInsights={isLoadingInsights}
      />

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <div className="border-b border-gray-800 p-4 flex items-center justify-between flex-shrink-0">
          <div className="flex items-center space-x-2">
            <Brain className="w-6 h-6 text-purple-400" />
            <h1 className="text-xl font-semibold">Autonomous AI Assistant</h1>
          </div>

          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${
                connectionStatus === 'connected' ? 'bg-green-400' : 
                connectionStatus === 'connecting' ? 'bg-yellow-400' : 'bg-red-400'
              }`} />
              <span className="text-sm text-gray-400 capitalize">{connectionStatus}</span>
            </div>
          </div>
        </div>

        {/* Agent Stream Panel */}
        <div className="px-4 py-2">
          <AgentStreamPanel
            ref={streamPanelRef}
            isVisible={isStreamPanelVisible}
            onToggle={() => setIsStreamPanelVisible(!isStreamPanelVisible)}
          />
        </div>

        {/* Chat Area */}
        <div className="flex-1 flex flex-col min-h-0">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.length === 0 ? (
              <div className="flex items-center justify-center h-full">
                <div className="text-center max-w-md">
                  <ModernTextAnimation text="Welcome to Autonomous AI" />
                  <p className="text-gray-400 mt-4">
                    Start a conversation with your autonomous AI assistant
                  </p>
                </div>
              </div>
            ) : (
              messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                      message.sender === 'user'
                        ? 'bg-purple-600 text-white'
                        : message.streaming
                        ? 'bg-gray-800 text-gray-300'
                        : 'bg-gray-800 text-white'
                    }`}
                  >
                    <div className="flex items-start space-x-2">
                      {message.sender === 'assistant' && (
                        <Bot size={16} className="mt-1 text-purple-400 flex-shrink-0" />
                      )}
                      {message.sender === 'user' && (
                        <User size={16} className="mt-1 text-white flex-shrink-0" />
                      )}
                      <div className="flex-1">
                        {message.streaming ? (
                          <div className="flex items-center space-x-2">
                            <div className="animate-pulse">Thinking...</div>
                            <div className="flex space-x-1">
                              <div className="w-1 h-1 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                              <div className="w-1 h-1 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                              <div className="w-1 h-1 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                            </div>
                          </div>
                        ) : (
                          <p className="text-sm">{message.content}</p>
                        )}
                        <p className="text-xs text-gray-400 mt-1">
                          {message.timestamp.toLocaleTimeString()}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              ))
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="border-t border-gray-800 p-4 flex-shrink-0">
            <form onSubmit={sendMessage} className="flex space-x-3">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Type your message..."
                className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 focus:outline-none focus:border-purple-500"
                disabled={isLoading}
              />
              <button
                type="submit"
                disabled={isLoading || !input.trim()}
                className="bg-purple-600 hover:bg-purple-700 disabled:bg-gray-700 px-4 py-2 rounded-lg transition-colors flex items-center space-x-2"
              >
                <Send size={18} />
                <span>Send</span>
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  )
}