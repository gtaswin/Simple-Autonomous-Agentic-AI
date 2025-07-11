'use client'

import React, { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Brain, Pause, Play, X, Settings } from 'lucide-react'
import ThoughtBubble from './ThoughtBubble'

interface ThinkingEvent {
  id: string
  type: 'observation' | 'reflection' | 'pattern' | 'insight' | 'decision' | 'reasoning'
  content: string[]
  agent: string
  timestamp: string
  confidence?: number
  context?: string
  reasoning_chain?: string[]
}

interface ThinkingStreamProps {
  websocketUrl?: string
  maxEvents?: number
  autoScroll?: boolean
  showControls?: boolean
  className?: string
}

export function ThinkingStream({ 
  websocketUrl = 'ws://localhost:8000/agent-stream',
  maxEvents = 50,
  autoScroll = true,
  showControls = true,
  className = ''
}: ThinkingStreamProps) {
  const [events, setEvents] = useState<ThinkingEvent[]>([])
  const [isConnected, setIsConnected] = useState(false)
  const [isPaused, setIsPaused] = useState(false)
  const [isReconnecting, setIsReconnecting] = useState(false)
  const streamRef = useRef<HTMLDivElement>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>()

  // Auto-scroll to bottom when new events arrive
  useEffect(() => {
    if (autoScroll && streamRef.current && !isPaused) {
      streamRef.current.scrollTop = streamRef.current.scrollHeight
    }
  }, [events, autoScroll, isPaused])

  // WebSocket connection management
  const connectWebSocket = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    try {
      setIsReconnecting(true)
      wsRef.current = new WebSocket(websocketUrl)

      wsRef.current.onopen = () => {
        setIsConnected(true)
        setIsReconnecting(false)
        console.log('🧠 Thinking stream connected')
      }

      wsRef.current.onmessage = (event) => {
        if (isPaused) return

        try {
          const data = JSON.parse(event.data)
          
          // Handle different message types
          if (data.type === 'thinking_content' && data.thinking) {
            const thinkingEvent: ThinkingEvent = {
              id: `thinking-${Date.now()}-${Math.random()}`,
              type: 'reasoning',
              content: data.thinking.content || [],
              agent: data.thinking.agent || 'AI',
              timestamp: data.thinking.timestamp || new Date().toISOString(),
              confidence: data.thinking.confidence,
              context: data.thinking.context
            }
            
            if (thinkingEvent.content.length > 0) {
              setEvents(prev => [...prev.slice(-(maxEvents - 1)), thinkingEvent])
            }
          }
          
          // Handle continuous thinking events
          if (data.type === 'continuous_thinking' && data.data) {
            const event: ThinkingEvent = {
              id: `continuous-${Date.now()}-${Math.random()}`,
              type: data.data.thinking_type || 'observation',
              content: [data.data.content],
              agent: 'thinking_agent',
              timestamp: data.data.timestamp || new Date().toISOString(),
              confidence: data.data.confidence,
              reasoning_chain: data.data.reasoning_chain
            }
            
            setEvents(prev => [...prev.slice(-(maxEvents - 1)), event])
          }

          // Handle new thoughts
          if (data.type === 'new_thought' && data.data) {
            const event: ThinkingEvent = {
              id: data.data.id || `thought-${Date.now()}`,
              type: data.data.type || 'observation',
              content: [data.data.content],
              agent: 'autonomous_agent',
              timestamp: data.data.timestamp || new Date().toISOString()
            }
            
            setEvents(prev => [...prev.slice(-(maxEvents - 1)), event])
          }

        } catch (error) {
          console.error('Failed to parse thinking stream message:', error)
        }
      }

      wsRef.current.onclose = () => {
        setIsConnected(false)
        setIsReconnecting(false)
        
        // Auto-reconnect after 3 seconds
        reconnectTimeoutRef.current = setTimeout(() => {
          if (!isPaused) {
            connectWebSocket()
          }
        }, 3000)
      }

      wsRef.current.onerror = (error) => {
        console.error('Thinking stream WebSocket error:', error)
        setIsReconnecting(false)
      }

    } catch (error) {
      console.error('Failed to connect to thinking stream:', error)
      setIsReconnecting(false)
    }
  }

  // Connect on component mount
  useEffect(() => {
    connectWebSocket()
    
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [])

  // Pause/Resume functionality
  const togglePause = () => {
    setIsPaused(!isPaused)
    
    if (isPaused && !isConnected) {
      connectWebSocket()
    }
  }

  // Clear events
  const clearEvents = () => {
    setEvents([])
  }

  // Get type-specific styling
  const getEventTypeStyle = (type: string) => {
    const styles = {
      observation: 'border-l-blue-500 bg-blue-500/5',
      reflection: 'border-l-purple-500 bg-purple-500/5',
      pattern: 'border-l-green-500 bg-green-500/5',
      insight: 'border-l-yellow-500 bg-yellow-500/5',
      decision: 'border-l-red-500 bg-red-500/5',
      reasoning: 'border-l-indigo-500 bg-indigo-500/5'
    }
    return styles[type as keyof typeof styles] || styles.observation
  }

  return (
    <div className={`thinking-stream h-full flex flex-col ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 bg-gray-800/50 backdrop-blur-sm border-b border-gray-700/50">
        <div className="flex items-center gap-3">
          <div className="relative">
            <Brain className="w-6 h-6 text-purple-400" />
            {isConnected && (
              <motion.div
                className="absolute -top-1 -right-1 w-3 h-3 bg-green-500 rounded-full"
                animate={{ scale: [1, 1.2, 1] }}
                transition={{ duration: 2, repeat: Infinity }}
              />
            )}
          </div>
          <div>
            <h3 className="text-lg font-semibold text-white">AI Thinking Stream</h3>
            <p className="text-sm text-gray-400">
              {isConnected ? (
                isPaused ? 'Paused' : 'Live reasoning process'
              ) : isReconnecting ? (
                'Reconnecting...'
              ) : (
                'Disconnected'
              )}
            </p>
          </div>
        </div>

        {/* Controls */}
        {showControls && (
          <div className="flex items-center gap-2">
            <motion.button
              onClick={togglePause}
              className="p-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              {isPaused ? (
                <Play className="w-4 h-4 text-green-400" />
              ) : (
                <Pause className="w-4 h-4 text-yellow-400" />
              )}
            </motion.button>

            <motion.button
              onClick={clearEvents}
              className="p-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <X className="w-4 h-4 text-red-400" />
            </motion.button>
          </div>
        )}
      </div>

      {/* Event Stream */}
      <div 
        ref={streamRef}
        className="flex-1 overflow-y-auto p-4 space-y-4 scroll-smooth"
        style={{
          scrollbarWidth: 'thin',
          scrollbarColor: '#374151 #1f2937'
        }}
      >
        <AnimatePresence mode="popLayout">
          {events.length === 0 ? (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex flex-col items-center justify-center py-12 text-center"
            >
              <Brain className="w-12 h-12 text-gray-600 mb-4" />
              <p className="text-gray-400 text-lg">Waiting for AI thoughts...</p>
              <p className="text-gray-600 text-sm mt-2">
                The thinking stream will show AI reasoning in real-time
              </p>
            </motion.div>
          ) : (
            events.map((event, index) => (
              <motion.div
                key={event.id}
                initial={{ opacity: 0, x: -20, scale: 0.95 }}
                animate={{ opacity: 1, x: 0, scale: 1 }}
                exit={{ opacity: 0, x: 20, scale: 0.95 }}
                transition={{ duration: 0.3, delay: 0.05 * index }}
                layout
              >
                <ThoughtBubble
                  thinking={event.content}
                  agent={event.agent}
                  timestamp={event.timestamp}
                  confidence={event.confidence}
                  className={`border-l-4 ${getEventTypeStyle(event.type)}`}
                />
              </motion.div>
            ))
          )}
        </AnimatePresence>
      </div>

      {/* Status Bar */}
      <div className="p-3 bg-gray-900/50 border-t border-gray-700/50 text-center">
        <div className="flex items-center justify-between text-sm text-gray-500">
          <span>{events.length} thoughts captured</span>
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
            <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ThinkingStream