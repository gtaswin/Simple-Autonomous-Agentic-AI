'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { ThinkingEvent, WebSocketMessage } from '@/types/thinking'

interface UseThinkingStreamOptions {
  websocketUrl?: string
  maxEvents?: number
  reconnectDelay?: number
  maxReconnectAttempts?: number
  onThinkingEvent?: (event: ThinkingEvent) => void
  onConnectionChange?: (connected: boolean) => void
}

export function useThinkingStream({
  websocketUrl = process.env.NEXT_PUBLIC_WS_URL ? `${process.env.NEXT_PUBLIC_WS_URL}/agent-stream` : 'ws://localhost:8000/agent-stream',
  maxEvents = 50,
  reconnectDelay = 3000,
  maxReconnectAttempts = 5,
  onThinkingEvent,
  onConnectionChange
}: UseThinkingStreamOptions = {}) {
  
  const [events, setEvents] = useState<ThinkingEvent[]>([])
  const [isConnected, setIsConnected] = useState(false)
  const [isReconnecting, setIsReconnecting] = useState(false)
  const [connectionAttempts, setConnectionAttempts] = useState(0)
  
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>()
  const reconnectAttemptsRef = useRef(0)

  // Add a new thinking event
  const addEvent = useCallback((event: ThinkingEvent) => {
    setEvents(prev => {
      const newEvents = [...prev, event]
      return newEvents.slice(-maxEvents) // Keep only the latest events
    })
    
    onThinkingEvent?.(event)
  }, [maxEvents, onThinkingEvent])

  // Clear all events
  const clearEvents = useCallback(() => {
    setEvents([])
  }, [])

  // Process incoming WebSocket messages
  const processMessage = useCallback((data: WebSocketMessage) => {
    try {
      // Handle thinking content
      if (data.type === 'thinking_content' && data.thinking) {
        const event: ThinkingEvent = {
          id: `thinking-${Date.now()}-${Math.random().toString(36).slice(2)}`,
          type: 'reasoning',
          content: data.thinking.content || [],
          agent: data.thinking.agent || 'AI',
          timestamp: data.thinking.timestamp || new Date().toISOString(),
          context: data.thinking.context
        }
        
        if (event.content.length > 0) {
          addEvent(event)
        }
      }
      
      // Handle continuous thinking
      if (data.type === 'continuous_thinking' && data.data) {
        const event: ThinkingEvent = {
          id: `continuous-${Date.now()}-${Math.random().toString(36).slice(2)}`,
          type: data.data.thinking_type || 'observation',
          content: [data.data.content].filter(Boolean),
          agent: 'thinking_agent',
          timestamp: data.data.timestamp || new Date().toISOString(),
          confidence: data.data.confidence,
          reasoning_chain: data.data.reasoning_chain
        }
        
        if (event.content.length > 0) {
          addEvent(event)
        }
      }

      // Handle new thoughts
      if (data.type === 'new_thought' && data.data?.content) {
        const event: ThinkingEvent = {
          id: data.data.id || `thought-${Date.now()}`,
          type: data.data.type || 'observation',
          content: [data.data.content],
          agent: 'autonomous_agent',
          timestamp: data.data.timestamp || new Date().toISOString()
        }
        
        addEvent(event)
      }

      // Handle autonomous insights
      if (data.type === 'autonomous_insight' && data.data?.content) {
        const event: ThinkingEvent = {
          id: data.data.insight_id || `insight-${Date.now()}`,
          type: 'insight',
          content: [data.data.content],
          agent: 'insight_agent',
          timestamp: data.data.timestamp || new Date().toISOString(),
          confidence: data.data.confidence
        }
        
        addEvent(event)
      }

    } catch (error) {
      console.error('Failed to process thinking stream message:', error)
    }
  }, [addEvent])

  // Connect to WebSocket
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return
    if (reconnectAttemptsRef.current >= maxReconnectAttempts) {
      console.warn('Max reconnection attempts reached for thinking stream')
      return
    }

    try {
      setIsReconnecting(true)
      setConnectionAttempts(reconnectAttemptsRef.current)
      
      wsRef.current = new WebSocket(websocketUrl)

      wsRef.current.onopen = () => {
        setIsConnected(true)
        setIsReconnecting(false)
        reconnectAttemptsRef.current = 0
        setConnectionAttempts(0)
        onConnectionChange?.(true)
        console.log('🧠 Thinking stream connected')
      }

      wsRef.current.onmessage = (event) => {
        try {
          const data: WebSocketMessage = JSON.parse(event.data)
          processMessage(data)
        } catch (error) {
          console.error('Failed to parse thinking stream message:', error)
        }
      }

      wsRef.current.onclose = (event) => {
        setIsConnected(false)
        setIsReconnecting(false)
        onConnectionChange?.(false)
        
        // Automatic reconnection with exponential backoff
        if (event.code !== 1000 && reconnectAttemptsRef.current < maxReconnectAttempts) {
          reconnectAttemptsRef.current++
          const delay = reconnectDelay * Math.pow(2, reconnectAttemptsRef.current - 1)
          
          reconnectTimeoutRef.current = setTimeout(() => {
            connect()
          }, delay)
          
          console.log(`Thinking stream disconnected, reconnecting in ${delay}ms (attempt ${reconnectAttemptsRef.current}/${maxReconnectAttempts})`)
        } else {
          console.log('Thinking stream connection closed')
        }
      }

      wsRef.current.onerror = (error) => {
        console.error('Thinking stream WebSocket error:', error)
        setIsReconnecting(false)
      }

    } catch (error) {
      console.error('Failed to connect to thinking stream:', error)
      setIsReconnecting(false)
    }
  }, [websocketUrl, maxReconnectAttempts, reconnectDelay, onConnectionChange, processMessage])

  // Disconnect from WebSocket
  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
    }
    
    if (wsRef.current) {
      wsRef.current.close(1000, 'Manual disconnect')
      wsRef.current = null
    }
    
    setIsConnected(false)
    setIsReconnecting(false)
    reconnectAttemptsRef.current = 0
  }, [])

  // Manual reconnect
  const reconnect = useCallback(() => {
    disconnect()
    reconnectAttemptsRef.current = 0
    setTimeout(connect, 100)
  }, [disconnect, connect])

  // Connect on mount, disconnect on unmount
  useEffect(() => {
    connect()
    
    return () => {
      disconnect()
    }
  }, [connect, disconnect])

  return {
    events,
    isConnected,
    isReconnecting,
    connectionAttempts,
    connect,
    disconnect,
    reconnect,
    clearEvents,
    addEvent
  }
}

export default useThinkingStream