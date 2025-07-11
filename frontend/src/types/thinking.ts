export interface ThinkingContent {
  content: string[]
  agent: string
  context?: string
  timestamp: string
  has_thinking: boolean
}

export interface ChatMessage {
  id: string
  message: string
  sender: 'user' | 'ai'
  timestamp: string
  thinking?: string[]
  hasThinking?: boolean
  agent?: string
  confidence?: number
  modelUsed?: string
}

export interface ThinkingEvent {
  id: string
  type: 'observation' | 'reflection' | 'pattern' | 'insight' | 'decision' | 'reasoning'
  content: string[]
  agent: string
  timestamp: string
  confidence?: number
  context?: string
  reasoning_chain?: string[]
}

export interface WebSocketMessage {
  type: string
  data?: any
  user_id?: string
  timestamp?: string
  thinking?: ThinkingContent
}

export interface AgentStatus {
  name: string
  status: 'operational' | 'busy' | 'error' | 'idle'
  confidence?: number
  lastActivity?: string
  currentTask?: string
}

export interface AutonomousInsight {
  id: string
  title: string
  content: string
  confidence: number
  category: string
  timestamp: string
  agent?: string
}

export interface DecisionEvent {
  id: string
  type: 'proactive' | 'reactive' | 'scheduled'
  description: string
  reasoning: string
  confidence: number
  reversible: boolean
  impact_level: 'low' | 'medium' | 'high'
  requires_approval: boolean
  timestamp: string
}