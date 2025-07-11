'use client'

import { useState, useEffect } from 'react'
import { ChevronDown, Bot, Brain, Search, Users, Activity, Check } from 'lucide-react'

interface Agent {
  name: string
  role: string
  status: 'operational' | 'limited' | 'offline'
  metrics?: {
    success_rate?: number
    recent_collaborations?: number
  }
  description: string
}

interface AgentDropdownProps {
  selectedAgent?: string
  onAgentSelect: (agentName: string) => void
  className?: string
}

const AVAILABLE_AGENTS: Record<string, Agent> = {
  'autonomous_agent': {
    name: 'Autonomous Agent',
    role: 'Primary AI Assistant',
    status: 'operational',
    description: 'Main autonomous AI with continuous reasoning and multi-agent collaboration'
  },
  'memory_agent': {
    name: 'Memory Agent',
    role: 'User Understanding Specialist',
    status: 'operational',
    description: 'Specialized in user context analysis, preference learning, and pattern recognition'
  },
  'research_agent': {
    name: 'Research Agent', 
    role: 'External Intelligence Specialist',
    status: 'operational',
    description: 'Handles web research, fact verification, and external data analysis'
  },
  'thinking_agent': {
    name: 'Thinking Agent',
    role: 'Continuous Reasoning Specialist', 
    status: 'operational',
    description: 'Provides continuous background thinking and insight generation'
  },
  'coordinator_agent': {
    name: 'Coordinator Agent',
    role: 'Strategic Synthesis Specialist',
    status: 'operational', 
    description: 'Coordinates multi-agent collaboration and strategic decision making'
  }
}

export function AgentDropdown({ selectedAgent = 'autonomous_agent', onAgentSelect, className = '' }: AgentDropdownProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [agentStatus, setAgentStatus] = useState<Record<string, any>>({})
  const [loading, setLoading] = useState(false)

  const fetchAgentStatus = async () => {
    try {
      setLoading(true)
      const response = await fetch('http://localhost:8000/agents/status')
      if (response.ok) {
        const data = await response.json()
        setAgentStatus(data.agents || {})
      }
    } catch (error) {
      console.error('Failed to fetch agent status:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchAgentStatus()
  }, [])

  const getAgentIcon = (agentName: string) => {
    switch (agentName) {
      case 'memory_agent': return Brain
      case 'research_agent': return Search
      case 'thinking_agent': return Activity
      case 'coordinator_agent': return Users
      default: return Bot
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'operational': return 'bg-green-500'
      case 'limited': return 'bg-yellow-500'
      default: return 'bg-gray-500'
    }
  }

  const currentAgent = AVAILABLE_AGENTS[selectedAgent]
  const CurrentIcon = getAgentIcon(selectedAgent)

  return (
    <div className={`relative ${className}`}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between p-3 bg-gray-800 hover:bg-gray-700 rounded-lg border border-gray-700 transition-colors"
      >
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
            <CurrentIcon className="w-4 h-4 text-white" />
          </div>
          <div className="text-left">
            <div className="text-sm font-medium text-white">{currentAgent?.name}</div>
            <div className="text-xs text-gray-400">{currentAgent?.role}</div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${getStatusColor(currentAgent?.status || 'offline')}`} />
          <ChevronDown className={`w-4 h-4 text-gray-400 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
        </div>
      </button>

      {isOpen && (
        <>
          <div 
            className="fixed inset-0 z-10" 
            onClick={() => setIsOpen(false)}
          />
          <div className="absolute top-full left-0 right-0 mt-2 bg-gray-800 border border-gray-700 rounded-lg shadow-xl z-20 max-h-80 overflow-y-auto">
            {Object.entries(AVAILABLE_AGENTS).map(([agentKey, agent]) => {
              const Icon = getAgentIcon(agentKey)
              const statusData = agentStatus[agentKey] || {}
              const isSelected = selectedAgent === agentKey
              
              return (
                <button
                  key={agentKey}
                  onClick={() => {
                    onAgentSelect(agentKey)
                    setIsOpen(false)
                  }}
                  className="w-full flex items-center gap-3 p-3 hover:bg-gray-700 transition-colors text-left"
                >
                  <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${
                    isSelected 
                      ? 'bg-gradient-to-br from-blue-500 to-purple-600' 
                      : 'bg-gray-700'
                  }`}>
                    <Icon className="w-4 h-4 text-white" />
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-white truncate">{agent.name}</span>
                      {isSelected && <Check className="w-4 h-4 text-green-400" />}
                    </div>
                    <div className="text-xs text-gray-400 truncate">{agent.role}</div>
                    <div className="text-xs text-gray-500 mt-1 truncate">{agent.description}</div>
                    
                    {statusData.metrics && (
                      <div className="flex items-center gap-2 mt-1 text-xs text-gray-500">
                        {statusData.metrics.success_rate !== undefined && (
                          <span>Success: {Math.round(statusData.metrics.success_rate)}%</span>
                        )}
                        {statusData.metrics.recent_collaborations !== undefined && (
                          <span>Recent: {statusData.metrics.recent_collaborations}</span>
                        )}
                      </div>
                    )}
                  </div>
                  
                  <div className="flex flex-col items-end gap-1">
                    <div className={`w-2 h-2 rounded-full ${getStatusColor(statusData.status || agent.status)}`} />
                    <span className="text-xs text-gray-500 capitalize">
                      {statusData.status || agent.status}
                    </span>
                  </div>
                </button>
              )
            })}
            
            {loading && (
              <div className="p-3 text-center">
                <div className="text-xs text-gray-400">Loading agent status...</div>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  )
}