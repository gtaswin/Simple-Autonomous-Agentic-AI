'use client'

import React, { useState, useEffect } from 'react'
import { Activity, Brain, Search, FileText, Users, Clock, CheckCircle, XCircle, Loader2, ChevronDown, ChevronUp } from 'lucide-react'

// Agent streaming types
interface AgentStatus {
  agent_name: string
  status: 'waiting' | 'active' | 'completed' | 'error' | 'skipped'
  progress_percentage: number
  current_activity: string
  execution_time?: number
  result_preview?: string
  error_message?: string
  metadata?: any
  timestamp: string
}

interface WorkflowProgress {
  overall_progress: number
  completed_agents: number
  total_agents: number
  current_phase: string
}

interface StreamUpdate {
  type: string
  event_type: string
  data: {
    workflow_id: string
    user_name: string
    agent_update?: AgentStatus
    workflow_progress?: WorkflowProgress
    agents_status?: Record<string, string>
    final_response?: string
    total_execution_time?: number
    agents_executed?: string[]
    workflow_pattern?: string
  }
  timestamp: string
}

interface AgentStreamPanelProps {
  isVisible: boolean
  onToggle: () => void
  className?: string
}

const AgentStreamPanel: React.FC<AgentStreamPanelProps> = ({
  isVisible,
  onToggle,
  className = ""
}) => {
  const [currentWorkflow, setCurrentWorkflow] = useState<string | null>(null)
  const [workflowProgress, setWorkflowProgress] = useState<WorkflowProgress | null>(null)
  const [agentStatuses, setAgentStatuses] = useState<Record<string, AgentStatus>>({})
  const [isWorkflowActive, setIsWorkflowActive] = useState(false)
  const [workflowStartTime, setWorkflowStartTime] = useState<Date | null>(null)
  const [recentUpdates, setRecentUpdates] = useState<StreamUpdate[]>([])
  const [parallelExecution, setParallelExecution] = useState<{
    isActive: boolean
    concurrentAgents: string[]
    phase: string
    speedupFactor?: number
  }>({ isActive: false, concurrentAgents: [], phase: '' })

  // Agent configuration for display
  const agentConfig = {
    router: { 
      icon: Activity, 
      label: 'Router', 
      color: 'text-purple-400',
      description: 'Analyzes request complexity'
    },
    memory_reader: { 
      icon: Brain, 
      label: 'Memory Reader', 
      color: 'text-blue-400',
      description: 'Retrieves relevant context'
    },
    knowledge_agent: { 
      icon: Search, 
      label: 'Knowledge Agent', 
      color: 'text-green-400',
      description: 'Performs external research'
    },
    organizer: { 
      icon: Users, 
      label: 'Organizer', 
      color: 'text-orange-400',
      description: 'Synthesizes final response'
    },
    memory_writer: { 
      icon: FileText, 
      label: 'Memory Writer', 
      color: 'text-indigo-400',
      description: 'Stores conversation data'
    }
  }

  // Connect to WebSocket for streaming updates
  useEffect(() => {
    // This will be handled by the parent component's WebSocket connection
    // The parent should call handleStreamUpdate when receiving agent stream events
  }, [])

  // Handle incoming stream updates
  const handleStreamUpdate = (update: StreamUpdate) => {
    if (update.type !== 'agent_stream_batch') return

    // Handle batch updates
    if (update.event_type === 'parallel_execution_start') {
      const data = update.data
      setParallelExecution({
        isActive: true,
        concurrentAgents: data.parallel_agents || [],
        phase: data.execution_phase || 'unknown',
        speedupFactor: undefined
      })
    }
    
    else if (update.event_type === 'parallel_execution_complete') {
      const data = update.data
      setParallelExecution(prev => ({
        ...prev,
        isActive: false,
        speedupFactor: data.speedup_factor
      }))
      
      // Clear parallel indicators after delay
      setTimeout(() => {
        setParallelExecution({ isActive: false, concurrentAgents: [], phase: '' })
      }, 3000)
    }
    
    else if (update.event_type === 'workflow_start') {
      setCurrentWorkflow(update.data.workflow_id)
      setIsWorkflowActive(true)
      setWorkflowStartTime(new Date())
      setAgentStatuses({})
      setWorkflowProgress({
        overall_progress: 0,
        completed_agents: 0,
        total_agents: update.data.agents_executed?.length || 5,
        current_phase: 'Starting workflow...'
      })
    }
    
    else if (update.event_type === 'agent_status' && update.data.agent_update) {
      const agentUpdate = update.data.agent_update
      setAgentStatuses(prev => ({
        ...prev,
        [agentUpdate.agent_name]: agentUpdate
      }))
      
      if (update.data.workflow_progress) {
        setWorkflowProgress(update.data.workflow_progress)
      }
    }
    
    else if (update.event_type === 'workflow_complete') {
      setIsWorkflowActive(false)
      setWorkflowProgress(prev => prev ? {
        ...prev,
        overall_progress: 100,
        current_phase: 'Workflow completed'
      } : null)
      
      // Clear after delay
      setTimeout(() => {
        setCurrentWorkflow(null)
        setAgentStatuses({})
        setWorkflowProgress(null)
      }, 5000)
    }

    // Add to recent updates (keep last 10)
    setRecentUpdates(prev => [update, ...prev].slice(0, 10))
  }

  // Status indicator component
  const StatusIndicator: React.FC<{ status: string }> = ({ status }) => {
    switch (status) {
      case 'waiting':
        return <div className="w-3 h-3 rounded-full bg-gray-500 animate-pulse" />
      case 'active':
        return <Loader2 className="w-3 h-3 text-blue-400 animate-spin" />
      case 'completed':
        return <CheckCircle className="w-3 h-3 text-green-400" />
      case 'error':
        return <XCircle className="w-3 h-3 text-red-400" />
      case 'skipped':
        return <div className="w-3 h-3 rounded-full bg-yellow-500" />
      default:
        return <div className="w-3 h-3 rounded-full bg-gray-400" />
    }
  }

  // Agent card component
  const AgentCard: React.FC<{ agentName: string; status?: AgentStatus }> = ({ agentName, status }) => {
    const config = agentConfig[agentName as keyof typeof agentConfig]
    if (!config) return null

    const Icon = config.icon
    const isActive = status?.status === 'active'
    const isCompleted = status?.status === 'completed'
    const hasError = status?.status === 'error'

    return (
      <div className={`
        relative p-3 rounded-lg border transition-all duration-300
        ${isActive ? 'border-blue-400 bg-blue-900/20 shadow-lg shadow-blue-400/20' : 
          isCompleted ? 'border-green-400 bg-green-900/20' :
          hasError ? 'border-red-400 bg-red-900/20' :
          'border-gray-700 bg-gray-800/50'}
      `}>
        <div className="flex items-center space-x-3">
          <div className="relative">
            <Icon className={`w-5 h-5 ${config.color} ${isActive ? 'animate-pulse' : ''}`} />
            <div className="absolute -top-1 -right-1">
              <StatusIndicator status={status?.status || 'waiting'} />
            </div>
          </div>
          <div className="flex-1 min-w-0">
            <h4 className="text-sm font-medium text-white truncate">{config.label}</h4>
            <p className="text-xs text-gray-400 truncate">{config.description}</p>
          </div>
        </div>
        
        {status && (
          <div className="mt-2">
            {/* Progress bar */}
            <div className="w-full bg-gray-700 rounded-full h-1.5 mb-2">
              <div 
                className={`h-1.5 rounded-full transition-all duration-300 ${
                  hasError ? 'bg-red-400' : isCompleted ? 'bg-green-400' : 'bg-blue-400'
                }`}
                style={{ width: `${status.progress_percentage}%` }}
              />
            </div>
            
            {/* Current activity */}
            <p className="text-xs text-gray-300 truncate">{status.current_activity}</p>
            
            {/* Result preview */}
            {status.result_preview && (
              <p className="text-xs text-gray-400 mt-1 truncate">{status.result_preview}</p>
            )}
            
            {/* Error message */}
            {status.error_message && (
              <p className="text-xs text-red-400 mt-1 truncate">{status.error_message}</p>
            )}
            
            {/* Execution time */}
            {status.execution_time && (
              <div className="flex items-center space-x-1 mt-1">
                <Clock className="w-3 h-3 text-gray-400" />
                <span className="text-xs text-gray-400">{status.execution_time.toFixed(2)}s</span>
              </div>
            )}
          </div>
        )}
      </div>
    )
  }

  // Expose the handler for parent components
  React.useImperativeHandle(React.useRef(), () => ({
    handleStreamUpdate
  }))

  if (!isVisible) {
    return (
      <div className={`${className}`}>
        <button
          onClick={onToggle}
          className="w-full bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-lg p-3 transition-colors"
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Activity className="w-4 h-4 text-blue-400" />
              <span className="text-sm font-medium text-gray-300">Agent Pipeline</span>
              {isWorkflowActive && (
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" />
              )}
            </div>
            <ChevronDown className="w-4 h-4 text-gray-400" />
          </div>
        </button>
      </div>
    )
  }

  return (
    <div className={`${className} bg-gray-900 border border-gray-800 rounded-lg`}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-800">
        <div className="flex items-center space-x-2">
          <Activity className="w-4 h-4 text-blue-400" />
          <h3 className="text-sm font-medium text-gray-300">Agent Pipeline</h3>
          {isWorkflowActive && (
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" />
              <span className="text-xs text-blue-400">Active</span>
              {parallelExecution.isActive && (
                <div className="flex items-center space-x-1">
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                  <span className="text-xs text-green-400">Parallel</span>
                </div>
              )}
            </div>
          )}
        </div>
        <button
          onClick={onToggle}
          className="text-gray-400 hover:text-gray-300 transition-colors"
        >
          <ChevronUp className="w-4 h-4" />
        </button>
      </div>

      {/* Content */}
      <div className="p-4">
        {/* Parallel execution indicator */}
        {parallelExecution.isActive && (
          <div className="mb-4 p-3 bg-green-900/20 border border-green-400/30 rounded-lg">
            <div className="flex items-center space-x-2 mb-2">
              <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse" />
              <span className="text-sm font-medium text-green-400">Parallel Execution</span>
            </div>
            <div className="text-xs text-green-300">
              Phase: {parallelExecution.phase}
            </div>
            <div className="text-xs text-green-300">
              Concurrent: {parallelExecution.concurrentAgents.join(', ')}
            </div>
          </div>
        )}
        
        {/* Speedup indicator */}
        {parallelExecution.speedupFactor && (
          <div className="mb-4 p-3 bg-purple-900/20 border border-purple-400/30 rounded-lg">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-purple-400">Performance Boost</span>
              <span className="text-lg font-bold text-purple-300">
                {parallelExecution.speedupFactor.toFixed(1)}x
              </span>
            </div>
            <div className="text-xs text-purple-300">Faster than sequential execution</div>
          </div>
        )}

        {/* Overall progress */}
        {workflowProgress && (
          <div className="mb-4">
            <div className="flex items-center justify-between text-sm text-gray-300 mb-2">
              <span>Overall Progress</span>
              <span>{workflowProgress.overall_progress.toFixed(0)}%</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div 
                className="bg-blue-400 h-2 rounded-full transition-all duration-300"
                style={{ width: `${workflowProgress.overall_progress}%` }}
              />
            </div>
            <p className="text-xs text-gray-400 mt-1">{workflowProgress.current_phase}</p>
            <p className="text-xs text-gray-500">
              {workflowProgress.completed_agents} of {workflowProgress.total_agents} agents completed
            </p>
          </div>
        )}

        {/* Agent cards */}
        <div className="space-y-3">
          {Object.keys(agentConfig).map(agentName => {
            const isParallel = parallelExecution.concurrentAgents.includes(agentName)
            return (
              <div key={agentName} className={isParallel ? 'relative' : ''}>
                {isParallel && (
                  <div className="absolute -top-1 -right-1 z-10">
                    <div className="w-4 h-4 bg-green-400 rounded-full flex items-center justify-center">
                      <div className="w-2 h-2 bg-white rounded-full" />
                    </div>
                  </div>
                )}
                <AgentCard 
                  agentName={agentName} 
                  status={agentStatuses[agentName]} 
                />
              </div>
            )
          })}
        </div>

        {/* Workflow timing */}
        {workflowStartTime && (
          <div className="mt-4 pt-3 border-t border-gray-800">
            <div className="flex items-center justify-between text-xs text-gray-400">
              <span>Started</span>
              <span>{workflowStartTime.toLocaleTimeString()}</span>
            </div>
            {!isWorkflowActive && workflowProgress?.overall_progress === 100 && (
              <div className="flex items-center justify-between text-xs text-gray-400 mt-1">
                <span>Duration</span>
                <span>{((Date.now() - workflowStartTime.getTime()) / 1000).toFixed(1)}s</span>
              </div>
            )}
          </div>
        )}

        {/* No active workflow state */}
        {!currentWorkflow && !isWorkflowActive && (
          <div className="text-center py-6">
            <Activity className="w-8 h-8 text-gray-600 mx-auto mb-2" />
            <p className="text-sm text-gray-400">No active workflow</p>
            <p className="text-xs text-gray-500">Agent pipeline will appear here during processing</p>
          </div>
        )}
      </div>
    </div>
  )
}

export default AgentStreamPanel