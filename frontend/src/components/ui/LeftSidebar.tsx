'use client'

import { Activity, Brain, Network, Sparkles, Trash2, ChevronDown, ChevronUp } from 'lucide-react'
import { useState } from 'react'

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

interface LeftSidebarProps {
  systemStatus: any
  connectionStatus: 'connected' | 'disconnected' | 'connecting'
  lastUpdated: number
  onClearMemory?: () => void
  isClearing?: boolean
  autonomousInsights?: AutonomousInsights | null
  onLoadAutonomousInsights?: () => void
  onClearAutonomousInsights?: () => void
  onGenerateInsights?: () => void
  isLoadingInsights?: boolean
}

const LeftSidebar: React.FC<LeftSidebarProps> = ({
  systemStatus,
  connectionStatus,
  lastUpdated,
  onClearMemory,
  isClearing = false,
  autonomousInsights,
  onLoadAutonomousInsights,
  onClearAutonomousInsights,
  onGenerateInsights,
  isLoadingInsights = false
}) => {
  const [expandedInsights, setExpandedInsights] = useState<Set<string>>(new Set())
  
  const toggleInsightExpansion = (insightId: string) => {
    const newExpanded = new Set(expandedInsights)
    if (newExpanded.has(insightId)) {
      newExpanded.delete(insightId)
    } else {
      newExpanded.add(insightId)
    }
    setExpandedInsights(newExpanded)
  }
  return (
    <div className="w-80 bg-gray-900 border-r border-gray-800 flex flex-col">
      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        
        {/* System Status */}
        <div className="space-y-3">
          <div className="flex items-center space-x-2">
            <Activity className="w-4 h-4 text-green-400" />
            <h3 className="text-sm font-medium text-gray-300">System Status</h3>
          </div>
          <div className="bg-gray-800 rounded-lg p-3 space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-gray-400">Health:</span>
              <span className={`${systemStatus?.system_health === 'healthy' ? 'text-green-400' : 'text-red-400'}`}>
                {systemStatus?.system_health || 'Unknown'}
              </span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-gray-400">Agents:</span>
              <span className="text-white">
                {systemStatus?.agents ? Object.keys(systemStatus.agents).length : 0} Active
              </span>
            </div>
          </div>
        </div>

        {/* AI Insights */}
        <div className="space-y-3">
          <div className="flex items-center space-x-2">
            <Sparkles className="w-4 h-4 text-blue-400" />
            <h3 className="text-sm font-medium text-gray-300">AI Insights</h3>
          </div>
          <div className="bg-gray-800 rounded-lg p-3">
            {isLoadingInsights ? (
              <div className="text-center py-4">
                <div className="animate-spin w-5 h-5 border-2 border-blue-400 border-t-transparent rounded-full mx-auto mb-2"></div>
                <p className="text-xs text-gray-400">Loading insights...</p>
              </div>
            ) : autonomousInsights && autonomousInsights.insights.length > 0 ? (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-400">
                    {autonomousInsights.total_insights} insights for {autonomousInsights.user_name}
                  </span>
                  <button
                    onClick={onLoadAutonomousInsights}
                    className="text-xs text-blue-400 hover:text-blue-300"
                  >
                    Refresh
                  </button>
                </div>
                <div className="max-h-96 overflow-y-auto space-y-2">
                  {autonomousInsights.insights.map((insight, index) => {
                    const isExpanded = expandedInsights.has(insight.insight_id)
                    const shouldTruncate = insight.content.length > 300
                    const displayContent = isExpanded ? insight.content : insight.content.substring(0, 300)
                    
                    return (
                      <div key={insight.insight_id} className="p-3 bg-gray-700 rounded-lg text-sm">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-blue-300 font-medium text-sm">
                            {insight.insight_type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                          </span>
                          <span className="text-gray-500 text-xs">
                            {new Date(insight.created_at).toLocaleDateString()}
                          </span>
                        </div>
                        <p className="text-gray-300 leading-relaxed whitespace-pre-wrap">
                          {displayContent}
                          {!isExpanded && shouldTruncate && '...'}
                        </p>
                        {shouldTruncate && (
                          <button
                            onClick={() => toggleInsightExpansion(insight.insight_id)}
                            className="mt-2 text-xs text-blue-400 hover:text-blue-300 flex items-center space-x-1"
                          >
                            {isExpanded ? (
                              <>
                                <ChevronUp className="w-3 h-3" />
                                <span>Show less</span>
                              </>
                            ) : (
                              <>
                                <ChevronDown className="w-3 h-3" />
                                <span>Read more</span>
                              </>
                            )}
                          </button>
                        )}
                      </div>
                    )
                  })}
                </div>
                <div className="space-y-2">
                  <button
                    onClick={onGenerateInsights}
                    className="w-full bg-blue-600 hover:bg-blue-700 text-white text-sm py-2 px-3 rounded transition-colors flex items-center justify-center space-x-2"
                  >
                    <Sparkles className="w-4 h-4" />
                    <span>Generate Insights</span>
                  </button>
                  <button
                    onClick={onClearAutonomousInsights}
                    className="w-full bg-red-600 hover:bg-red-700 text-white text-sm py-2 px-3 rounded transition-colors flex items-center justify-center space-x-2"
                  >
                    <Trash2 className="w-4 h-4" />
                    <span>Clear Insights</span>
                  </button>
                </div>
              </div>
            ) : (
              <div className="text-center py-4">
                <p className="text-sm text-gray-400 mb-3">No AI insights yet</p>
                <div className="space-y-2">
                  <button
                    onClick={onGenerateInsights}
                    className="w-full bg-blue-600 hover:bg-blue-700 text-white text-sm py-2 px-3 rounded transition-colors flex items-center justify-center space-x-2"
                  >
                    <Sparkles className="w-4 h-4" />
                    <span>Generate Insights</span>
                  </button>
                  <button
                    onClick={onLoadAutonomousInsights}
                    className="text-sm text-blue-400 hover:text-blue-300"
                  >
                    Load Insights
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Connection Status */}
        <div className="space-y-3">
          <div className="flex items-center space-x-2">
            <Network className="w-4 h-4 text-green-400" />
            <h3 className="text-sm font-medium text-gray-300">Connection</h3>
          </div>
          <div className="bg-gray-800 rounded-lg p-3 space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-gray-400">WebSocket:</span>
              <span className={`${
                connectionStatus === 'connected' ? 'text-green-400' : 
                connectionStatus === 'connecting' ? 'text-yellow-400' : 'text-red-400'
              }`}>
                {connectionStatus}
              </span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-gray-400">Last Update:</span>
              <span className="text-white">
                {lastUpdated ? new Date(lastUpdated).toLocaleTimeString() : 'Never'}
              </span>
            </div>
          </div>
        </div>

        {/* Memory Controls */}
        <div className="space-y-3">
          <div className="flex items-center space-x-2">
            <Trash2 className="w-4 h-4 text-red-400" />
            <h3 className="text-sm font-medium text-gray-300">Memory Controls</h3>
          </div>
          <div className="bg-gray-800 rounded-lg p-3">
            <button
              onClick={onClearMemory}
              disabled={isClearing}
              className="w-full bg-red-600 hover:bg-red-700 disabled:bg-red-400 disabled:cursor-not-allowed text-white text-sm py-2 px-3 rounded transition-colors flex items-center justify-center space-x-2"
            >
              <Trash2 className={`w-4 h-4 ${isClearing ? 'animate-spin' : ''}`} />
              <span>{isClearing ? 'Clearing...' : 'Clear Memory'}</span>
            </button>
            <p className="text-xs text-gray-400 mt-2 text-center">
              Clears session chat & working memory
            </p>
          </div>
        </div>

      </div>

      {/* Footer */}
      <div className="p-4 border-t border-gray-800">
        <div className="text-xs text-gray-500 text-center">
          Autonomous AI System v2.0
        </div>
      </div>
    </div>
  )
}

export default LeftSidebar