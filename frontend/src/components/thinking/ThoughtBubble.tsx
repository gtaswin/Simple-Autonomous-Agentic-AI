'use client'

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Brain, ChevronDown, ChevronUp, Clock, Zap } from 'lucide-react'

interface ThoughtBubbleProps {
  thinking: string[]
  agent?: string
  timestamp?: string
  confidence?: number
  isExpanded?: boolean
  className?: string
}

export function ThoughtBubble({ 
  thinking, 
  agent = 'AI', 
  timestamp, 
  confidence, 
  isExpanded = false,
  className = '' 
}: ThoughtBubbleProps) {
  const [expanded, setExpanded] = useState(isExpanded)

  if (!thinking || thinking.length === 0) return null

  const getAgentColor = (agentName: string) => {
    const colors = {
      'memory_agent': 'from-blue-500 to-cyan-500',
      'research_agent': 'from-green-500 to-emerald-500', 
      'thinking_agent': 'from-purple-500 to-violet-500',
      'coordinator_agent': 'from-orange-500 to-amber-500',
      'AI': 'from-indigo-500 to-purple-500'
    }
    return colors[agentName as keyof typeof colors] || colors.AI
  }

  const getConfidenceColor = (conf: number) => {
    if (conf >= 0.8) return 'text-green-400'
    if (conf >= 0.6) return 'text-yellow-400'
    return 'text-red-400'
  }

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95, y: 10 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.95, y: -10 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className={`thinking-bubble ${className}`}
    >
      {/* Thinking Header */}
      <motion.div
        className="flex items-center gap-3 p-3 bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-t-xl cursor-pointer hover:bg-gray-700/50 transition-colors"
        onClick={() => setExpanded(!expanded)}
        whileHover={{ scale: 1.01 }}
        whileTap={{ scale: 0.99 }}
      >
        {/* Agent Avatar */}
        <div className={`w-8 h-8 rounded-full bg-gradient-to-br ${getAgentColor(agent)} flex items-center justify-center shadow-lg`}>
          <Brain className="w-4 h-4 text-white" />
        </div>

        {/* Header Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-gray-200">{agent} thinking</span>
            {confidence !== undefined && (
              <div className="flex items-center gap-1">
                <Zap className="w-3 h-3 text-yellow-400" />
                <span className={`text-xs font-medium ${getConfidenceColor(confidence)}`}>
                  {Math.round(confidence * 100)}%
                </span>
              </div>
            )}
          </div>
          
          {timestamp && (
            <div className="flex items-center gap-1 mt-1">
              <Clock className="w-3 h-3 text-gray-500" />
              <span className="text-xs text-gray-500">
                {new Date(timestamp).toLocaleTimeString()}
              </span>
            </div>
          )}
        </div>

        {/* Expand/Collapse Icon */}
        <motion.div
          animate={{ rotate: expanded ? 180 : 0 }}
          transition={{ duration: 0.2 }}
        >
          <ChevronDown className="w-4 h-4 text-gray-400" />
        </motion.div>

        {/* Thinking Count Badge */}
        <div className="bg-gray-700 text-gray-300 text-xs px-2 py-1 rounded-full">
          {thinking.length} thought{thinking.length !== 1 ? 's' : ''}
        </div>
      </motion.div>

      {/* Thinking Content */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: "easeOut" }}
            className="overflow-hidden"
          >
            <div className="p-4 bg-gray-900/50 backdrop-blur-sm border-x border-b border-gray-700/50 rounded-b-xl">
              <div className="space-y-3">
                {thinking.map((thought, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1, duration: 0.4 }}
                    className="group"
                  >
                    {/* Thought Step */}
                    <div className="flex gap-3">
                      <div className="flex flex-col items-center">
                        <div className="w-6 h-6 rounded-full bg-gradient-to-br from-purple-500 to-blue-500 flex items-center justify-center text-xs font-bold text-white shadow-lg">
                          {index + 1}
                        </div>
                        {index < thinking.length - 1 && (
                          <div className="w-0.5 h-4 bg-gradient-to-b from-purple-500 to-blue-500 mt-1 opacity-50" />
                        )}
                      </div>
                      
                      <div className="flex-1 min-w-0">
                        <motion.div
                          className="bg-gray-800/30 border border-gray-700/30 rounded-lg p-3 group-hover:bg-gray-700/30 transition-colors"
                          whileHover={{ x: 4 }}
                          transition={{ duration: 0.2 }}
                        >
                          <p className="text-sm text-gray-200 leading-relaxed whitespace-pre-wrap">
                            {thought}
                          </p>
                        </motion.div>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>

              {/* Thinking Footer */}
              <div className="mt-4 pt-3 border-t border-gray-700/30">
                <div className="flex items-center justify-between text-xs text-gray-500">
                  <span>Reasoning complete</span>
                  <span>{thinking.length} step{thinking.length !== 1 ? 's' : ''} analyzed</span>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Thinking Tail (Speech Bubble Style) */}
      <motion.div
        className="relative"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2 }}
      >
        <svg
          className="absolute -top-1 left-8 w-4 h-3 text-gray-800 fill-current"
          viewBox="0 0 16 12"
        >
          <path d="M0 0 Q8 0 8 12 Q8 0 16 0 Z" />
        </svg>
      </motion.div>
    </motion.div>
  )
}

export default ThoughtBubble