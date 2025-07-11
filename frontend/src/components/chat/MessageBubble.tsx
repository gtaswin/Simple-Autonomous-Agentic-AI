'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { User, Bot, Copy, Check, Brain, Clock } from 'lucide-react'
import ThoughtBubble from '../thinking/ThoughtBubble'

interface MessageBubbleProps {
  message: string
  sender: 'user' | 'ai'
  timestamp?: string
  thinking?: string[]
  hasThinking?: boolean
  agent?: string
  confidence?: number
  modelUsed?: string
  className?: string
}

export function MessageBubble({
  message,
  sender,
  timestamp,
  thinking = [],
  hasThinking = false,
  agent = 'AI',
  confidence,
  modelUsed,
  className = ''
}: MessageBubbleProps) {
  const [copied, setCopied] = useState(false)

  const isUser = sender === 'user'

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(message)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (error) {
      console.error('Failed to copy message:', error)
    }
  }

  const formatTimestamp = (ts?: string) => {
    if (!ts) return ''
    try {
      return new Date(ts).toLocaleTimeString([], { 
        hour: '2-digit', 
        minute: '2-digit' 
      })
    } catch {
      return ''
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className={`message-bubble flex ${isUser ? 'justify-end' : 'justify-start'} ${className}`}
    >
      <div className={`max-w-[85%] ${isUser ? 'order-2' : 'order-1'}`}>
        
        {/* AI Thinking (shown before AI response) */}
        {!isUser && hasThinking && thinking.length > 0 && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2, duration: 0.3 }}
            className="mb-3"
          >
            <ThoughtBubble
              thinking={thinking}
              agent={agent}
              timestamp={timestamp}
              confidence={confidence}
              isExpanded={false}
            />
          </motion.div>
        )}

        {/* Message Container */}
        <div className={`flex items-start gap-3 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
          
          {/* Avatar */}
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.1, type: "spring", stiffness: 260, damping: 20 }}
            className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center shadow-lg ${
              isUser 
                ? 'bg-gradient-to-br from-blue-500 to-cyan-500' 
                : 'bg-gradient-to-br from-purple-500 to-indigo-500'
            }`}
          >
            {isUser ? (
              <User className="w-5 h-5 text-white" />
            ) : (
              <Bot className="w-5 h-5 text-white" />
            )}
          </motion.div>

          {/* Message Content */}
          <div className={`flex-1 ${isUser ? 'text-right' : 'text-left'}`}>
            
            {/* Message Bubble */}
            <motion.div
              className={`inline-block max-w-full p-4 rounded-2xl shadow-lg backdrop-blur-sm border group relative ${
                isUser
                  ? 'bg-gradient-to-br from-blue-600 to-cyan-600 text-white border-blue-500/30'
                  : 'bg-gray-800/90 text-gray-100 border-gray-700/50 hover:bg-gray-700/90'
              } transition-colors duration-200`}
              whileHover={{ scale: 1.01 }}
              transition={{ duration: 0.2 }}
            >
              {/* Message Text */}
              <div className="prose prose-invert max-w-none">
                <p className="whitespace-pre-wrap leading-relaxed text-sm">
                  {message}
                </p>
              </div>

              {/* Copy Button */}
              <motion.button
                onClick={copyToClipboard}
                className={`absolute top-2 ${isUser ? 'left-2' : 'right-2'} opacity-0 group-hover:opacity-100 p-1.5 rounded-lg transition-all duration-200 ${
                  isUser 
                    ? 'bg-blue-500/20 hover:bg-blue-400/30' 
                    : 'bg-gray-700/50 hover:bg-gray-600/70'
                }`}
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
              >
                {copied ? (
                  <Check className="w-3 h-3 text-green-400" />
                ) : (
                  <Copy className="w-3 h-3 text-gray-300" />
                )}
              </motion.button>
            </motion.div>

            {/* Message Metadata */}
            <div className={`flex items-center gap-3 mt-2 text-xs text-gray-500 ${isUser ? 'justify-end' : 'justify-start'}`}>
              
              {/* Timestamp */}
              {timestamp && (
                <div className="flex items-center gap-1">
                  <Clock className="w-3 h-3" />
                  <span>{formatTimestamp(timestamp)}</span>
                </div>
              )}

              {/* AI Metadata */}
              {!isUser && (
                <>
                  {modelUsed && (
                    <div className="bg-gray-800/50 px-2 py-1 rounded text-xs">
                      {modelUsed}
                    </div>
                  )}
                  
                  {confidence !== undefined && (
                    <div className="flex items-center gap-1">
                      <Brain className="w-3 h-3" />
                      <span className={`font-medium ${
                        confidence >= 0.8 ? 'text-green-400' :
                        confidence >= 0.6 ? 'text-yellow-400' : 'text-red-400'
                      }`}>
                        {Math.round(confidence * 100)}%
                      </span>
                    </div>
                  )}

                  {hasThinking && thinking.length > 0 && (
                    <div className="flex items-center gap-1 text-purple-400">
                      <Brain className="w-3 h-3" />
                      <span>{thinking.length} thought{thinking.length !== 1 ? 's' : ''}</span>
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  )
}

export default MessageBubble