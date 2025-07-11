'use client'

import React from 'react'
import { motion } from 'framer-motion'
import { Brain, Loader2, Zap } from 'lucide-react'

interface ThinkingIndicatorProps {
  agent?: string
  message?: string
  type?: 'typing' | 'thinking' | 'processing' | 'analyzing'
  animated?: boolean
  className?: string
}

export function ThinkingIndicator({
  agent = 'AI',
  message,
  type = 'thinking',
  animated = true,
  className = ''
}: ThinkingIndicatorProps) {

  const getTypeConfig = (type: string) => {
    const configs = {
      typing: {
        icon: Loader2,
        message: 'is typing...',
        color: 'text-blue-400',
        bgColor: 'from-blue-500/20 to-cyan-500/20',
        borderColor: 'border-blue-500/30'
      },
      thinking: {
        icon: Brain,
        message: 'is thinking...',
        color: 'text-purple-400',
        bgColor: 'from-purple-500/20 to-indigo-500/20',
        borderColor: 'border-purple-500/30'
      },
      processing: {
        icon: Zap,
        message: 'is processing...',
        color: 'text-yellow-400',
        bgColor: 'from-yellow-500/20 to-orange-500/20',
        borderColor: 'border-yellow-500/30'
      },
      analyzing: {
        icon: Brain,
        message: 'is analyzing...',
        color: 'text-green-400',
        bgColor: 'from-green-500/20 to-emerald-500/20',
        borderColor: 'border-green-500/30'
      }
    }
    return configs[type as keyof typeof configs] || configs.thinking
  }

  const config = getTypeConfig(type)
  const Icon = config.icon

  return (
    <motion.div
      initial={{ opacity: 0, y: 10, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -10, scale: 0.95 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className={`thinking-indicator ${className}`}
    >
      <div className="flex justify-start">
        <div className="max-w-md">
          
          {/* Indicator Container */}
          <motion.div
            className={`
              inline-flex items-center gap-3 p-4 rounded-2xl 
              bg-gradient-to-r ${config.bgColor} 
              backdrop-blur-sm border ${config.borderColor}
              shadow-lg
            `}
            animate={animated ? {
              scale: [1, 1.02, 1],
              opacity: [0.8, 1, 0.8]
            } : {}}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: "easeInOut"
            }}
          >
            
            {/* Animated Icon */}
            <motion.div
              animate={animated ? {
                rotate: type === 'typing' ? 360 : 0,
                scale: [1, 1.1, 1]
              } : {}}
              transition={{
                rotate: { duration: 2, repeat: Infinity, ease: "linear" },
                scale: { duration: 1.5, repeat: Infinity, ease: "easeInOut" }
              }}
              className={config.color}
            >
              <Icon className="w-5 h-5" />
            </motion.div>

            {/* Agent Name and Status */}
            <div className="flex flex-col">
              <span className="text-sm font-medium text-gray-200">
                {agent}
              </span>
              <span className={`text-xs ${config.color}`}>
                {message || config.message}
              </span>
            </div>

            {/* Animated Dots */}
            <div className="flex items-center gap-1">
              {[0, 1, 2].map((i) => (
                <motion.div
                  key={i}
                  className={`w-1.5 h-1.5 rounded-full bg-current ${config.color}`}
                  animate={animated ? {
                    opacity: [0.3, 1, 0.3],
                    scale: [0.8, 1.2, 0.8]
                  } : {}}
                  transition={{
                    duration: 1.5,
                    repeat: Infinity,
                    delay: i * 0.2,
                    ease: "easeInOut"
                  }}
                />
              ))}
            </div>
          </motion.div>

          {/* Thinking Pulse Effect */}
          {animated && (
            <motion.div
              className={`absolute inset-0 rounded-2xl bg-gradient-to-r ${config.bgColor} opacity-30`}
              animate={{
                scale: [1, 1.1, 1],
                opacity: [0, 0.3, 0]
              }}
              transition={{
                duration: 2,
                repeat: Infinity,
                ease: "easeInOut"
              }}
            />
          )}
        </div>
      </div>
    </motion.div>
  )
}

export default ThinkingIndicator