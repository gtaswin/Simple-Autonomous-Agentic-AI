'use client'

import React from 'react'
import { motion } from 'framer-motion'
import { cn } from '@/utils/cn'

interface GlassPanelProps {
  children: React.ReactNode
  className?: string
  variant?: 'default' | 'elevated' | 'subtle' | 'bordered'
  animate?: boolean
  hover?: boolean
  onClick?: () => void
}

export function GlassPanel({
  children,
  className = '',
  variant = 'default',
  animate = true,
  hover = false,
  onClick
}: GlassPanelProps) {
  
  const getVariantStyles = (variant: string) => {
    const styles = {
      default: 'bg-gray-800/50 backdrop-blur-sm border border-gray-700/50',
      elevated: 'bg-gray-800/70 backdrop-blur-md border border-gray-600/50 shadow-xl',
      subtle: 'bg-gray-900/30 backdrop-blur-sm border border-gray-800/30',
      bordered: 'bg-gray-800/40 backdrop-blur-sm border-2 border-gray-600/30'
    }
    return styles[variant as keyof typeof styles] || styles.default
  }

  const Component = animate ? motion.div : 'div'
  
  const motionProps = animate ? {
    initial: { opacity: 0, scale: 0.95 },
    animate: { opacity: 1, scale: 1 },
    transition: { duration: 0.3, ease: "easeOut" },
    ...(hover ? {
      whileHover: { scale: 1.02, y: -2 },
      transition: { duration: 0.2 }
    } : {})
  } : {}

  return (
    <Component
      className={cn(
        'rounded-xl overflow-hidden',
        getVariantStyles(variant),
        hover && 'transition-all duration-200 hover:border-gray-500/50 cursor-pointer',
        onClick && 'cursor-pointer',
        className
      )}
      onClick={onClick}
      {...motionProps}
    >
      {children}
    </Component>
  )
}

export default GlassPanel