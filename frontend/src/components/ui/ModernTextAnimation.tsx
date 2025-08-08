'use client'

import { useState, useEffect } from 'react'

interface ModernTextAnimationProps {
  text: string
  delay?: number
  className?: string
  onComplete?: () => void
}

const ModernTextAnimation: React.FC<ModernTextAnimationProps> = ({
  text,
  delay = 500,
  className = '',
  onComplete
}) => {
  const [isVisible, setIsVisible] = useState(false)

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsVisible(true)
      onComplete?.()
    }, delay)

    return () => clearTimeout(timer)
  }, [delay, onComplete])

  // Reset when text changes
  useEffect(() => {
    setIsVisible(false)
  }, [text])

  return (
    <h1 className={`text-3xl font-bold bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent transition-all duration-1000 ease-out ${
      isVisible ? 'opacity-100 transform-none' : 'opacity-0 translate-y-2'
    } ${className}`}>
      {text}
    </h1>
  )
}

export default ModernTextAnimation