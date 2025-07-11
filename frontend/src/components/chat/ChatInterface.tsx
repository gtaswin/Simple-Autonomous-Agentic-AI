'use client'

import React from 'react'
import { cn } from '@/utils/cn'

interface ChatInterfaceProps {
  className?: string
  showThinkingStream?: boolean
  websocketUrl?: string
}

export function ChatInterface({
  className = '',
  showThinkingStream = true,
  websocketUrl
}: ChatInterfaceProps) {
  // This component is deprecated - redirected to main page
  return (
    <div className={cn("h-full bg-gray-950 flex items-center justify-center", className)}>
      <div className="text-white text-center">
        <div className="animate-spin w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full mx-auto mb-4"></div>
        <p>Redirecting to main chat interface...</p>
      </div>
    </div>
  )
}