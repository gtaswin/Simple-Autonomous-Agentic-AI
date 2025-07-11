'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'

export default function ChatPage() {
  const router = useRouter()
  
  useEffect(() => {
    // Redirect to main page for unified chat experience
    router.replace('/')
  }, [router])

  return (
    <div className="h-screen bg-gray-950 flex items-center justify-center">
      <div className="text-white text-center">
        <div className="animate-spin w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full mx-auto mb-4"></div>
        <p>Redirecting to main chat interface...</p>
      </div>
    </div>
  )
}