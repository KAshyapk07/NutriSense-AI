import { useState, useEffect, useRef } from 'react'
import { Navigate, Routes, Route, useLocation } from 'react-router-dom'
import { AnimatePresence } from 'framer-motion'
import { useAuth } from './hooks/use-auth'
import LoadingScreen from './pages/loading'
import Home from './pages/home'
import Results from './pages/results'
import SearchPage from './pages/search'
import ProductChatPage from './pages/product-chat'
import ImagePage from './pages/image'
import ComparePage from './pages/compare'
import ModifyPage from './pages/modify'
import HealthySwapsPage from './pages/healthy-swaps'
import ChefPage from './pages/chef'
import ChefRemotePage from './pages/chef-remote'
import KitchenPage from './pages/kitchen'
import Profile from './pages/profile'
import Settings from './pages/settings'
import LoginPage from './pages/login'
import RegisterPage from './pages/register'
import { Sidebar } from './components/layout/sidebar'

const AUTH_ROUTES = ['/login', '/register']

function App() {
  const { isAuthenticated, loading } = useAuth()
  const location = useLocation()

  const [minSplashDone, setMinSplashDone] = useState(false)
  const [postAuthSplash, setPostAuthSplash] = useState(false)
  const prevAuth = useRef(isAuthenticated)

  useEffect(() => {
    const timer = setTimeout(() => setMinSplashDone(true), 3000)
    return () => clearTimeout(timer)
  }, [])

  useEffect(() => {
    if (!prevAuth.current && isAuthenticated) {
      setPostAuthSplash(true)
      const timer = setTimeout(() => setPostAuthSplash(false), 3000)
      prevAuth.current = isAuthenticated
      return () => clearTimeout(timer)
    }
    prevAuth.current = isAuthenticated
  }, [isAuthenticated])

  const path = location.pathname.replace(/\/+$/, '') || '/'

  // ── Chef Remote: completely separate flow, no auth ──
  // Phone scans QR → splash for 3s → chef-remote UI. Zero auth dependency.
  if (path.startsWith('/chef-remote')) {
    return (
      <>
        <AnimatePresence>
          {!minSplashDone && <LoadingScreen />}
        </AnimatePresence>
        <ChefRemotePage />
      </>
    )
  }

  // ── PC / main app flow ──
  const isAuthPage = AUTH_ROUTES.includes(path)
  const authReady = !loading
  const showSplash = !isAuthPage && (!minSplashDone || !authReady || postAuthSplash)

  if (!showSplash && !isAuthenticated && !isAuthPage) {
    return <Navigate to="/login" replace state={{ from: location }} />
  }

  if (!showSplash && isAuthenticated && isAuthPage) {
    return <Navigate to="/" replace />
  }

  return (
    <>
      <AnimatePresence>
        {showSplash && <LoadingScreen />}
      </AnimatePresence>

      {!isAuthPage && <Sidebar />}

      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />
        <Route path="/results" element={<Results />} />
        <Route path="/search" element={<SearchPage />} />
        <Route path="/product-chat" element={<ProductChatPage />} />
        <Route path="/image" element={<ImagePage />} />
        <Route path="/compare" element={<ComparePage />} />
        <Route path="/modify" element={<ModifyPage />} />
        <Route path="/healthy-swaps" element={<HealthySwapsPage />} />
        <Route path="/chef" element={<ChefPage />} />
        <Route path="/chef-remote" element={<ChefRemotePage />} />
        <Route path="/kitchen" element={<KitchenPage />} />
        <Route path="/profile" element={<Profile />} />
        <Route path="/settings" element={<Settings />} />
      </Routes>
    </>
  )
}

export default App
