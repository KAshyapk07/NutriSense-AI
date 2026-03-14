import { Navigate, Routes, Route, useLocation } from 'react-router-dom'
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

  if (loading) return <LoadingScreen />

  const isAuthPage = AUTH_ROUTES.includes(location.pathname)

  if (!isAuthenticated && !isAuthPage) {
    return <Navigate to="/login" replace state={{ from: location }} />
  }

  if (isAuthenticated && isAuthPage) {
    return <Navigate to="/" replace />
  }

  return (
    <>
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
