import { Routes, Route } from 'react-router-dom'
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
import Profile from './pages/profile'
import Settings from './pages/settings'
import { Sidebar } from './components/layout/sidebar'

function App() {
  return (
    <>
      {/* Global sidebar — rendered above all page content */}
      <Sidebar />

      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/results" element={<Results />} />
        <Route path="/search" element={<SearchPage />} />
        <Route path="/product-chat" element={<ProductChatPage />} />
        <Route path="/image" element={<ImagePage />} />
        <Route path="/compare" element={<ComparePage />} />
        <Route path="/modify" element={<ModifyPage />} />
        <Route path="/healthy-swaps" element={<HealthySwapsPage />} />
        <Route path="/chef" element={<ChefPage />} />
        <Route path="/chef-remote" element={<ChefRemotePage />} />
        <Route path="/profile" element={<Profile />} />
        <Route path="/settings" element={<Settings />} />
      </Routes>
    </>
  )
}

export default App

