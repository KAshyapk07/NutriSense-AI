import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App'
import { ThemeProvider } from './hooks/use-theme'
import { SidebarProvider } from './hooks/use-sidebar'
import { PreferencesProvider } from './hooks/use-preferences'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <ThemeProvider>
        <SidebarProvider>
          <PreferencesProvider>
            <App />
          </PreferencesProvider>
        </SidebarProvider>
      </ThemeProvider>
    </BrowserRouter>
  </React.StrictMode>,
)
