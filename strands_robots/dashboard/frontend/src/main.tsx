import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import { initScheme } from './lib/scheme'
import './styles.css'

// The remembered colour scheme goes on <html> before React paints, so a dark
// choice never flashes the paper page first.
initScheme()

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode><App /></React.StrictMode>,
)
