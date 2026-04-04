const { app, BrowserWindow, ipcMain, shell } = require('electron')
const path = require('path')
const os = require('os')
const keytar = require('keytar')

const APP_PROTOCOL = 'nutrisense'
const AUTH_CHANNEL = 'on-auth-success'
const CREDENTIAL_SERVICE = 'NutriSense-AI'
const CREDENTIAL_ACCOUNT = 'refresh-token'

let mainWindow = null

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    minWidth: 1024,
    minHeight: 640,
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  })

  const rendererUrl = process.env.ELECTRON_RENDERER_URL || 'http://127.0.0.1:5173'
  mainWindow.loadURL(rendererUrl)
}

function parseDeepLink(rawUrl) {
  try {
    const url = new URL(rawUrl)
    if (url.protocol !== `${APP_PROTOCOL}:`) return null

    const normalizedPath = `${url.hostname}${url.pathname}`.replace(/^\/+/, '')
    if (!normalizedPath.startsWith('auth')) return null

    const token = url.searchParams.get('token')
    if (!token) return null
    return token
  } catch {
    return null
  }
}

function emitAuthToken(token) {
  if (!mainWindow || !mainWindow.webContents) return
  mainWindow.webContents.send(AUTH_CHANNEL, { firebaseIdToken: token })
}

function handleDeepLink(rawUrl) {
  const token = parseDeepLink(rawUrl)
  if (!token) return
  if (mainWindow) {
    if (mainWindow.isMinimized()) mainWindow.restore()
    mainWindow.focus()
  }
  emitAuthToken(token)
}

function registerProtocol() {
  if (process.defaultApp) {
    app.setAsDefaultProtocolClient(APP_PROTOCOL, process.execPath, [process.argv[1]])
    return
  }
  app.setAsDefaultProtocolClient(APP_PROTOCOL)
}

const gotSingleInstanceLock = app.requestSingleInstanceLock()
if (!gotSingleInstanceLock) {
  app.quit()
} else {
  app.on('second-instance', (_event, argv) => {
    const deepLink = argv.find((arg) => arg.startsWith(`${APP_PROTOCOL}://`))
    if (deepLink) handleDeepLink(deepLink)
  })
}

app.whenReady().then(() => {
  registerProtocol()
  createWindow()

  if (process.platform === 'darwin') {
    app.on('open-url', (event, url) => {
      event.preventDefault()
      handleDeepLink(url)
    })
  } else {
    const deepLink = process.argv.find((arg) => arg.startsWith(`${APP_PROTOCOL}://`))
    if (deepLink) handleDeepLink(deepLink)
  }

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

ipcMain.handle('system:get-lan-ip', () => {
  const nets = os.networkInterfaces()
  for (const iface of Object.values(nets)) {
    for (const net of iface) {
      if (net.family === 'IPv4' && !net.internal) {
        return net.address
      }
    }
  }
  return null
})

ipcMain.handle('auth:open-system-browser', async (_event, url) => {
  if (typeof url !== 'string' || !url.startsWith('https://')) {
    throw new Error('Only HTTPS auth URLs are allowed.')
  }
  await shell.openExternal(url)
})

ipcMain.handle('auth:store-refresh-token', async (_event, refreshToken) => {
  if (typeof refreshToken !== 'string' || refreshToken.length < 10) {
    throw new Error('Invalid refresh token payload.')
  }
  await keytar.setPassword(CREDENTIAL_SERVICE, CREDENTIAL_ACCOUNT, refreshToken)
})

ipcMain.handle('auth:get-refresh-token', async () => {
  return keytar.getPassword(CREDENTIAL_SERVICE, CREDENTIAL_ACCOUNT)
})

ipcMain.handle('auth:clear-refresh-token', async () => {
  await keytar.deletePassword(CREDENTIAL_SERVICE, CREDENTIAL_ACCOUNT)
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})
