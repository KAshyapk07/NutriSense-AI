const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('desktopSystem', {
  getLanIp: () => ipcRenderer.invoke('system:get-lan-ip'),
})

contextBridge.exposeInMainWorld('desktopAuth', {
  openSystemBrowser: (url) => ipcRenderer.invoke('auth:open-system-browser', url),
  storeRefreshToken: (token) => ipcRenderer.invoke('auth:store-refresh-token', token),
  getRefreshToken: () => ipcRenderer.invoke('auth:get-refresh-token'),
  clearRefreshToken: () => ipcRenderer.invoke('auth:clear-refresh-token'),
  onAuthSuccess: (callback) => {
    const listener = (_event, payload) => callback(payload)
    ipcRenderer.on('on-auth-success', listener)
    return () => ipcRenderer.removeListener('on-auth-success', listener)
  },
})
