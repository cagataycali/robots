import { useCallback, useEffect, useRef, useState } from 'react'
import { useRegisterSW } from 'virtual:pwa-register/react'

/**
 * PWA plumbing: install prompt, update prompt, online state, screen wake lock.
 *
 * Updates are `prompt`, not `autoUpdate`, on purpose. Auto-update reloads the
 * page whenever a new bundle ships - including mid-task, which would tear down
 * the camera sockets and the run form of a robot that is currently moving. The
 * operator decides when to reload.
 */
export function usePwa() {
  const {
    needRefresh: [needRefresh, setNeedRefresh],
    updateServiceWorker,
  } = useRegisterSW({ immediate: true })

  const [online, setOnline] = useState(navigator.onLine)
  const [installable, setInstallable] = useState(false)
  const promptRef = useRef<any>(null)
  const wakeRef = useRef<any>(null)

  useEffect(() => {
    const up = () => setOnline(true)
    const down = () => setOnline(false)
    window.addEventListener('online', up)
    window.addEventListener('offline', down)

    const onPrompt = (e: Event) => {
      e.preventDefault()          // keep our own chip instead of the mini-infobar
      promptRef.current = e
      setInstallable(true)
    }
    window.addEventListener('beforeinstallprompt', onPrompt)
    const onInstalled = () => { setInstallable(false); promptRef.current = null }
    window.addEventListener('appinstalled', onInstalled)

    return () => {
      window.removeEventListener('online', up)
      window.removeEventListener('offline', down)
      window.removeEventListener('beforeinstallprompt', onPrompt)
      window.removeEventListener('appinstalled', onInstalled)
    }
  }, [])

  const install = useCallback(async () => {
    const prompt = promptRef.current
    if (!prompt) return
    promptRef.current = null
    setInstallable(false)
    try { await prompt.prompt() } catch { /* user dismissed */ }
  }, [])

  const update = useCallback(() => {
    setNeedRefresh(false)
    updateServiceWorker(true)
  }, [setNeedRefresh, updateServiceWorker])

  /**
   * Hold the screen awake while any robot is running. A phone that sleeps
   * mid-task drops the camera sockets and the operator loses sight of a moving
   * arm - the one moment the screen must stay on.
   */
  const keepAwake = useCallback(async (want: boolean) => {
    const anyNav = navigator as any
    if (!anyNav.wakeLock) return
    if (want && !wakeRef.current) {
      try {
        wakeRef.current = await anyNav.wakeLock.request('screen')
        wakeRef.current.addEventListener?.('release', () => { wakeRef.current = null })
      } catch { /* denied or not visible */ }
    } else if (!want && wakeRef.current) {
      try { await wakeRef.current.release() } catch { /* already gone */ }
      wakeRef.current = null
    }
  }, [])

  const standalone = window.matchMedia('(display-mode: standalone)').matches
    || (navigator as any).standalone === true

  return { online, needRefresh, update, installable, install, keepAwake, standalone }
}
