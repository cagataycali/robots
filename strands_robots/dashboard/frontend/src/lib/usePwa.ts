/** PWA plumbing: install prompt, update prompt, online state, screen wake lock. */
import { useCallback, useEffect, useRef, useState } from 'react'
import { useRegisterSW } from 'virtual:pwa-register/react'
import { shouldCheckForUpdate, SW_UPDATE_INTERVAL_MS, bundleAgeText } from './swUpdate'
import { wakeLockAction, wakeLockNote } from './wakeLock'

/**
 * PWA plumbing: install prompt, update prompt, online state, screen wake lock. Updates are
 * `prompt`, not `autoUpdate`, on purpose.
 */
export function usePwa() {
  const {
    needRefresh: [needRefresh, setNeedRefresh],
    updateServiceWorker,
  } = useRegisterSW({
    immediate: true,
    onRegisteredSW(_url, registration) {
      if (!registration) return
      regRef.current = registration
      lastCheckRef.current = Date.now()
    },
  })

  const [online, setOnline] = useState(navigator.onLine)
  // The registration and the last time we actually asked it to look. Refs, not
  // state: an update check must never itself cause a render (this hook sits above
  // the whole app).
  const regRef = useRef<ServiceWorkerRegistration | null>(null)
  const lastCheckRef = useRef<number | null>(null)
  // When this bundle started running, so the prompt can say how long they have
  // been on the old one instead of a bare "a new version is available".
  const loadedAtRef = useRef<number>(Date.now())
  const [installable, setInstallable] = useState(false)
  const promptRef = useRef<any>(null)
  const wakeRef = useRef<any>(null)
  const wantAwakeRef = useRef(false)

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

  // Ask the service worker to look for a new build: on a timer, and whenever the page returns to
  // the foreground.
  useEffect(() => {
    const check = (reason: 'interval' | 'visible') => {
      const reg = regRef.current
      if (!reg) return
      if (!shouldCheckForUpdate({
        lastCheckedAt: lastCheckRef.current,
        nowMs: Date.now(),
        online: navigator.onLine,
        visible: document.visibilityState === 'visible',
        reason,
      })) return
      lastCheckRef.current = Date.now()
      void reg.update().catch(() => { /* offline, or the server is down */ })
    }
    const timer = window.setInterval(() => check('interval'), 60_000)
    const onVisible = () => check('visible')
    document.addEventListener('visibilitychange', onVisible)
    return () => {
      window.clearInterval(timer)
      document.removeEventListener('visibilitychange', onVisible)
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
   * Hold the screen awake while any robot is running. A phone that sleeps mid-task drops the
   * camera sockets and the operator loses sight of a moving arm - the one moment the screen must
   * stay on.
   */
  const applyWakeLock = useCallback(async () => {
    const anyNav = navigator as any
    const action = wakeLockAction({
      want: wantAwakeRef.current,
      held: !!wakeRef.current,
      visible: document.visibilityState === 'visible',
      supported: !!anyNav.wakeLock,
    })
    if (action === 'request') {
      try {
        wakeRef.current = await anyNav.wakeLock.request('screen')
        // The browser releases the lock itself when the page is hidden; this keeps `held` honest so
        // the next visibility change knows to take it again.
        wakeRef.current.addEventListener?.('release', () => { wakeRef.current = null })
      } catch { /* denied, or the page went hidden mid-request */ }
    } else if (action === 'release') {
      try { await wakeRef.current.release() } catch { /* already gone */ }
      wakeRef.current = null
    }
  }, [])

  const keepAwake = useCallback(async (want: boolean) => {
    wantAwakeRef.current = want
    await applyWakeLock()
  }, [applyWakeLock])

  useEffect(() => {
    const onVisible = () => { void applyWakeLock() }
    document.addEventListener('visibilitychange', onVisible)
    return () => document.removeEventListener('visibilitychange', onVisible)
  }, [applyWakeLock])

  const standalone = window.matchMedia('(display-mode: standalone)').matches
    || (navigator as any).standalone === true

  return {
    online, needRefresh, update, installable, install, keepAwake, standalone,
    /** honest word about the screen: null when there is nothing to say (see lib/wakeLock) */
    wakeNote: () => wakeLockNote({
      want: wantAwakeRef.current,
      held: !!wakeRef.current,
      visible: document.visibilityState === 'visible',
      supported: !!(navigator as any).wakeLock,
    }),
    /** how long this tab has been running the bundle it loaded, for the update prompt */
    bundleAge: () => bundleAgeText(loadedAtRef.current, Date.now()),
    updateIntervalMs: SW_UPDATE_INTERVAL_MS,
  }
}
