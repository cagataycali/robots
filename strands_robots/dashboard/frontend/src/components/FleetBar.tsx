import type { MeshInfo } from '../types'
import type { ConnState } from '../lib/useMesh'
import { backendLabel } from '../lib/endpoints'
import { connBadge } from '../lib/connBadge'
import { absentNotice, quietNotice, type AbsentChild } from '../lib/absentChildren'
import StrandsMark from './StrandsMark'
import { useEffect, useRef, useState } from 'react'
import { useDialogFocus } from '../lib/useDialogFocus'
import { chooseScheme, effectiveScheme, nextScheme, storedScheme, systemScheme, type Scheme } from '../lib/scheme'

interface Props {
  conn: ConnState
  peerCount: number
  dashboardId: string
  safetyFlash: string | null
  mesh: MeshInfo
  online: boolean
  installable: boolean
  activityCount: number
  onInstall: () => void
  onSettings: () => void
  onWireSecurity: () => void
  onActivity: () => void
  absentChildren?: readonly AbsentChild[]
  quietChildren?: readonly string[]
  onDevices: () => void
  onHelp: () => void
}

export default function FleetBar({
  conn, peerCount, dashboardId, safetyFlash, mesh, online, installable,
  activityCount, absentChildren, quietChildren, onInstall, onSettings, onWireSecurity, onActivity, onDevices,
  onHelp,
}: Props) {
  // The mesh session and this browser's socket fail independently: the page can be LIVE while
  // the robot mesh is down, and vice versa.
  const absentDeath = absentNotice(absentChildren)
  const quiet = quietNotice(quietChildren, absentChildren)
  const meshDown = mesh.online === false
  const badge = connBadge(conn, { meshDown })

  // Phone (< 640px): the bar keeps the wordmark, the LIVE pill and one menu button; the
  // actions below open as a bottom sheet. STOP ALL is NOT in here - it stays in its own fixed
  // layer (components/EstopButton.tsx), so a menu never has to be opened to reach the brake.
  const [menuOpen, setMenuOpen] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)
  useDialogFocus(menuRef, menuOpen)
  useEffect(() => {
    if (!menuOpen) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setMenuOpen(false) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [menuOpen])
  /** One action closes the sheet first: the drawer it opens is the next screen. */
  const via = (fn: () => void) => () => { setMenuOpen(false); fn() }

  const connPill = (
    <span
      className={`conn ${conn}${badge.tone ? ` ${badge.tone}` : ''}`}
      title={badge.title}
      aria-label={badge.aria}
    >
      {badge.label}
    </span>
  )

  /** The same controls, rendered in the bar on a wide screen and in the sheet on a phone. */
  const actions = (
    <>
      {safetyFlash && (
        <span className={`safety ${safetyFlash}`}>
          {safetyFlash === 'estop' ? '🛑 E-STOP' : '✅ RESUMED'}
        </span>
      )}
      {!online && <span className="badge warn" title="this device has no network">offline</span>}
      {mesh.local_dev && (
        <button
          className="badge warnchip"
          onClick={via(onWireSecurity)}
          title="Robot mesh traffic is not encrypted. Fine on a trusted LAN - click for details and how to enable wire security."
        >
          mesh unencrypted · local only
        </button>
      )}
      {meshDown && <span className="badge danger" title="the dashboard's own mesh session is closed">mesh down</span>}

      {installable && (
        <button className="chip" onClick={via(onInstall)} title="Install as an app">⤓ install</button>
      )}
      <button className="chip" onClick={via(onDevices)} title="Local hardware and managed robots">⚙ devices</button>
      {/* U22: a robot the operator started died and the fleet only got shorter. */}
      {quiet && (
        <button
          className="chip warn"
          onClick={via(onDevices)}
          title={`${quiet.detail}\n\nOpen devices for its log — the refusal that kept it out of the fleet is in there (a missing calibration, a busy servo bus), and despawn is there too.`}
        >🫥 {quiet.headline}</button>
      )}
      {absentDeath && (
        <button
          className="chip warn"
          onClick={via(onDevices)}
          title={`${absentDeath.detail}\n\nOpen devices for the exit status and the last output.`}
        >⚰ {absentDeath.headline}</button>
      )}
      <button className="chip" onClick={via(onActivity)} title="Command history">
        ☰ activity{activityCount > 0 ? ` (${activityCount})` : ''}
      </button>
      <button className="chip" onClick={via(onSettings)} title="Settings">⚒ settings</button>
      {/* JOURNEYS #7: the page had 0 links and 0 onboarding words. */}
      <button
        className="chip"
        onClick={via(onHelp)}
        title="What this page is, how to stop a robot, and where the docs are"
        aria-keyshortcuts="?"
      >? help</button>

      <SchemeToggle />
      <span className="peers">{peerCount} peer{peerCount === 1 ? '' : 's'}</span>
    </>
  )

  return (
    <header className="fleetbar">
      <div className="brand">
        {/* The docs header, verbatim (overrides/partials/logo.html): the wordmark links out to
            strandsagents.com, "/robots" names this project. */}
        <a className="logo" href="https://strandsagents.com/" title="Strands Agents" aria-label="Strands Agents">
          <StrandsMark height={18} title="Strands Agents" />
        </a>
        <div>
          <h1 className="project">/robots</h1>
          <div className="sub" title={`API: ${backendLabel()}`}>
            {dashboardId || 'fleet cockpit'}
            <span className="backend"> · {backendLabel()}</span>
          </div>
        </div>
      </div>

      {/* phone row: LIVE pill + menu (hidden from 640px up, see styles.css) */}
      <div className="fleet-phone">
        {connPill}
        <button
          className={`chip menubtn${menuOpen ? ' on' : ''}`}
          onClick={() => setMenuOpen(o => !o)}
          aria-expanded={menuOpen}
          aria-controls="fleet-menu"
          aria-label={menuOpen ? 'close the menu' : 'open the menu: devices, activity, settings, help'}
          title="devices, activity, settings, help"
        >☰ menu{activityCount > 0 ? ` (${activityCount})` : ''}</button>
      </div>

      <div className="fleet-right">
        {actions}
        {connPill}
      </div>

      {menuOpen && (
        <div className="menu-backdrop" onClick={() => setMenuOpen(false)}>
          <div
            ref={menuRef}
            id="fleet-menu"
            className="menu-sheet"
            role="dialog"
            aria-modal="true"
            aria-label="Menu"
            onClick={e => e.stopPropagation()}
          >
            <div className="menu-head">
              <h2>Menu</h2>
              <span className="muted small mono">{dashboardId || 'fleet cockpit'} · {backendLabel()}</span>
              <button className="btn ghost" onClick={() => setMenuOpen(false)} aria-label="close the menu">✕</button>
            </div>
            <div className="menu-actions">{actions}</div>
          </div>
        </div>
      )}
    </header>
  )
}

/** paper / dark, the docs' toggle: the OS decides until the operator presses this, then the choice sticks. */
function SchemeToggle() {
  const [scheme, setScheme] = useState<Scheme>(() => effectiveScheme(storedScheme(), systemScheme()))
  useEffect(() => {
    // A stored choice wins; only an OS change with no stored choice moves the page.
    const mq = window.matchMedia('(prefers-color-scheme: dark)')
    const follow = (e: MediaQueryListEvent) => { if (!storedScheme()) setScheme(systemScheme(e.matches)) }
    mq.addEventListener('change', follow)
    return () => mq.removeEventListener('change', follow)
  }, [])
  const to = nextScheme(scheme)
  return (
    <button
      className="chip scheme"
      onClick={() => { chooseScheme(to); setScheme(to) }}
      title={`switch to the ${to} scheme`}
      aria-label={`colour scheme: ${scheme}. Switch to ${to}`}
    >{scheme === 'dark' ? '◐ dark' : '◑ paper'}</button>
  )
}
