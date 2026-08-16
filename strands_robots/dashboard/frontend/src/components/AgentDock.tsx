import { useEffect, useRef, useState } from 'react'
import { useVoice } from '../lib/useVoice'
import { post, wsUrl } from '../lib/endpoints'
import { useConfig } from '../lib/useConfig'

interface ChatMsg {
  role: 'user' | 'agent' | 'notice'
  text: string
  reasoning?: string
  tools?: { name: string; status: string }[]
}

/** Bottom-docked fleet agent: text chat + speech-to-speech toggle. */
export default function AgentDock({ onSettings, startOpen = false }: {
  onSettings?: () => void
  /** true when launched from the manifest's "Ask the agent" shortcut. */
  startOpen?: boolean
}) {
  const [open, setOpen] = useState(startOpen)
  const [input, setInput] = useState('')
  const [msgs, setMsgs] = useState<ChatMsg[]>([])
  const [busy, setBusy] = useState(false)
  const [connError, setConnError] = useState<string | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const voice = useVoice()
  const { config, reload } = useConfig()
  const agent = config?.agent

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [msgs])

  /** Append to the trailing agent bubble, creating one if the last is not ours. */
  const patchAgent = (fn: (m: ChatMsg) => void) => {
    setMsgs(prev => {
      const next = [...prev]
      let last = next[next.length - 1]
      if (!last || last.role !== 'agent') {
        last = { role: 'agent', text: '', tools: [] }
        next.push(last)
      } else {
        last = { ...last, tools: [...(last.tools ?? [])] }
        next[next.length - 1] = last
      }
      fn(last)
      return next
    })
  }

  const ensureWs = (): Promise<WebSocket> => new Promise((resolve, reject) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return resolve(wsRef.current)
    const ws = new WebSocket(wsUrl('/ws/chat'))
    ws.onopen = () => { wsRef.current = ws; setConnError(null); resolve(ws) }
    ws.onerror = () => reject(new Error('could not reach the agent socket'))
    ws.onmessage = (msg) => {
      let ev: any
      try { ev = JSON.parse(msg.data) } catch { return }
      // A notice is about the conversation itself, not an answer - putting it in
      // the agent bubble makes the model look like it said it.
      if (ev.type === 'notice') {
        setMsgs(prev => [...prev, { role: 'notice', text: ev.text }])
        return
      }
      if (ev.type === 'pong') return
      patchAgent(last => {
        if (ev.type === 'token') last.text += ev.data
        else if (ev.type === 'reasoning') last.reasoning = (last.reasoning ?? '') + ev.data
        else if (ev.type === 'tool' && ev.status === 'start') last.tools!.push({ name: ev.name, status: 'running' })
        else if (ev.type === 'tool') {
          const t = last.tools!.find(t => t.status === 'running')
          if (t) t.status = ev.status
        }
        else if (ev.type === 'done') { if (!last.text && ev.text) last.text = ev.text; setBusy(false) }
        else if (ev.type === 'error') { last.text += `\n⚠ ${ev.error}`; setBusy(false) }
      })
      if (ev.type === 'done' || ev.type === 'error') void reload()
    }
    ws.onclose = (e) => {
      wsRef.current = null
      setBusy(false)
      if (e.code === 1008) setConnError('the server rejected this token — set it in Settings')
    }
  })

  const send = async () => {
    const text = input.trim()
    if (!text || busy) return
    setInput('')
    setMsgs(prev => [...prev, { role: 'user', text }])
    setBusy(true)
    try {
      const ws = await ensureWs()
      ws.send(JSON.stringify({ type: 'chat', text }))
    } catch (e: any) {
      setBusy(false)
      setConnError(e?.message ?? String(e))
    }
  }

  const clearHistory = async () => {
    try {
      await post('/api/agent/reset', { clear_history: true })
      setMsgs([])
      await reload()
    } catch (e: any) {
      setConnError(e?.message ?? String(e))
    }
  }

  return (
    <>
      {open && (
        <div className="dock-panel">
          <div className="dock-head">
            <span className="chip static" title={agent?.built ? 'model resolved' : 'built on the first turn'}>
              🤖 {agent?.model_id || 'default model'}
            </span>
            {agent?.busy && <span className="chip static">turn in flight</span>}
            {agent?.messages ? <span className="chip static">{agent.messages} msgs</span> : null}
            {agent?.bridge_online === false && (
              <span className="badge danger" title="the agent has no mesh bridge - its fleet tools cannot reach any robot">
                no mesh
              </span>
            )}
            <span className="spacer" />
            <button className="btn ghost" onClick={clearHistory} title="Forget the conversation">clear</button>
            {onSettings && <button className="btn ghost" onClick={onSettings} title="Model & prompt">⚒</button>}
          </div>
          <div className="dock-scroll" ref={scrollRef}>
            {msgs.length === 0 && (
              <div className="dock-hint">
                Ask the fleet agent anything:<br />
                <em>"what robots are online?"</em><br />
                <em>"tell so101-arm-1 to pick up the red cube"</em><br />
                <em>"everyone stop"</em>
                <p className="hint">
                  It can start and stop real robots. Everything it does is recorded in Activity.
                </p>
              </div>
            )}
            {msgs.map((m, i) => (
              m.role === 'notice' ? (
                <div key={i} className="dock-notice">ⓘ {m.text}</div>
              ) : (
                <div key={i} className={`bubble ${m.role}`}>
                  {m.tools?.map((t, j) => (
                    <span key={j} className={`toolchip ${t.status}`}>⚙ {t.name}</span>
                  ))}
                  {m.reasoning && (
                    <details className="reasoning"><summary>thinking</summary><pre>{m.reasoning}</pre></details>
                  )}
                  <div className="bubble-text">{m.text || (busy && i === msgs.length - 1 ? '…' : '')}</div>
                </div>
              )
            ))}
          </div>
          {connError && <div className="dock-notice bad">⚠ {connError}</div>}
          {voice.transcript && <div className="voice-transcript">{voice.transcript}</div>}
        </div>
      )}
      <div className="dock-bar">
        <button
          className={`mic ${voice.state}`}
          onClick={() => { setOpen(true); voice.toggle() }}
          title="Speech-to-speech fleet control"
        >
          {voice.state === 'live' ? '🔴' : voice.state === 'connecting' ? '⏳' : '🎙'}
        </button>
        <input
          placeholder="ask the fleet agent… (e.g. 'everyone pick up your cube')"
          value={input}
          onFocus={() => setOpen(true)}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && send()}
          disabled={busy}
        />
        <button className="dock-send" onClick={send} disabled={busy || !input.trim()}>↑</button>
        <button className="dock-min" onClick={() => setOpen(o => !o)}>{open ? '▾' : '▴'}</button>
      </div>
    </>
  )
}
