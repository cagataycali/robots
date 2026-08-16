import { useEffect, useRef, useState } from 'react'
import { useVoice } from '../lib/useVoice'

interface ChatMsg {
  role: 'user' | 'agent'
  text: string
  tools?: { name: string; status: string }[]
}

/** Bottom-docked fleet agent: text chat + speech-to-speech toggle. */
export default function AgentDock() {
  const [open, setOpen] = useState(false)
  const [input, setInput] = useState('')
  const [msgs, setMsgs] = useState<ChatMsg[]>([])
  const [busy, setBusy] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const voice = useVoice()

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [msgs])

  const ensureWs = (): Promise<WebSocket> => new Promise((resolve, reject) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return resolve(wsRef.current)
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    const ws = new WebSocket(`${proto}://${location.host}/ws/chat`)
    ws.onopen = () => { wsRef.current = ws; resolve(ws) }
    ws.onerror = reject
    ws.onmessage = (msg) => {
      let ev: any
      try { ev = JSON.parse(msg.data) } catch { return }
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
        if (ev.type === 'token') last.text += ev.data
        else if (ev.type === 'tool' && ev.status === 'start') last.tools!.push({ name: ev.name, status: 'running' })
        else if (ev.type === 'tool') {
          const t = last.tools!.find(t => t.status === 'running')
          if (t) t.status = ev.status
        }
        else if (ev.type === 'done') { if (!last.text && ev.text) last.text = ev.text; setBusy(false) }
        else if (ev.type === 'error') { last.text += `\n⚠ ${ev.error}`; setBusy(false) }
        return next
      })
    }
    ws.onclose = () => { wsRef.current = null; setBusy(false) }
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
    } catch { setBusy(false) }
  }

  return (
    <>
      {open && (
        <div className="dock-panel">
          <div className="dock-scroll" ref={scrollRef}>
            {msgs.length === 0 && (
              <div className="dock-hint">
                Ask the fleet agent anything:<br />
                <em>"what robots are online?"</em><br />
                <em>"tell so101-arm-1 to pick up the red cube"</em><br />
                <em>"everyone stop"</em>
              </div>
            )}
            {msgs.map((m, i) => (
              <div key={i} className={`bubble ${m.role}`}>
                {m.tools?.map((t, j) => (
                  <span key={j} className={`toolchip ${t.status}`}>⚙ {t.name}</span>
                ))}
                <div className="bubble-text">{m.text || (busy && i === msgs.length - 1 ? '…' : '')}</div>
              </div>
            ))}
          </div>
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
