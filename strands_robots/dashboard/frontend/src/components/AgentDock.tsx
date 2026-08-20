import { useEffect, useRef, useState } from 'react'
import { useVoice } from '../lib/useVoice'
import { post, wsUrl } from '../lib/endpoints'
import { useConfig } from '../lib/useConfig'
import { sendFailureVerdict, interruptionNotice, bubbleLabel } from '../lib/chatDelivery'
import ConsentSheet from './ConsentSheet'
import { type ConsentNeed } from '../lib/consent'

interface ChatMsg {
  role: 'user' | 'agent' | 'notice'
  text: string
  reasoning?: string
  tools?: { name: string; status: string }[]
  /** user bubbles only: false = it provably never left the browser. ABSENT
   *  means delivered, which is the normal case and needs no decoration. */
  delivered?: boolean
  /** notice bubbles: render as a failure, not as information. */
  bad?: boolean
}

/** Bottom-docked fleet agent: text chat + speech-to-speech toggle. */
export default function AgentDock({ onSettings, startOpen = false, exampleRobot }: {
  onSettings?: () => void
  /** true when launched from the manifest's "Ask the agent" shortcut. */
  startOpen?: boolean
  /** a real online robot to name in examples - the placeholder is the
   *  de-facto tutorial, and it should teach a one-robot command before a
   *  fleet-wide one, with a name that actually exists on this desk. */
  exampleRobot?: string
}) {
  const [open, setOpen] = useState(startOpen)
  const [input, setInput] = useState('')
  const [msgs, setMsgs] = useState<ChatMsg[]>([])
  const [busy, setBusy] = useState(false)
  const [connError, setConnError] = useState<string | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  // onclose fires from a closure that captured `busy` at socket-open time, so the
  // "was a turn in flight?" question has to be asked of a ref, not of the state.
  const busyRef = useRef(false)
  // The trailing agent bubble as it stands right now, for the same reason: a
  // close handler must judge the answer it actually interrupted.
  const lastAgentRef = useRef<{ chars: number; running: string[] }>({ chars: 0, running: [] })
  const scrollRef = useRef<HTMLDivElement>(null)
  // A refused turn: the guard's decision, plus the sentence to re-send if it is granted. Without
  // this the chat box was the ONE surface where a continuable refusal was not continuable — the
  // operator had to go hunting through Settings for a permission the agent had just named.
  const [need, setNeed] = useState<ConsentNeed | null>(null)
  const refusedPrompt = useRef<string>('')
  const voice = useVoice()
  const { config, reload } = useConfig()
  const agent = config?.agent

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
    const last = msgs[msgs.length - 1]
    lastAgentRef.current = last?.role === 'agent'
      ? { chars: last.text.length, running: (last.tools ?? []).filter(t => t.status === 'running').map(t => t.name) }
      : { chars: 0, running: [] }
  }, [msgs])

  useEffect(() => { busyRef.current = busy }, [busy])

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
        setOpen(true)
        return
      }
      if (ev.type === 'pong') return
      if (ev.type === 'tool' && ev.needs_consent) setNeed(ev.needs_consent as ConsentNeed)
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
      // A reply landing in a collapsed dock is a reply the user never sees -
      // the transcript is the product here, so incoming activity reopens it.
      setOpen(true)
    }
    ws.onclose = (e) => {
      wsRef.current = null
      setBusy(false)
      // A drop mid-turn used to be silent unless it was 1008: the half-streamed
      // answer just stopped and read as finished — and if a tool had started,
      // "no answer" was a claim about the FLEET that nobody had checked.
      const verdict = interruptionNotice({
        code: e.code,
        wasBusy: busyRef.current,
        partialChars: lastAgentRef.current.chars,
        runningTools: lastAgentRef.current.running,
      })
      if (!verdict) return
      if (busyRef.current) {
        setMsgs(prev => [...prev, { role: 'notice', text: verdict.text, bad: true }])
        setOpen(true)
      } else {
        setConnError(verdict.text.replace(/^⚠ /, ''))
      }
    }
  })

  const send = async (retryText?: string) => {
    const text = (retryText ?? input).trim()
    if (!text || busy) return
    if (!retryText) setInput('')
    refusedPrompt.current = text
    setMsgs(prev => [...prev, { role: 'user', text }])
    setBusy(true)
    setConnError(null)
    try {
      const ws = await ensureWs()
      ws.send(JSON.stringify({ type: 'chat', text }))
    } catch (e: any) {
      setBusy(false)
      const verdict = sendFailureVerdict({ error: e?.message ?? String(e) })
      // The message never left the browser: mark THAT bubble (it must not read
      // as delivered) and give the operator their text back rather than making
      // them retype a sentence the UI silently swallowed.
      setMsgs(prev => prev.map((m, i) =>
        i === prev.length - 1 && m.role === 'user' ? { ...m, delivered: false } : m))
      if (verdict.retrySafe) setInput(text)
      setConnError(verdict.text.replace(/^⚠ /, ''))
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
                <em>"tell {exampleRobot ?? 'so101-arm-1'} to wave hello"</em><br />
                <em>"everyone stop" — the safety brake, it halts every robot</em>
                <p className="hint">
                  {/* Q80: this line used to read "it can start and stop real robots", which was true
                      and was the problem — that path had no confirmation of any kind. The agent may
                      now stop anything, and start motion on a SIM peer; starting a real arm is the
                      human pressing ▶, where the confirm sheet and the fit check live. */}
                  It can stop any robot, and start tasks in simulation. Starting a real arm stays with
                  you — press ▶ on its card. Everything it does is recorded in Activity.
                </p>
              </div>
            )}
            {msgs.map((m, i) => (
              m.role === 'notice' ? (
                <div key={i} className={m.bad ? 'dock-notice bad' : 'dock-notice'}>{m.bad ? '' : 'ⓘ '}{m.text}</div>
              ) : (
                <div key={i} className={`bubble ${m.role}${m.delivered === false ? ' undelivered' : ''}`}>
                  {m.tools?.map((t, j) => (
                    <span key={j} className={`toolchip ${t.status}`}>⚙ {t.name}</span>
                  ))}
                  {m.reasoning && (
                    <details className="reasoning"><summary>thinking</summary><pre>{m.reasoning}</pre></details>
                  )}
                  <div className="bubble-text">{m.text || (busy && i === msgs.length - 1 ? '…' : '')}</div>
                  {bubbleLabel(m.delivered) && (
                    <div className="bubble-foot">
                      <span className="badge warn">{bubbleLabel(m.delivered)}</span>
                      <button className="btn tiny" onClick={() => void send(m.text)} disabled={busy}>
                        send again
                      </button>
                    </div>
                  )}
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
          placeholder={`ask the fleet agent… (e.g. '${exampleRobot ?? 'so101-arm-1'}, wave hello')`}
          aria-label="message to the fleet agent"
          value={input}
          onFocus={() => setOpen(true)}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') void send() }}
          disabled={busy}
        />
        {/* "↑" is not a name. HelpSheet's ✕ and this dock's own hide/show button already carry
            one; the send button did not, so a screen reader reached the end of the agent's input
            row and announced an unlabelled "button". */}
        <button className="dock-send" onClick={() => void send()} disabled={busy || !input.trim()}
                aria-label="send to the agent" title="send to the agent">↑</button>
        <button className="dock-min" onClick={() => setOpen(o => !o)}
                aria-label={open ? 'hide the conversation' : 'show the conversation'}
                title={open ? 'hide the conversation' : 'show the conversation'}>
          {open ? '▾ hide' : `▴ chat${msgs.length ? ` (${msgs.length})` : ''}`}
        </button>
      </div>

      {/* The refusal names a permission; this makes it a decision, in the place the operator was
          already looking. target='spawn' because a chat turn CAN simply be re-sent once the grant
          lands — no process holds a stale env (the fleet tool reads it per call), unlike a running
          peer that needs a respawn. */}
      {need ? (
        <ConsentSheet
          need={need}
          target="spawn"
          onCancel={() => setNeed(null)}
          onRetry={() => { const again = refusedPrompt.current; setNeed(null); void send(again) }}
        />
      ) : null}
    </>
  )
}
