import { useCallback, useRef, useState } from 'react'

export type VoiceState = 'idle' | 'connecting' | 'live' | 'error'

/** Browser mic (PCM16 @16k) ↔ /ws/voice ↔ bidi fleet agent. */
export function useVoice() {
  const [state, setState] = useState<VoiceState>('idle')
  const [transcript, setTranscript] = useState('')
  const wsRef = useRef<WebSocket | null>(null)
  const ctxRef = useRef<AudioContext | null>(null)
  const nodesRef = useRef<{ src?: MediaStreamAudioSourceNode; proc?: ScriptProcessorNode; stream?: MediaStream }>({})
  const playRef = useRef<{ ctx?: AudioContext; nextT: number; rate: number }>({ nextT: 0, rate: 24000 })

  const stop = useCallback(() => {
    wsRef.current?.send(JSON.stringify({ type: 'stop' }))
    wsRef.current?.close()
    wsRef.current = null
    nodesRef.current.proc?.disconnect()
    nodesRef.current.src?.disconnect()
    nodesRef.current.stream?.getTracks().forEach(t => t.stop())
    ctxRef.current?.close()
    ctxRef.current = null
    playRef.current.ctx?.close()
    playRef.current.ctx = undefined
    setState('idle')
  }, [])

  const start = useCallback(async () => {
    if (state === 'live' || state === 'connecting') { stop(); return }
    setState('connecting')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } })
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      const ws = new WebSocket(`${proto}://${location.host}/ws/voice`)
      wsRef.current = ws

      ws.onmessage = (msg) => {
        try {
          const ev = JSON.parse(msg.data)
          if (ev.type === 'voice_meta') { playRef.current.rate = ev.rate || 24000 }
          else if (ev.type === 'audio') playPcm(ev.data)
          else if (ev.type === 'transcript' && ev.text) setTranscript(`${ev.role === 'user' ? '🎙' : '🤖'} ${ev.text}`)
          else if (ev.type === 'error') { setTranscript(`⚠ ${ev.error}`); setState('error') }
        } catch { /* ignore */ }
      }
      ws.onclose = () => setState(s => (s === 'error' ? s : 'idle'))
      ws.onopen = () => {
        setState('live')
        // mic capture → downsample to 16k PCM16 → binary WS
        const ctx = new AudioContext()
        ctxRef.current = ctx
        const src = ctx.createMediaStreamSource(stream)
        const proc = ctx.createScriptProcessor(4096, 1, 1)
        nodesRef.current = { src, proc, stream }
        const inRate = ctx.sampleRate
        src.connect(proc)
        proc.connect(ctx.destination)
        proc.onaudioprocess = (e) => {
          if (ws.readyState !== WebSocket.OPEN) return
          const f32 = e.inputBuffer.getChannelData(0)
          const ratio = inRate / 16000
          const outLen = Math.floor(f32.length / ratio)
          const pcm = new Int16Array(outLen)
          for (let i = 0; i < outLen; i++) {
            const v = f32[Math.floor(i * ratio)]
            pcm[i] = Math.max(-32768, Math.min(32767, v * 32767))
          }
          ws.send(pcm.buffer)
        }
      }
    } catch (e) {
      setTranscript(`⚠ ${e}`)
      setState('error')
    }
  }, [state, stop])

  const playPcm = (b64: string) => {
    const p = playRef.current
    if (!p.ctx) { p.ctx = new AudioContext(); p.nextT = p.ctx.currentTime }
    const raw = atob(b64)
    const pcm = new Int16Array(raw.length / 2)
    for (let i = 0; i < pcm.length; i++) pcm[i] = (raw.charCodeAt(2 * i) | (raw.charCodeAt(2 * i + 1) << 8)) << 16 >> 16
    const buf = p.ctx.createBuffer(1, pcm.length, p.rate)
    const ch = buf.getChannelData(0)
    for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768
    const srcN = p.ctx.createBufferSource()
    srcN.buffer = buf
    srcN.connect(p.ctx.destination)
    const t = Math.max(p.ctx.currentTime, p.nextT)
    srcN.start(t)
    p.nextT = t + buf.duration
  }

  return { state, transcript, toggle: start, stop }
}
