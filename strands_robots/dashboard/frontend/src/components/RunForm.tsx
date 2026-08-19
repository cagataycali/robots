import { useEffect, useMemo, useState } from 'react'
import type { PolicyProvider } from '../types'
import { post } from '../lib/endpoints'
import CheckpointPicker from './CheckpointPicker'
import { useConfig } from '../lib/useConfig'
import { peekDeployIntent, clearDeployIntent, type DeployIntent } from '../lib/deployIntent'

export interface RunBody {
  instruction: string
  policy_provider: string
  duration: number
  [key: string]: any
}

interface Props {
  peerId: string
  running: boolean
  busy: boolean
  disabled?: boolean
  onRun: (body: RunBody) => void
  onStop: () => void
}

interface ValidateResult {
  ok: boolean
  stage: string
  error?: string
  note?: string
  observation_keys?: string[]
}

/**
 * The run form is *generated from the policy registry*, not hardcoded.
 *
 * `registry/policies.json` already states what each provider needs
 * (`requires`), what it accepts (`config_keys`) and what to prefill
 * (`defaults`). A fixed five-provider dropdown that sends only
 * `{instruction, policy_provider}` guarantees a failed run for every provider
 * that needs a port or a checkpoint - and the failure arrives 30 s later as a
 * timeout, on the robot, with no hint of the missing field.
 *
 * Fields are restricted to what the mesh command validator actually carries
 * (`wire_fields`); the provider's remaining kwargs are listed as
 * "local only" rather than rendered as inputs that get silently dropped.
 */
export default function RunForm({ peerId, running, busy, disabled, onRun, onStop }: Props) {
  const { policies } = useConfig()
  const [providerName, setProviderName] = useState('mock')
  const [instruction, setInstruction] = useState('')
  const [duration, setDuration] = useState(15)
  const [advanced, setAdvanced] = useState(false)
  const [fields, setFields] = useState<Record<string, string>>({})
  const [validating, setValidating] = useState(false)
  const [validation, setValidation] = useState<ValidateResult | null>(null)
  const [staged, setStaged] = useState<DeployIntent | null>(null)

  // A deploy intent staged from the Training tab prefills THIS form - once,
  // visibly, and only into fields; running still takes the human pressing
  // Run. Consumed on apply so it cannot ambush a second robot's form later.
  useEffect(() => {
    const intent = peekDeployIntent()
    if (!intent) return
    const target = policies.find(p =>
      p.wire_safe && p.wire_fields?.some(f => f.key === 'pretrained_name_or_path' || f.key === 'model_path'))
    if (!target) return
    const pathKey = target.wire_fields.find(f => f.key === 'pretrained_name_or_path' || f.key === 'model_path')!.key
    setProviderName(target.name)
    setFields(prev => ({
      ...prev,
      [pathKey]: intent.checkpoint,
      ...(intent.policy_type && target.wire_fields.some(f => f.key === 'policy_type')
        ? { policy_type: intent.policy_type } : {}),
    }))
    setStaged(intent)
    // the prefilled field lives in the advanced section - open it, because a
    // prefill the user cannot SEE is not "review then Run", it is a surprise
    setAdvanced(true)
    clearDeployIntent()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [policies])

  const provider: PolicyProvider | undefined = useMemo(
    () => policies.find(p => p.name === providerName),
    [policies, providerName],
  )
  const wireFields = provider?.wire_fields ?? []

  const value = (key: string, fallback: any) => {
    const raw = fields[key]
    if (raw !== undefined) return raw
    return fallback === null || fallback === undefined ? '' : String(fallback)
  }

  const missing = wireFields
    .filter(f => f.required && !String(value(f.key, f.default)).trim())
    .map(f => f.key)

  const parseField = (type: string, raw: string): any => {
    if (type === 'int') return Number.parseInt(raw, 10)
    if (type === 'float') return Number.parseFloat(raw)
    if (type === 'bool') return raw === 'true' || raw === '1'
    if (type === 'json') return JSON.parse(raw)
    return raw
  }

  /** Registry-keyed config for local validation. */
  const buildConfig = (): Record<string, any> => {
    const out: Record<string, any> = {}
    for (const f of wireFields) {
      const raw = String(value(f.key, f.default)).trim()
      if (!raw) continue
      try { out[f.key] = parseField(f.type, raw) } catch { out[f.key] = raw }
    }
    return out
  }

  const submit = () => {
    if (!instruction.trim() || missing.length) return
    const body: RunBody = {
      instruction: instruction.trim(),
      policy_provider: providerName,
      duration,
    }
    for (const f of wireFields) {
      const raw = String(value(f.key, f.default)).trim()
      if (!raw) continue
      try {
        body[f.wire_key] = parseField(f.type, raw)
      } catch {
        setValidation({ ok: false, stage: 'form', error: `${f.key}: not valid JSON` })
        return
      }
    }
    setValidation(null)
    onRun(body)
  }

  const validate = async () => {
    setValidating(true)
    try {
      setValidation(await post<ValidateResult>('/api/policies/validate', {
        policy_provider: providerName,
        policy_config: buildConfig(),
        peer_id: peerId,
      }))
    } catch (e: any) {
      setValidation({ ok: false, stage: 'request', error: e?.message ?? String(e) })
    } finally {
      setValidating(false)
    }
  }

  const locked = provider && !provider.wire_safe
  const blocked = !!disabled || busy

  return (
    <div className="runform">
      {staged && (
        <div className="deploy-banner">
          <span>🚀 prefilled from {staged.source} — review below, then press Run. Nothing has started.</span>
          <button className="btn ghost" onClick={() => { setStaged(null); setFields({}) }}>discard</button>
        </div>
      )}
      <div className="controls">
        <select
          value={providerName}
          onChange={e => { setProviderName(e.target.value); setFields({}); setValidation(null) }}
          disabled={blocked}
          title={provider?.description}
        >
          {policies.length === 0 && <option value="mock">mock</option>}
          {policies.map(p => (
            <option key={p.name} value={p.name} disabled={!p.wire_safe}>
              {p.wire_safe ? '' : '🔒 '}{p.name}{p.requires.length ? ` (${p.requires.join(',')})` : ''}
            </option>
          ))}
        </select>
        <input
          placeholder="pick up the red cube"
          value={instruction}
          onChange={e => setInstruction(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && submit()}
          disabled={blocked}
        />
        {running
          ? <button className="btn stop" onClick={onStop} disabled={busy} title="Stop this robot">■</button>
          : (
            <button
              className="btn go"
              onClick={submit}
              disabled={blocked || !instruction.trim() || missing.length > 0 || !!locked}
              title={locked
                ? `${providerName} is not in the mesh policy allowlist`
                : missing.length ? `missing: ${missing.join(', ')}` : 'Run'}
            >▶</button>
          )}
        <button
          className={advanced ? 'btn ghost on' : 'btn ghost'}
          onClick={() => setAdvanced(a => !a)}
          title="Provider options"
        >⚙</button>
      </div>

      {missing.length > 0 && !advanced && (
        <button className="needs" onClick={() => setAdvanced(true)}>
          {providerName} needs {missing.join(', ')} → open options
        </button>
      )}

      {advanced && (
        <div className="advanced">
          <label className="field">
            <span>duration (s)</span>
            <input
              type="number" min={1} max={600} value={duration}
              onChange={e => setDuration(Math.max(1, Number(e.target.value) || 1))}
              disabled={blocked}
            />
          </label>

          {wireFields.map(f => (
            <label className={f.required && !String(value(f.key, f.default)).trim() ? 'field missing' : 'field'} key={f.key}>
              <span>
                {f.key}{f.required && <b title="required"> *</b>}
                {f.wire_key !== f.key && <em title={`sent as ${f.wire_key}`}> →{f.wire_key}</em>}
              </span>
              {f.type === 'bool' ? (
                <select value={value(f.key, f.default) || 'false'} onChange={e => setFields(s => ({ ...s, [f.key]: e.target.value }))} disabled={blocked}>
                  <option value="false">false</option>
                  <option value="true">true</option>
                </select>
              ) : f.key === 'pretrained_name_or_path' || f.key === 'model_path' ? (
                /* Checkpoint fields get a Hub+local-cache type-ahead: the
                   registry names the field, this names the VALUES (thousands
                   of public LeRobot checkpoints). Picking one also prefills
                   policy_type when the checkpoint declares its family. */
                <CheckpointPicker
                  value={value(f.key, f.default)}
                  disabled={blocked}
                  onPick={(repoId, policyType) => setFields(s => {
                    const next = { ...s, [f.key]: repoId }
                    if (policyType && wireFields.some(w => w.key === 'policy_type') && !s.policy_type) {
                      next.policy_type = policyType
                    }
                    return next
                  })}
                />
              ) : (
                <input
                  type={f.type === 'int' || f.type === 'float' ? 'number' : 'text'}
                  value={value(f.key, f.default)}
                  placeholder={f.type === 'json' ? '{"…": …}' : f.type}
                  onChange={e => setFields(s => ({ ...s, [f.key]: e.target.value }))}
                  disabled={blocked}
                />
              )}
            </label>
          ))}

          <div className="advanced-actions">
            <button className="btn ghost" onClick={validate} disabled={validating}>
              {validating ? 'checking…' : '✓ validate'}
            </button>
            {provider?.server_based && (
              <span className="hint">needs a running inference server at that host/port</span>
            )}
          </div>

          {provider && provider.unsettable_over_mesh.length > 0 && (
            <div className="hint local-only">
              local-only kwargs (the mesh command schema does not carry them):{' '}
              <code>{provider.unsettable_over_mesh.join(', ')}</code>
            </div>
          )}
          {locked && (
            <div className="hint warn">
              🔒 {providerName} is rejected by the mesh policy allowlist. Add it to{' '}
              <code>STRANDS_MESH_POLICY_TYPE_ALLOW</code> on every peer.
            </div>
          )}
        </div>
      )}

      {validation && (
        <div className={validation.ok ? 'validation ok' : 'validation bad'}>
          {validation.ok ? `✓ ${providerName} resolves` : `✗ ${validation.stage}: ${validation.error}`}
          {validation.note && <div className="hint">{validation.note}</div>}
        </div>
      )}
    </div>
  )
}
