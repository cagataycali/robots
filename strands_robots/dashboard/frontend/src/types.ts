export interface Presence {
  robot_id: string
  robot_type: 'robot' | 'sim' | string
  hostname?: string
  tool_name?: string
  task_status?: string
  instruction?: string
  connected?: boolean
  hw?: string
  cameras?: string[]
  sim_robots?: string[]
  action_keys?: string[]
  topics?: string[]
  timestamp?: number
}

export interface JointState { position?: number; velocity?: number }

export interface PeerState {
  peer_id: string
  t?: number
  joints?: Record<string, JointState | number[] | number>
  task?: { status?: string; instruction?: string; steps?: number; duration?: number }
  sim_time?: number
  robots?: Record<string, { active: boolean }>
}

export interface StreamStep {
  peer_id: string
  step: number
  t: number
  instruction: string
  policy: string
  observation: Record<string, unknown>
  action: Record<string, number | number[]>
}

export interface Peer {
  peer_id: string
  last_seen?: number
  stale?: boolean
  presence?: Presence
  state?: PeerState
  stream?: StreamStep
  cameras?: Record<string, { t?: number; shape?: number[] }>
}

export type MeshEvent =
  | { type: 'snapshot'; dashboard_peer_id: string; peers: Record<string, Peer>; t: number }
  | { type: 'presence'; peer_id: string; data: Presence }
  | { type: 'state'; peer_id: string; data: PeerState }
  | { type: 'stream'; peer_id: string; data: StreamStep }
  | { type: 'camera_meta'; peer_id: string; cam: string; data: { t?: number; shape?: number[] } }
  | { type: 'safety'; kind: 'estop' | 'resume'; data: Record<string, unknown> }
