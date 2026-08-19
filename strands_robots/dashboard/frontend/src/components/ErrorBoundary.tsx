/**
 * The one thing a robot dashboard must never do: vanish.
 *
 * React unmounts the WHOLE tree when a render throws, so before this existed a
 * single bad payload (a session object with no `episodes`, an API returning an
 * object where a string was expected) blanked the app to a white page — and
 * took the E-STOP button out of the DOM with it, at exactly the moment the
 * operator most wants it. Measured in JOURNEYS.md #1: one click on ⏺ record
 * emptied #root.
 *
 * So every screen is wrapped individually. A crash is then contained to its own
 * panel: the fleet, the cards and the stop button keep rendering, and the
 * broken screen says what happened instead of pretending nothing did.
 */
import { Component, type ErrorInfo, type ReactNode } from 'react'

type Props = {
  /** Named in the message, e.g. "the record screen". */
  label: string
  /** Called when the user dismisses a crashed overlay (usually its onClose). */
  onDismiss?: () => void
  children: ReactNode
}

type State = { error: Error | null }

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // The console is where the operator's screenshot comes from, so keep the
    // component stack: the message alone rarely names the guilty screen.
    console.error(`[dashboard] ${this.props.label} crashed:`, error, info.componentStack)
  }

  /** Try this screen again without losing the rest of the session. */
  private retry = () => this.setState({ error: null })

  render() {
    const { error } = this.state
    if (!error) return this.props.children
    return (
      <div className="crashcard" role="alert">
        <h3>{this.props.label} stopped working</h3>
        <p>
          The rest of the dashboard is still live — the fleet, the robot cards and
          <b> STOP ALL</b> all still work.
        </p>
        <pre>{error.message || String(error)}</pre>
        <div className="crashcard-actions">
          <button className="btn go" onClick={this.retry}>try again</button>
          {this.props.onDismiss && (
            <button className="btn ghost" onClick={this.props.onDismiss}>close</button>
          )}
          <button className="btn ghost" onClick={() => location.reload()}>reload the page</button>
        </div>
        <p className="hint">
          Details are in the browser console. If it crashes again on the same screen, that
          screen's data is the problem, not your fleet.
        </p>
      </div>
    )
  }
}
