/**
 * Waiting for the request a claim actually depends on — instead of sleeping and hoping.
 *
 * Why this file exists (2026-08-21): audit-record-joint-warning read the arm <select> 2.5s after the
 * click. The options render from the mesh peers immediately, but the MEASURED roles arrive from a
 * separate /api/devices request, so the audit saw "role not measured" on an arm measured at 12.6V and
 * a journal entry recorded a role bug in the product. A race in an AUDIT does not fail loudly: it
 * invents a defect in the thing being audited, and the fix then goes to innocent code.
 *
 * Screens here fetch AFTER mount, several of them per panel, so the honest wait is "until the response
 * the assertion reads from has arrived".
 *
 * Companion trap, same day: never probe an API with page.evaluate(fetch). The app attaches its bearer
 * token itself, so a bare in-page fetch returns 401 while the app's own request beside it returns 200 —
 * evidence that reads exactly like an auth bug and is only the probe.
 */

/**
 * Arm the waits BEFORE the click that triggers them, then await the returned promise after it.
 *
 *   const ready = apiSettled(page, '/api/devices', '/api/calibration')
 *   await tab.click()
 *   await ready            // resolves when both have answered, or after `timeout`
 *
 * Never rejects: an audit whose subject legitimately does not call one of these paths must still be
 * able to run — the assertion that follows is what decides pass or fail, not the plumbing.
 */
export function apiSettled(page, ...paths) {
  const timeout = typeof paths[paths.length - 1] === 'number' ? paths.pop() : 15000
  const list = paths.length ? paths : ['/api/']
  return Promise.all(list.map(p =>
    page.waitForResponse(r => r.url().includes(p) && r.status() < 400, { timeout })
      .catch(() => null),
  ))
}

/** The paths a screen's own words depend on, so a caller names a SCREEN rather than plumbing. */
export const SCREEN_APIS = {
  devices: ['/api/devices', '/api/calibration'],
  record: ['/api/devices'],
}
