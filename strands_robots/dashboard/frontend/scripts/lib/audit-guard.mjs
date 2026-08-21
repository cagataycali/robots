/**
 * The last-resort hardware guard for browser audits.
 *
 * Written 2026-08-22 after audit-teleop-stop's own header promised "nothing reaches hardware" while its
 * glob pattern ('**\/teleop*') could not match '/teleop/stop' — playwright's '*' does not cross a '/' —
 * so two real stop commands went to a real arm. Nothing in that audit was wrong except one character
 * class, and nothing detected it: an escaped request looks exactly like a fixture that did not apply.
 *
 * Most audits in this suite fixture only the reads they need. That is fine for reads — an unintercepted
 * GET returns the live truth, which is usually what we want — but a POST/PUT/PATCH/DELETE nobody claimed
 * reaches the RUNNING dashboard, which spawns processes, respawns robots, submits training jobs and
 * commands physical arms. An audit must not be able to do that by omission.
 *
 * blockMutations(page) registers FIRST, so every specific handler an audit adds afterwards still wins
 * (playwright matches in REVERSE registration order). Anything mutating that no handler claimed is
 * answered with a 409 whose body says the guard blocked it, and RECORDED — so the audit can assert that
 * nothing unexpected was even attempted, instead of hoping.
 *
 *   const guard = await blockMutations(page)
 *   …
 *   guard.assertNoEscapes(failures)   // pushes one FAIL line per unclaimed mutation
 *
 * Escapes accumulate in a MODULE-LEVEL sink, so an audit that opens several pages (a factory that
 * builds one per scenario) guards each of them and still asserts exactly once, at the end, from the
 * top level — no plumbing a handle out of every closure.
 */
/** Every unclaimed mutation seen by any guarded page in this process. */
export const ESCAPES = []

/** The auth ceremony is NOT hardware and the APP ITSELF starts it: a page opened without a token POSTs
 *  /api/auth/login/begin on its own, so guarding it would fail future audits for a benign request they
 *  never made a choice about. Found while mutation-verifying this guard — the injected escape came back
 *  with the login POST beside it. Anything that spawns, respawns, records, trains or commands an arm is
 *  still blocked. */
const BENIGN = [/\/api\/auth\//]

/** One guard per page: the wrapped browser installs it at birth and an audit may still call it itself
 *  (older audits do, and the explicit call reads better) — installing twice would count one escape twice. */
const GUARDED = new WeakSet()

export async function blockMutations(page, { allow = [] } = {}) {
  allow = [...BENIGN, ...allow]
  if (GUARDED.has(page)) return { blocked: ESCAPES, assertNoEscapes }
  GUARDED.add(page)
  await page.route('**/api/**', async route => {
    const req = route.request()
    const method = req.method()
    if (method === 'GET' || method === 'HEAD' || method === 'OPTIONS') return route.continue()
    if (allow.some(rx => rx.test(req.url()))) return route.continue()
    ESCAPES.push(`${method} ${new URL(req.url()).pathname}`)
    return route.fulfill({
      status: 409, contentType: 'application/json',
      body: JSON.stringify({ detail: 'blocked by the audit hardware guard: this audit did not intercept this mutating request, so it was NOT forwarded to the running dashboard' }),
    })
  })
  return { blocked: ESCAPES, assertNoEscapes }
}

/** Every unclaimed mutation is a defect in the AUDIT, not the product: it means the audit is one typo
 *  away from commanding real hardware. Report it as such, naming the route. */
export function assertNoEscapes(failures) {
  for (const b of ESCAPES)
    failures.push(`AUDIT ISOLATION: ${b} was attempted but not intercepted by this audit — the guard blocked it, `
      + 'but on a suite without the guard it would have reached the running dashboard. Intercept it explicitly '
      + "(regex, not a glob: playwright's * does not cross a /).")
  return ESCAPES.length === 0
}
