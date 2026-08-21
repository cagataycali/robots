/**
 * The browser every audit in this suite should launch: identical to playwright's, except that EVERY page
 * it creates is born with the hardware guard installed.
 *
 * Why not ask each audit to install it? Because 17 audits intercept routes and click buttons, and the
 * escape that motivated this (two real teleop_stop commands sent to a real arm, 2026-08-22) was ONE
 * CHARACTER CLASS in one glob in one of them. A protection that each new audit must remember is a
 * protection the next audit will forget — and its absence is invisible, because an escaped request looks
 * exactly like a fixture that did not apply. So the guard rides along with the browser, and the only
 * thing an audit does differently is import from here.
 *
 * Enforcement is automatic too: if any page attempted a mutating request that the audit did not
 * intercept, this module prints it and FAILS the process on exit, even for an audit that never calls
 * assertNoEscapes. An audit's own explicit call still works and reads better; this is the floor.
 */
import { chromium as real } from '/Users/cagatay/.tiny/npm/node_modules/playwright/index.mjs'
import { blockMutations, ESCAPES } from './audit-guard.mjs'

const wrapPage = async page => { await blockMutations(page); return page }

const wrapContext = ctx => new Proxy(ctx, {
  get(target, prop, recv) {
    if (prop === 'newPage') return async (...args) => wrapPage(await target.newPage(...args))
    const v = Reflect.get(target, prop, recv)
    return typeof v === 'function' ? v.bind(target) : v
  },
})

const wrapBrowser = browser => new Proxy(browser, {
  get(target, prop, recv) {
    if (prop === 'newContext') return async (...args) => wrapContext(await target.newContext(...args))
    if (prop === 'newPage') return async (...args) => wrapPage(await target.newPage(...args))
    const v = Reflect.get(target, prop, recv)
    return typeof v === 'function' ? v.bind(target) : v
  },
})

export const chromium = {
  ...real,
  launch: async (...args) => wrapBrowser(await real.launch(...args)),
  launchPersistentContext: async (...args) => wrapContext(await real.launchPersistentContext(...args)),
}

// The floor: an audit that forgets to assert still cannot pass while a mutation escaped it.
process.on('exit', () => {
  if (!ESCAPES.length) return
  for (const e of ESCAPES)
    console.error(`  FAIL  AUDIT ISOLATION: ${e} was attempted and NOT intercepted by this audit — the guard `
      + 'blocked it, so nothing reached the running dashboard, but the audit is one typo away from commanding '
      + "real hardware. Intercept it explicitly (regex, not a glob: playwright's * does not cross a /).")
  process.exitCode = 1
})
