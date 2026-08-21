/**
 * Run EVERY pure-logic test under src/lib — the harness they never had.
 *
 * Each of these files asserts a rule this dashboard learned the hard way (an arm's role must never be
 * guessed from a name, a camera that dropped may not be called absent, an e-stop may not reassure from
 * an empty list). They are plain node:assert modules that import their subject from `/tmp/<name>.mjs`,
 * which means they only ever ran in the iteration that wrote them: whoever added one esbuilt it by
 * hand, watched it pass, and moved on. There was no `npm test`, so a refactor could quietly break any
 * of these rules and nothing would say so. An untriggerable test is a comment.
 *
 * This runner discovers the files, compiles each subject to the /tmp path its test already expects
 * (keeping the existing convention rather than rewriting 40 imports), runs them in a fresh node, and
 * reports one line each plus a total. Exit code is the sum of failures, so it can gate anything later.
 *
 *   node scripts/run-lib-tests.mjs            # all
 *   node scripts/run-lib-tests.mjs boardList  # only tests matching a substring
 */
import { execFileSync, spawnSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

const LIB = new URL('../src/lib/', import.meta.url).pathname
const filter = process.argv[2] ?? ''
const tests = fs.readdirSync(LIB).filter(f => f.endsWith('.test.mjs')).filter(f => f.includes(filter)).sort()

if (tests.length === 0) {
  console.error(`no lib tests match ${JSON.stringify(filter)}`)
  process.exit(2)
}

let failed = 0
const orphans = []

/** esbuild one lib file to an exact /tmp path (bundled: several libs import a sibling, and
 *  compiling file-by-file leaves a dangling import that dies as a resolution error — which
 *  reads like a broken rule rather than a missing build step). */
const build = (src, out) =>
  execFileSync('npx', ['esbuild', src, '--bundle', '--format=esm', `--outfile=${out}`, '--log-level=error'],
               { cwd: path.join(LIB, '../..'), stdio: 'pipe' })

for (const t of tests) {
  const subject = t.replace(/\.test\.mjs$/, '')
  const body = fs.readFileSync(path.join(LIB, t), 'utf8')

  // THREE conventions exist in this directory, and a runner that knows only the common one
  // reports the other two as failures of the product:
  //   1. `from '/tmp/<name>.mjs'`      — the usual case, name matches the test
  //   2. a header comment carrying the exact esbuild command (Q51's settingsTiming builds
  //      settingsMeta.ts to /tmp/sm.mjs), so the file documents its own build
  //   3. no /tmp import at all — self-contained (uploadArming restates a rule that cannot be
  //      imported without React), which must simply be RUN, not called orphaned
  const needed = [...new Set([...body.matchAll(/\/tmp\/([\w.-]+)\.mjs/g)].map(m => m[1]))]
  let buildFailed = false
  for (const name of needed) {
    let src = path.join(LIB, `${name}.ts`)
    if (!fs.existsSync(src)) {
      // convention 2: find `esbuild src/lib/<X>.ts … --outfile=/tmp/<name>.mjs` in the file itself
      const hint = body.match(new RegExp(`esbuild\\s+(\\S*src/lib/[\\w.-]+\\.ts)[^\\n]*--outfile=/tmp/${name}\\.mjs`))
      if (hint) src = path.join(LIB, '../..', hint[1])
    }
    if (!fs.existsSync(src)) {
      orphans.push(`${t} → /tmp/${name}.mjs (no source found)`)
      console.log(`  MISS  ${subject} — nothing builds /tmp/${name}.mjs`)
      buildFailed = true
      break
    }
    try {
      build(src, `/tmp/${name}.mjs`)
    } catch (e) {
      console.log(`  BUILD ${subject} — ${String(e.stderr ?? e).trim().split('\n')[0]}`)
      buildFailed = true
      break
    }
  }
  if (buildFailed) { failed += 1; continue }

  // A test that imports the .ts directly needs node to strip types (cameraFreshness, cameraPacing).
  const importsTs = /from\s+'[^']*\.ts'|import\(\s*'[^']*\.ts'/.test(body)
  const args = importsTs ? ['--experimental-strip-types', path.join(LIB, t)] : [path.join(LIB, t)]
  const r = spawnSync(process.execPath, args, { encoding: 'utf8' })
  if (r.status === 0) {
    console.log(`  ok    ${subject}${importsTs ? ' (ts)' : ''}`)
  } else {
    failed += 1
    console.log(`  FAIL  ${subject}`)
    for (const line of (r.stderr || r.stdout || '').trim().split('\n').filter(Boolean).slice(0, 6))
      console.log(`          ${line}`)
  }
}

console.log(failed
  ? `\n  ${failed} of ${tests.length} lib test file(s) FAILED${orphans.length ? `\n  orphaned: ${orphans.join('; ')}` : ''}`
  : `\n  PASS  ${tests.length} lib test files, every pure rule still holds`)

// A green rule that reaches no screen is the failure mode this project keeps paying for (three times
// in two days). These tests prove each rule is RIGHT; check-lib-wired proves it is REACHED. Skipped
// when a filter was given, because then only part of lib/ was under test and the graph is not the
// subject. It runs LAST so its verdict is the final line, and a dead module fails `npm test`.
let structural = 0
if (!filter) {
  // Two structural rules, run last so their verdict is the final line. Neither can be expressed as a
  // unit test, because both are about the SHAPE of the codebase rather than the behaviour of a function:
  //   check-lib-wired  — a rule that reaches no screen (paid for three times in two days);
  //   check-one-fetcher — a request that skips lib/endpoints, and with it the bearer token and the
  //                       server-age explanation that makes all 10 currently-dark routes degrade
  //                       honestly without a line of per-route code.
  //   check-retry-inputs — a retrying socket handed only PART of the evidence planRetry reads, which
  //     is how Q102 shipped wired to one of two sockets and cost a second iteration.
  //   check-routes-exist — an /api path no route serves (Q125). serverAge.ts answers this against
  //     the LIVE server, which cannot tell a typo from an old server: both are 404, and the UI then
  //     explains the wrong one ("restart the dashboard" about a route that never existed).
  //   check-css-wired — a STYLESHEET that reaches no bundle. check-lib-wired's sibling, and the same
  //     disease in a place the compiler cannot see: src/index.css was imported by nothing, so #2486's
  //     label rows and U22's death note shipped as class names no sheet defined, and a Q136 phone fix
  //     was then appended to the same dead file. All three builds were green.
  //   check-class-defined — Q137's disease generalised: markup naming a class NO stylesheet defines.
  //     Ten were already undefined when it was written; they are a listed, visible queue so the
  //     ELEVENTH fails today instead of after another arc of silence.
  //   check-dead-rules — Q143, the mirror of the one above: a RULE that nothing renders. Dead CSS
  //   ships to every visitor and makes the stylesheet read like a design system that is in use.
  for (const guard of ['check-lib-wired.mjs', 'check-css-wired.mjs', 'check-class-defined.mjs',
    'check-dead-rules.mjs',
    //   check-control-labels — Q144: a form control with no accessible name. A placeholder is not a
    //   label: it vanishes the moment the operator types, and a screen reader reads "edit text".
    'check-control-labels.mjs',
    //   check-live-regions — Q145: the e-stop verdict and the record refusal are ANSWERS to a tap; they
    //   must announce themselves, or they are seen and never heard.
    'check-live-regions.mjs',
                       'check-one-fetcher.mjs',
                       'check-retry-inputs.mjs',
                       'check-clamp-pairing.mjs', 'check-routes-exist.mjs', 'check-authed-images.mjs']) {
    const w = spawnSync(process.execPath, [new URL(guard, import.meta.url).pathname], { encoding: 'utf8' })
    process.stdout.write(w.stdout || '')
    process.stderr.write(w.stderr || '')
    if (w.status !== 0) structural = 1
  }
}
process.exit(failed || structural ? 1 : 0)
