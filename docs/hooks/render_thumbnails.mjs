// Render every renderable robot's catalog thumbnail with the site's own viewer, so the
// cards look exactly like the 3D view: paper background, soft shadow, faint grid.
//
//   (cd site && python3 -m http.server 8765 &)      # a built site
//   node docs/hooks/render_thumbnails.mjs [names...]  # needs playwright + chromium in the cwd's node_modules
//
// Writes docs/assets/img/robots/<name>.webp (800x600, q82). Idempotent; pass names to redo a few.
import { chromium } from "playwright";
import { readFileSync } from "node:fs";
import { execFileSync } from "node:child_process";

const HERE = new URL(".", import.meta.url).pathname;
const OUT = `${HERE}../assets/img/robots/`;
const manifest = JSON.parse(readFileSync(`${HERE}../assets/viewer/robots.json`, "utf8")).robots;
const names = process.argv.slice(2).length ? process.argv.slice(2) : Object.values(manifest).filter((r) => r.sim).map((r) => r.name);
const base = process.env.SITE_URL || "http://127.0.0.1:8765/";

const browser = await chromium.launch({ args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
const page = await browser.newPage({ viewport: { width: 1000, height: 800 }, deviceScaleFactor: 1 });
page.on("pageerror", (e) => console.log("  pageerror", e.message.slice(0, 120)));
await page.goto(base, { waitUntil: "load" });
let ok = 0, bad = [];
for (const name of names) {
  const t0 = Date.now();
  try {
    await page.evaluate((n) => {
      document.querySelectorAll("robot-viewer").forEach((v) => v.remove());
      const v = document.createElement("robot-viewer");
      v.setAttribute("name", n);
      v.style.cssText = "width:800px;height:600px;display:block;position:fixed;top:0;left:0;z-index:99999;border:0;border-radius:0;margin:0";
      document.body.prepend(v);
      v.load();
    }, name);
    await page.waitForFunction(() => ["ready", "error"].includes(document.querySelector("robot-viewer")?._state), null, { timeout: 180000 });
    const state = await page.evaluate(() => document.querySelector("robot-viewer")._state);
    if (state !== "ready") throw new Error(await page.evaluate(() => document.querySelector("robot-viewer").shadowRoot.querySelector(".status")?.textContent));
    await page.evaluate(() => { const v = document.querySelector("robot-viewer"); for (const s of [".joints", ".code", ".chrome"]) v.shadowRoot.querySelector(s).hidden = true; });
    await page.waitForTimeout(400);
    const png = `/tmp/thumb-${name}.png`;
    await page.locator("robot-viewer").screenshot({ path: png });
    execFileSync(process.env.PYTHON || "/Users/cagatay/robots/.venv/bin/python", ["-c", `from PIL import Image; Image.open("${png}").convert("RGB").save("${OUT}${name}.webp", quality=82)`]);
    ok++;
    console.log("ok  ", name, `${Date.now() - t0}ms`);
  } catch (e) {
    bad.push(name);
    console.log("FAIL", name, String(e.message || e).slice(0, 160));
  }
}
await browser.close();
console.log(`${ok} ok, ${bad.length} failed${bad.length ? ": " + bad.join(", ") : ""}`);
