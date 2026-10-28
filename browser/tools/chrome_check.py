"""Headless Google Chrome check of the page: WebGPU on, ?auto=1 runs load + (photo) + generate.

    uv run --no-project --with websockets python tools/chrome_check.py http://localhost:8811/ <profile_dir> <runs.json>
runs.json: [{"label": "...", "query": "&seed=42...", "ref_frames": "./model/x.rgb" (optional), "ref_photo": "..." (optional),
             "png": "<path to save the canvas>" (optional)}]
"""
import asyncio, base64, json, subprocess, sys, time, urllib.request
import websockets

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
URL, PROFILE, RUNS = sys.argv[1], sys.argv[2], json.load(open(sys.argv[3]))
PORT = 9333
EXPR = """(async () => { for (let i = 0; i < 1800 && !window.__done; i++) await new Promise(r => setTimeout(r, 100));
  const d = window.__drishya, t = window.__last && window.__last.timings;
  const psnr = async (url, arr) => { if (!url || !arr) return null; const r = await fetch(url); if (!r.ok) return null;
    const b = new Uint8Array(await r.arrayBuffer()); let se = 0, mx = 0;
    for (let i = 0; i < b.length; i++) { const e = arr[i] - b[i]; se += e * e; mx = Math.max(mx, Math.abs(e)); }
    return {psnr: se ? 10 * Math.log10(255 * 255 / (se / b.length)) : Infinity, maxAbs: mx, n: b.length}; };
  let photoArr = null;
  if (REFPHOTO) { const c = document.getElementById('photoview').getContext('2d').getImageData(0, 0, 256, 256).data;
    photoArr = new Uint8Array(256 * 256 * 3); for (let i = 0; i < 65536; i++) for (let k = 0; k < 3; k++) photoArr[3 * i + k] = c[4 * i + k]; }
  return JSON.stringify({done: !!window.__done, status: document.getElementById('status').textContent,
    photostatus: document.getElementById('photostatus').textContent,
    step: document.getElementById('stepstatus').textContent, meta: document.getElementById('meta').textContent,
    stats: d && d.stats, mode: t && t.mode, totalMs: t && t.totalMs, imageMs: t && t.imageMs, textMs: t && t.textMs,
    decodeMs: t && t.decodeMs, avgStepMs: t && t.stepMs.reduce((x, y) => x + y, 0) / t.stepMs.length,
    sinceNav: performance.now(), frames: await psnr(REFFRAMES, window.__last && window.__last.frames),
    photoPrep: await psnr(REFPHOTO, photoArr), frame0: document.getElementById('view').toDataURL('image/png')}); })()"""


async def run(ws, url, ref_frames, ref_photo):
    n = 0
    async def call(method, params=None):
        nonlocal n; n += 1; my = n
        await ws.send(json.dumps({"id": my, "method": method, "params": params or {}}))
        while True:
            m = json.loads(await ws.recv())
            if m.get("id") == my:
                return m
    await call("Page.enable"); await call("Page.navigate", {"url": url})
    await asyncio.sleep(1)
    expr = EXPR.replace("REFFRAMES", json.dumps(ref_frames)).replace("REFPHOTO", json.dumps(ref_photo))
    r = await call("Runtime.evaluate", {"expression": expr, "awaitPromise": True, "returnByValue": True, "timeout": 300000})
    return json.loads(r["result"]["result"]["value"])


async def main():
    p = subprocess.Popen([CHROME, "--headless=new", f"--user-data-dir={PROFILE}", f"--remote-debugging-port={PORT}",
                          "--enable-unsafe-webgpu", "--no-first-run", "--no-default-browser-check", "about:blank"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json")); break
            except Exception:
                time.sleep(0.2)
        page = [t for t in tabs if t["type"] == "page"][0]
        async with websockets.connect(page["webSocketDebuggerUrl"], max_size=2**26) as ws:
            for spec in RUNS:
                t0 = time.time()
                res = await run(ws, URL + "?auto=1" + spec.get("query", ""), spec.get("ref_frames"), spec.get("ref_photo"))
                res["wallSec"] = time.time() - t0
                png = res.pop("frame0", None)
                if png and spec.get("png"):
                    open(spec["png"], "wb").write(base64.b64decode(png.split(",", 1)[1]))
                print(spec["label"], json.dumps(res, indent=1), flush=True)
    finally:
        p.terminate()

asyncio.run(main())
