"""A static feature browser: one HTML file that reads the JSON written by `features.analyze`.

Serve the features directory over HTTP (python -m http.server) or GitHub Pages. For small runs,
--embed inlines every JSON file so the page also opens straight from disk.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Model X-ray</title>
<style>
:root{--bg:#f7f6f3;--panel:#fff;--text:#1c1c1e;--muted:#6e6e73;--border:#e3e1dc;--accent:#b4410e;--hl:217,89,30;--chip:#efece6}
@media (prefers-color-scheme:dark){:root{--bg:#121212;--panel:#1b1b1d;--text:#ececed;--muted:#9b9ba1;--border:#2d2d30;--accent:#f59e5b;--hl:245,140,70;--chip:#262629}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif}
header{padding:14px 20px;border-bottom:1px solid var(--border);background:var(--panel);display:flex;gap:16px;align-items:baseline;flex-wrap:wrap}
header h1{font-size:17px;margin:0}
header .meta{color:var(--muted);font-size:13px}
main{display:grid;grid-template-columns:340px 1fr;min-height:calc(100vh - 54px)}
aside{border-right:1px solid var(--border);background:var(--panel);display:flex;flex-direction:column;min-height:0}
.controls{padding:12px;display:grid;gap:8px;border-bottom:1px solid var(--border)}
select,input{width:100%;padding:7px 9px;border:1px solid var(--border);border-radius:7px;background:var(--bg);color:var(--text);font:inherit}
#sitestats{font-size:12px;color:var(--muted)}
#list{overflow:auto;flex:1}
.item{padding:8px 12px;border-bottom:1px solid var(--border);cursor:pointer}
.item:hover,.item.active{background:var(--chip)}
.item .id{font-weight:600}
.item .sub{color:var(--muted);font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#more{margin:10px 12px;padding:7px;border:1px solid var(--border);background:var(--bg);color:var(--text);border-radius:7px;cursor:pointer}
section{padding:20px 24px;min-width:0}
h2{margin:0 0 4px;font-size:20px}
.label{color:var(--muted);margin-bottom:12px}
.stats{display:flex;gap:18px;flex-wrap:wrap;margin:10px 0 16px;color:var(--muted)}
.stats b{color:var(--text)}
.chips{display:flex;gap:6px;flex-wrap:wrap;margin:4px 0 14px}
.chip{background:var(--chip);border-radius:6px;padding:2px 7px;font-family:ui-monospace,Menlo,monospace;font-size:12.5px}
.chip small{color:var(--muted);margin-left:4px}
h3{font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin:16px 0 6px}
.ex{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:8px 10px;margin-bottom:6px;font-family:ui-monospace,Menlo,monospace;font-size:12.5px;white-space:pre-wrap;word-break:break-word}
.tok{border-radius:3px}
.tok.max{outline:1.5px solid var(--accent)}
.empty{color:var(--muted);padding:40px 0}
@media (max-width:760px){main{grid-template-columns:1fr}aside{max-height:45vh;border-right:0;border-bottom:1px solid var(--border)}section{padding:16px}}
</style>
</head>
<body>
<header><h1>Model X-ray</h1><span class="meta" id="meta"></span></header>
<main>
  <aside>
    <div class="controls">
      <select id="site"></select>
      <input id="q" placeholder="Search: feature id, label or promoted token">
      <select id="sort">
        <option value="density">Sort: most frequent</option>
        <option value="max">Sort: strongest activation</option>
        <option value="id">Sort: id</option>
      </select>
      <div id="sitestats"></div>
    </div>
    <div id="list"></div>
  </aside>
  <section id="detail"><div class="empty">Pick a feature on the left.</div></section>
</main>
<script>
const EMBED = __EMBED__;
const cache = {};
async function getJSON(path) {
  if (EMBED) return EMBED[path];
  if (!cache[path]) cache[path] = fetch(path).then(r => { if (!r.ok) throw new Error(path); return r.json(); });
  return cache[path];
}
const $ = id => document.getElementById(id);
const esc = s => s.replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const shown = s => esc(s).replace(/\n/g, "↵\n").replace(/\t/g, "→");
let manifest, site, rows = [], byId = new Map(), filtered = [], limit = 200, current = null;

function fmtDensity(d) { return d > 0 ? `1 in ${Math.round(1 / d).toLocaleString()} tokens` : "never fired"; }

async function init() {
  manifest = await getJSON("manifest.json");
  $("meta").textContent = `${manifest.model} · ${manifest.kind}s · ${manifest.tokens.toLocaleString()} tokens analysed`;
  $("site").innerHTML = manifest.sites.map(s => `<option value="${s.site}">${s.site} (${s.d_sae.toLocaleString()} features)</option>`).join("");
  const hash = new URLSearchParams(location.hash.slice(1));
  const start = manifest.sites.some(s => s.site === hash.get("site")) ? hash.get("site") : manifest.sites[0].site;
  $("site").value = start;
  await loadSite(start);
  if (hash.get("f")) showFeature(+hash.get("f"));
}

async function loadSite(name) {
  site = manifest.sites.find(s => s.site === name);
  rows = await getJSON(`${name}/index.json`);
  byId = new Map(rows.map(r => [r.id, r]));
  const e = site.eval;
  $("sitestats").innerHTML = `${(site.alive_fraction * 100).toFixed(1)}% of features fired` +
    (e ? ` · explained variance ${e.explained_variance.toFixed(3)} · loss recovered ${e.loss_recovered.toFixed(3)}` : "");
  limit = 200;
  applyFilter();
}

function applyFilter() {
  const q = $("q").value.trim().toLowerCase();
  const key = $("sort").value;
  filtered = rows.filter(r => r.density > 0 && (!q || String(r.id) === q ||
    (r.label || "").toLowerCase().includes(q) || r.up.some(t => t.toLowerCase().includes(q))));
  if (key === "density") filtered.sort((a, b) => b.density - a.density);
  else if (key === "max") filtered.sort((a, b) => b.max_act - a.max_act);
  else filtered.sort((a, b) => a.id - b.id);
  renderList();
}

function renderList() {
  const html = filtered.slice(0, limit).map(r => `<div class="item${current === r.id ? " active" : ""}" data-id="${r.id}">
    <div class="id">#${r.id}${r.label ? " · " + esc(r.label) : ""}</div>
    <div class="sub">${fmtDensity(r.density)} · ↑ ${r.up.map(t => esc(JSON.stringify(t))).join(" ")}</div></div>`).join("");
  $("list").innerHTML = html + (filtered.length > limit ? `<button id="more">Show more (${(filtered.length - limit).toLocaleString()} left)</button>` : "") +
    (filtered.length ? "" : `<div class="empty" style="padding:20px">No features match.</div>`);
}

async function showFeature(id) {
  current = id;
  const c = Math.floor(id / site.chunk_size);
  const chunk = await getJSON(`${site.site}/chunk_${c}.json`);
  const f = chunk.features[id];
  const row = byId.get(id);
  history.replaceState(null, "", `#site=${encodeURIComponent(site.site)}&f=${id}`);
  const chips = list => list.map(([t, v]) => `<span class="chip">${shown(t)}<small>${v.toFixed(2)}</small></span>`).join("");
  const examples = f.examples.map(ex => {
    const top = Math.max(...ex.acts, 1e-9);
    const iMax = ex.acts.indexOf(top);
    return `<div class="ex">${ex.tokens.map((t, i) => {
      const a = ex.acts[i] / top;
      const style = a > 0 ? ` style="background:rgba(var(--hl),${(0.12 + 0.7 * a).toFixed(2)})"` : "";
      return `<span class="tok${i === iMax ? " max" : ""}"${style} title="${ex.acts[i].toFixed(3)}">${shown(t)}</span>`;
    }).join("")}</div>`;
  }).join("");
  $("detail").innerHTML = `<h2>${esc(site.site)} · feature ${id}</h2>
    <div class="label">${row.label ? esc(row.label) : "No label yet"}</div>
    <div class="stats"><span>Fires on <b>${fmtDensity(row.density)}</b></span><span>Max activation <b>${row.max_act.toFixed(3)}</b></span><span>Mean when active <b>${row.mean_act.toFixed(3)}</b></span></div>
    <h3>Pushes these next tokens up</h3><div class="chips">${chips(f.up)}</div>
    <h3>Pushes these down</h3><div class="chips">${chips(f.down)}</div>
    <h3>Strongest activations</h3>${examples || '<div class="empty">No activations recorded.</div>'}`;
  renderList();
}

$("site").addEventListener("change", e => loadSite(e.target.value));
$("q").addEventListener("input", applyFilter);
$("sort").addEventListener("change", applyFilter);
$("list").addEventListener("click", e => {
  if (e.target.id === "more") { limit += 400; renderList(); return; }
  const item = e.target.closest(".item");
  if (item) showFeature(+item.dataset.id);
});
init().catch(err => { $("detail").innerHTML = `<div class="empty">Could not load data (${esc(String(err))}). Serve this folder over HTTP, e.g. python -m http.server.</div>`; });
</script>
</body>
</html>
"""


def _embedded_files(features_dir: Path, sample: int | None, seed: int = 0) -> dict:
    """All browser JSON keyed by relative path. With `sample`, keep a random sample of that many
    features that fired, per site, so a preview stays small."""
    manifest = json.loads((features_dir / "manifest.json").read_text())
    files = {"manifest.json": manifest}
    rng = random.Random(seed)
    for site in manifest["sites"]:
        name, size = site["site"], site["chunk_size"]
        index = json.loads((features_dir / name / "index.json").read_text())
        if sample:
            alive = [row for row in index if row["density"] > 0]
            index = sorted(rng.sample(alive, min(sample, len(alive))), key=lambda row: row["id"])
        files[f"{name}/index.json"] = index
        keep = {str(row["id"]) for row in index}
        for chunk_id in sorted({row["id"] // size for row in index}):
            chunk = json.loads((features_dir / name / f"chunk_{chunk_id}.json").read_text())
            chunk["features"] = {k: v for k, v in chunk["features"].items() if k in keep}
            files[f"{name}/chunk_{chunk_id}.json"] = chunk
    return files


def build_browser(features_dir: str | Path, embed: bool = False, sample: int | None = None,
                  out_name: str = "index.html") -> Path:
    features_dir = Path(features_dir)
    payload = "null"
    if embed:
        files = _embedded_files(features_dir, sample)
        payload = json.dumps(files, separators=(",", ":")).replace("</", "<\\/")
    out = features_dir / out_name
    out.write_text(TEMPLATE.replace("__EMBED__", payload))
    return out
