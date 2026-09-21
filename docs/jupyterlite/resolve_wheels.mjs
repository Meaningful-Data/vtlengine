// Resolve the vtlengine wheel's dependencies exactly the way micropip does for a user of
// stock Pyodide, by running micropip itself in the served distribution (Node.js 20+):
//
//     node resolve_wheels.mjs <output>/static/pyodide <vtlengine wheel> <micropip-lock.json>
//
// The wheel goes into the Emscripten file system and micropip.install() resolves its
// requirements against the distribution's lockfile and PyPI, with micropip's rules (newest
// allowed release, no backtracking, no pre-releases): what `micropip.install("vtlengine")`
// does in a notebook. micropip.freeze() then writes the lockfile of that install to the
// third argument; patch_lock.py merges its new entries into the served pyodide-lock.json.
import fs from "node:fs";
import path from "node:path";

const [dist, wheel, out, ...unexpected] = process.argv.slice(2);
if (!dist || !wheel || !out || unexpected.length > 0) {
  console.error("usage: node resolve_wheels.mjs <static/pyodide> <vtlengine wheel> <micropip-lock.json>");
  process.exit(2);
}
const indexURL = path.resolve(dist) + path.sep;
const { loadPyodide } = await import(path.join(indexURL, "pyodide.mjs"));
const pyodide = await loadPyodide({ indexURL });
await pyodide.loadPackage("micropip", { messageCallback: () => {} });
const micropip = pyodide.pyimport("micropip");
const name = path.basename(wheel);
pyodide.FS.mkdirTree("/wheels");
pyodide.FS.writeFile(`/wheels/${name}`, fs.readFileSync(wheel));
console.log(`  Pyodide ${pyodide.version}, micropip ${micropip.__version__}: micropip.install("emfs:/wheels/${name}")`);
await micropip.install(`emfs:/wheels/${name}`);
fs.writeFileSync(out, micropip.freeze());
