// Check that a vtlengine wasm wheel installs with micropip on stock Pyodide and runs.
//
// This is what a Pyodide user does, `micropip.install(...)` resolving the wheel's dependencies
// against the Pyodide lockfile and PyPI (micropip does not backtrack), as opposed to the
// pip-based `pyodide venv` cibuildwheel tests the wheel with. release.yml runs it on the wheel
// it has just built. Locally:
//
//     npm install --no-save pyodide@314.0.6   # the line the wheel targets (pyemscripten_2026_0)
//     node scripts/check_micropip_install.mjs wheelhouse/vtlengine-*.whl [--preinstall REQ]...
//
// The wheel is copied into the Emscripten filesystem and installed from there (emfs:), then
// `import vtlengine` and one statement on both engines are run. `--preinstall REQ` installs a
// requirement beforehand, e.g. `--preinstall "pysdmx[xml]==1.16.0"` exercises the rest of the
// check while the lxml floor blocks the plain install (see KNOWN_FAILURES).
//
// Exit code 0: installed and ran, or the plain install failed with an allowlisted error
// (reported as a warning). Exit code 1: any other failure. Exit code 2: usage error.
// Unit tests of the decision logic: node --test scripts/check_micropip_install.test.mjs
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

/**
 * Install failures expected until an upstream fix ships. The check passes with a warning when
 * the plain install fails with one of these, and asks to drop the entry once it succeeds.
 */
export const KNOWN_FAILURES = [
  {
    needle: "Can't find a pure Python 3 wheel for 'lxml>=6.1.0",
    reason:
      "pysdmx[xml] requires lxml>=6.1.0 while Pyodide 314.x ships lxml 6.0.2, so micropip " +
      "cannot resolve the plain install until pyodide/pyodide-recipes#656 reaches a Pyodide " +
      "release.",
  },
];

const PY_SMOKE = `
import duckdb, lxml, networkx, numpy, pandas, pyarrow, pysdmx, sqlglot, vtlengine
from vtlengine import run

print("versions:", " | ".join(f"{m.__name__} {m.__version__}" for m in
      (vtlengine, pysdmx, lxml, pandas, numpy, pyarrow, duckdb, networkx, sqlglot)))
data_structures = {"datasets": [{"name": "DS_1", "DataStructure": [
    {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
    {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True}]}]}
for use_duckdb in (False, True):
    result = run(script="DS_r <- DS_1 * 10;", data_structures=data_structures,
                 datapoints={"DS_1": pandas.DataFrame({"Id_1": [1, 2], "Me_1": [10.0, 20.0]})},
                 use_duckdb=use_duckdb)
    got = result["DS_r"].data.sort_values("Id_1")["Me_1"].tolist()
    if got != [100.0, 200.0]:
        raise AssertionError(f"use_duckdb={use_duckdb}: expected [100.0, 200.0], got {got}")
    print(f"run() with use_duckdb={use_duckdb}: OK")
`;

export function parseArgs(argv) {
  const args = { wheel: null, preinstall: [] };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--preinstall") {
      if (i + 1 >= argv.length) throw new Error("--preinstall needs a requirement");
      args.preinstall.push(argv[++i]);
    } else if (arg.startsWith("--")) {
      throw new Error(`unknown option ${arg}`);
    } else if (args.wheel === null) {
      args.wheel = arg;
    } else {
      throw new Error(`expected exactly one wheel, got ${args.wheel} and ${arg}`);
    }
  }
  if (args.wheel === null) throw new Error("the wheel to check is required");
  return args;
}

/** Turn the outcome of the plain install into a verdict and an exit code. */
export function classifyInstall({ error, preinstalled, knownFailures = KNOWN_FAILURES }) {
  if (error) {
    const known = knownFailures.find((entry) => error.includes(entry.needle));
    if (known) return { verdict: "known-failure", exitCode: 0, note: known.reason };
    return { verdict: "failure", exitCode: 1, note: error };
  }
  if (!preinstalled && knownFailures.length > 0) {
    return {
      verdict: "unexpected-pass",
      exitCode: 0,
      note:
        "The plain install no longer fails with an allowlisted error: drop the obsolete " +
        "KNOWN_FAILURES entries.",
    };
  }
  return { verdict: "pass", exitCode: 0, note: "" };
}

/** Print a one-line message, doubled as a GitHub Actions annotation when running there. */
function annotate(level, message) {
  if (process.env.GITHUB_ACTIONS === "true") {
    const encoded = message.replace(/%/g, "%25").replace(/\r/g, "%0D").replace(/\n/g, "%0A");
    console.log(`::${level}::${encoded}`);
  }
  console.log(`${level.toUpperCase()}: ${message}`);
}

/** The Python traceback of a Pyodide error, or the JavaScript message otherwise. */
function describe(error) {
  return String(error?.message ?? error).trim();
}

function lastLine(text) {
  return text.split("\n").at(-1);
}

async function main() {
  let args;
  try {
    args = parseArgs(process.argv.slice(2));
  } catch (error) {
    console.error(`error: ${error.message}`);
    console.error("usage: node scripts/check_micropip_install.mjs <wheel> [--preinstall REQ]...");
    return 2;
  }
  const wheelName = path.basename(args.wheel);
  const wheelBytes = fs.readFileSync(args.wheel);

  let loadPyodide;
  try {
    ({ loadPyodide } = await import("pyodide"));
  } catch (error) {
    console.error(`error: ${describe(error)}`);
    console.error("Install the Pyodide runtime first: npm install --no-save pyodide@314.0.6");
    return 2;
  }
  const pyodide = await loadPyodide();
  await pyodide.loadPackage("micropip");
  const micropip = pyodide.pyimport("micropip");
  console.log(`Pyodide ${pyodide.version}, micropip ${micropip.__version__}, wheel ${wheelName}`);

  pyodide.FS.mkdirTree("/wheels");
  pyodide.FS.writeFile(`/wheels/${wheelName}`, wheelBytes);

  for (const requirement of args.preinstall) {
    console.log(`micropip.install(${JSON.stringify(requirement)})`);
    try {
      await micropip.install(requirement);
    } catch (error) {
      const traceback = describe(error);
      annotate("error", `Preinstalling ${requirement} failed: ${lastLine(traceback)}`);
      console.log(traceback);
      return 1;
    }
  }

  console.log(`micropip.install("emfs:/wheels/${wheelName}")`);
  let installError = null;
  try {
    await micropip.install(`emfs:/wheels/${wheelName}`);
  } catch (error) {
    installError = describe(error);
  }
  const outcome = classifyInstall({
    error: installError,
    preinstalled: args.preinstall.length > 0,
  });
  if (outcome.verdict === "known-failure") {
    annotate(
      "warning",
      `${wheelName} does not install with micropip on stock Pyodide, for a known reason: ` +
        outcome.note,
    );
    console.log(lastLine(installError.replace(/\nSee: .*|\nYou can use .*/gs, "")));
    return 0;
  }
  if (outcome.verdict === "failure") {
    annotate(
      "error",
      `${wheelName} does not install with micropip on stock Pyodide: ${lastLine(outcome.note)}`,
    );
    console.log(outcome.note);
    return 1;
  }
  if (outcome.verdict === "unexpected-pass") annotate("notice", outcome.note);

  try {
    await pyodide.runPythonAsync(PY_SMOKE);
  } catch (error) {
    const traceback = describe(error);
    annotate("error", `${wheelName} installed but failed to run: ${lastLine(traceback)}`);
    console.log(traceback);
    return 1;
  }
  console.log(`OK: ${wheelName} installs with micropip on stock Pyodide and runs on both engines`);
  return 0;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  process.exitCode = await main();
}
