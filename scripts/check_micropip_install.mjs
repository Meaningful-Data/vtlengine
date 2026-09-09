// Check that a vtlengine wasm wheel installs with micropip on stock Pyodide and runs.
//
// This is what a Pyodide user does, `micropip.install(...)` resolving the wheel's dependencies
// against the Pyodide lockfile and PyPI (micropip does not backtrack), as opposed to the
// pip-based `pyodide venv` cibuildwheel tests the wheel with. pyodide_test.yml (pull requests)
// and release.yml run it on the wheel they have just built. Locally:
//
//     npm install --no-save pyodide@314.0.6   # the line the wheel targets (pyemscripten_2026_0)
//     node scripts/check_micropip_install.mjs wheelhouse/vtlengine-*.whl
//
// The wheel is copied into the Emscripten filesystem and installed from there (emfs:), then
// `import vtlengine` and one statement on both engines are run.
//
// Exit code 0: installed and ran. Exit code 1: the install or the run failed. Exit code 2: usage
// error.
import fs from "node:fs";
import path from "node:path";

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

/** The line naming the exception, without the hints micropip appends after it. */
function summary(traceback) {
  return traceback.replace(/\n(?:See|You can use)\b.*$/s, "").split("\n").at(-1);
}

async function main() {
  const [wheel, ...unexpected] = process.argv.slice(2);
  if (!wheel || unexpected.length > 0) {
    console.error("usage: node scripts/check_micropip_install.mjs <wheel>");
    return 2;
  }
  const wheelName = path.basename(wheel);
  const wheelBytes = fs.readFileSync(wheel);

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

  console.log(`micropip.install("emfs:/wheels/${wheelName}")`);
  try {
    await micropip.install(`emfs:/wheels/${wheelName}`);
  } catch (error) {
    const traceback = describe(error);
    annotate(
      "error",
      `${wheelName} does not install with micropip on stock Pyodide: ${summary(traceback)}`,
    );
    console.log(traceback);
    return 1;
  }

  try {
    await pyodide.runPythonAsync(PY_SMOKE);
  } catch (error) {
    const traceback = describe(error);
    annotate("error", `${wheelName} installed but failed to run: ${summary(traceback)}`);
    console.log(traceback);
    return 1;
  }
  console.log(`OK: ${wheelName} installs with micropip on stock Pyodide and runs on both engines`);
  return 0;
}

process.exitCode = await main();
