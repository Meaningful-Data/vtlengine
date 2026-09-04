// Unit tests for the decision logic of check_micropip_install.mjs; the Pyodide run itself is
// exercised by running the script against a wheel.
// Run with:  node --test scripts/check_micropip_install.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";

import { classifyInstall, KNOWN_FAILURES, parseArgs } from "./check_micropip_install.mjs";

// Verbatim from micropip 0.11.1 on Pyodide 314.0.6 (2026-09-04).
const LXML_MESSAGE = [
  `ValueError: Can't find a pure Python 3 wheel for 'lxml>=6.1.0; extra == "xml"'.`,
  "See: https://pyodide.org/en/stable/usage/faq.html" +
    "#why-can-t-micropip-find-a-pure-python-wheel-for-a-package",
].join("\n");

const ALLOWLIST = [{ needle: "lxml>=6.1.0", reason: "waiting for pyodide-recipes#656" }];

test("the shipped allowlist matches micropip's lxml message", () => {
  const outcome = classifyInstall({ error: LXML_MESSAGE, preinstalled: false });
  assert.equal(outcome.verdict, "known-failure");
  assert.equal(outcome.exitCode, 0);
  assert.ok(KNOWN_FAILURES.length > 0);
});

test("an allowlisted failure passes with the entry's reason", () => {
  const outcome = classifyInstall({
    error: LXML_MESSAGE,
    preinstalled: false,
    knownFailures: ALLOWLIST,
  });
  assert.equal(outcome.verdict, "known-failure");
  assert.equal(outcome.exitCode, 0);
  assert.match(outcome.note, /pyodide-recipes#656/);
});

test("any other failure fails the check with the original message", () => {
  const error = "ValueError: Can't find a pure Python 3 wheel for 'duckdb>=9'.";
  const outcome = classifyInstall({ error, preinstalled: false, knownFailures: ALLOWLIST });
  assert.equal(outcome.verdict, "failure");
  assert.equal(outcome.exitCode, 1);
  assert.match(outcome.note, /duckdb>=9/);
});

test("a plain install that succeeds despite an allowlist asks to drop the entry", () => {
  const outcome = classifyInstall({ error: null, preinstalled: false, knownFailures: ALLOWLIST });
  assert.equal(outcome.verdict, "unexpected-pass");
  assert.equal(outcome.exitCode, 0);
  assert.match(outcome.note, /KNOWN_FAILURES/);
});

test("a success after --preinstall is a plain pass", () => {
  const outcome = classifyInstall({ error: null, preinstalled: true, knownFailures: ALLOWLIST });
  assert.equal(outcome.verdict, "pass");
  assert.equal(outcome.exitCode, 0);
});

test("a success with an empty allowlist is a plain pass", () => {
  const outcome = classifyInstall({ error: null, preinstalled: false, knownFailures: [] });
  assert.equal(outcome.verdict, "pass");
  assert.equal(outcome.exitCode, 0);
});

test("parseArgs takes the wheel and repeated --preinstall requirements", () => {
  const wheel = "dist/vtlengine-1.0-cp314-cp314-pyemscripten_2026_0_wasm32.whl";
  const args = parseArgs([wheel, "--preinstall", "pysdmx[xml]==1.16.0", "--preinstall", "parsy"]);
  assert.equal(args.wheel, wheel);
  assert.deepEqual(args.preinstall, ["pysdmx[xml]==1.16.0", "parsy"]);
});

test("parseArgs rejects a missing wheel, a second wheel and unknown options", () => {
  assert.throws(() => parseArgs([]), /wheel/);
  assert.throws(() => parseArgs(["a.whl", "b.whl"]), /one wheel/);
  assert.throws(() => parseArgs(["a.whl", "--bogus"]), /--bogus/);
  assert.throws(() => parseArgs(["a.whl", "--preinstall"]), /--preinstall/);
});
