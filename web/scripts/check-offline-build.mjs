import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
function files(dir) { return fs.readdirSync(dir, { withFileTypes: true }).flatMap(entry => entry.isDirectory() ? files(path.join(dir, entry.name)) : [path.join(dir, entry.name)]); }
assert(fs.existsSync("dist/index.html"), "Build the offline frontend first");
const assets = files("dist");
assert(!assets.some(file => /pyodide|\.wasm$|\.whl$|(?:Account|Community|Submissions|presence)[.-]/i.test(file)), "No online modules or browser Python runtime");
for (const file of assets.filter(file => /\.(?:js|html)$/.test(file))) {
  const content = fs.readFileSync(file, "utf8");
  for (const forbidden of [/paperleet\.com/, /paperleet\.github\.io/, /\/api\/(?:account|community|practice|presence)\//, /loadPyodide/])
    assert(!forbidden.test(content), `${file}: no hosted endpoint or browser judge may enter the offline bundle`);
}
assert(fs.readFileSync("dist/index.html", "utf8").includes("Content-Security-Policy"));
console.log("PASS offline production assets: no hosted services, online business chunks, WASM or Python downloads");
