#!/usr/bin/env node
/**
 * Assemble the static explainer into _site/.
 *
 * No bundler, no dependencies: the site is plain HTML, CSS and JS, so this only
 * copies the tree and stamps the build revision into the footer. Deterministic:
 * same inputs, same output.
 */
import { cp, mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const sourceDir = path.join(root, "site");
const outputDir = path.join(root, "_site");
const projectFile = path.join(root, "pyproject.toml");

async function readVersion() {
  if (process.env.EXU_VERSION) return process.env.EXU_VERSION;
  const project = await readFile(projectFile, "utf8");
  const match = project.match(/^version\s*=\s*"([^"]+)"/m);
  return match ? match[1] : "0.0.0";
}

async function stamp(directory, replacements) {
  const entries = await readdir(directory, { withFileTypes: true });
  for (const entry of entries) {
    const target = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      await stamp(target, replacements);
      continue;
    }
    if (!entry.name.endsWith(".html")) continue;
    let content = await readFile(target, "utf8");
    for (const [token, value] of Object.entries(replacements)) {
      content = content.split(token).join(value);
    }
    await writeFile(target, content, "utf8");
  }
}

async function main() {
  const version = await readVersion();
  const commit = (process.env.GITHUB_SHA || "dev").slice(0, 7);
  await rm(outputDir, { recursive: true, force: true });
  await mkdir(outputDir, { recursive: true });
  await cp(sourceDir, outputDir, { recursive: true });
  await stamp(outputDir, { __BUILD_COMMIT__: commit, __BUILD_VERSION__: version });
  // GitHub Pages runs Jekyll by default, which drops files starting with an underscore.
  await writeFile(path.join(outputDir, ".nojekyll"), "", "utf8");
  const files = await readdir(outputDir, { recursive: true });
  console.log(`built _site/ (${files.length} entries) at version ${version}, revision ${commit}`);
}

main().catch((error) => {
  console.error(`site build failed: ${error.message}`);
  process.exit(1);
});
