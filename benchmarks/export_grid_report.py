"""Export one offline HTML report, preserving the original portable artifacts.

Requires the existing documentation environment (markdown-it-py via MyST).
No scientific calculations are rerun. Embedded gzip copies preserve the exact
input bytes and SHA256 hashes; the interactive view uses extracted summaries.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import html
import json
import math
import re
from pathlib import Path

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parent
FILES = (
    "grid_forces.json", "grid_orbits.json", "grid_timing.json",
    "grid_timing_awake.json", "grid_timing_host_context.json",
)


def radius_profile(points: list, errors: list, bins: int = 32) -> list[dict]:
    """Summarize all force probes in explicitly defined log-radius bins."""
    log_radii = [math.log10(math.sqrt(sum(x*x for x in p))) for p in points]
    low, high = min(log_radii), max(log_radii)
    groups = [[] for _ in range(bins)]
    for radius, error in zip(log_radii, errors, strict=True):
        index = min(bins - 1, int((radius - low) / (high - low) * bins))
        groups[index].append(error)
    result = []
    for i, group in enumerate(groups):
        if not group:
            continue
        group.sort()
        position = 0.5 * (len(group) - 1)
        a = int(position)
        next_value = group[min(a + 1, len(group) - 1)]
        median = group[a] + (position - a) * (next_value - group[a])
        result.append({
            "radius": 10 ** (low + (i + 0.5) * (high - low) / bins),
            "lower": 10 ** (low + i * (high - low) / bins),
            "upper": 10 ** (low + (i + 1) * (high - low) / bins),
            "maximum": max(group), "median": median, "count": len(group),
        })
    return result


def script_json(value: object) -> str:
    """Keep JSON inert even when a source string contains HTML delimiters."""
    return json.dumps(value, separators=(",", ":"), allow_nan=False).replace(
        "<", "\\u003c"
    )


def render_report(markdown: str, asset_names: set[str]) -> str:
    """Render the complete report with offline asset links and unique headings."""
    renderer = MarkdownIt("commonmark", {"html": False}).enable("table")

    def heading(tokens, index, options, env):
        title = tokens[index + 1].content
        tokens[index].attrSet("id", "full-" + re.sub(r"[^a-z0-9]+", "-", title.lower()))
        return renderer.renderer.renderToken(tokens, index, options, env)

    def link(tokens, index, options, env):
        token = tokens[index]
        href = token.attrGet("href") or ""
        name = Path(href).name
        if name in asset_names:
            token.attrSet("href", "#data")
            token.attrSet("data-download", name)
        elif href == "FORCE_GRID.html":
            token.attrSet("href", "#overview")
        elif href.startswith(("http://", "https://")):
            token.attrSet("rel", "noreferrer")
        elif not href.startswith("#"):
            raise ValueError(f"Unbundled report link: {href}")
        return renderer.renderer.renderToken(tokens, index, options, env)

    def image(tokens, index, options, env):
        if Path(tokens[index].attrGet("src") or "").name != "comparison.png":
            raise ValueError("Unknown report figure.")
        return '<a href="#figures">View the embedded scientific figure.</a>'

    renderer.renderer.rules.update(heading_open=heading, link_open=link, image=image)
    return renderer.render(markdown)


def export(output: Path) -> None:
    """Build a deterministic document from the measured artifacts and source report."""
    if output.exists():
        raise FileExistsError(f"Choose a new output path: {output}")
    sources = {name: (ROOT / "results" / name).read_bytes() for name in FILES}
    data = {name: json.loads(raw) for name, raw in sources.items()}
    forces, orbits = data[FILES[0]], data[FILES[1]]
    awake = data["grid_timing_awake.json"]
    if awake["parent_orbit_sha256"] != hashlib.sha256(sources[FILES[1]]).hexdigest():
        raise ValueError("Protected timing control references another orbit report.")
    if orbits["force_result_sha256"] != hashlib.sha256(sources[FILES[0]]).hexdigest():
        raise ValueError("Orbit and force artifacts do not match.")
    assets = {}
    for name, raw in sources.items():
        assets[name] = {
            "name": name, "mime": "application/json", "encoding": "gzip",
            "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "base64": base64.b64encode(gzip.compress(raw, mtime=0)).decode(),
        }
    markdown = (ROOT / "FORCE_GRID.md").read_text()
    for name, raw, mime in [
        ("FORCE_GRID.md", markdown.encode(), "text/markdown"),
        ("comparison.pdf",
         (ROOT / "figures/force-grid-20261018/comparison.pdf").read_bytes(),
         "application/pdf"),
    ]:
        assets[name] = {"name": name, "mime": mime, "encoding": "identity",
                        "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                        "base64": base64.b64encode(raw).decode()}
    assets["comparison.png"] = {"name": "comparison.png", "encoding": "figure"}
    force_rows = []
    for row in forces["force_rows"]:
        record = {key: row[key] for key in (
            "geometry", "variant", "batch_timing", "serial_timing", "probe_count"
        )}
        if row["variant"] != "direct":
            reference = next(
                r for r in forces["force_rows"]
                if r["geometry"] == row["geometry"] and r["variant"] == "direct"
            )
            record["setup"] = row["setup"]
            record["error"] = {key: row["force_error"][key] for key in (
                "max_relative", "p99_relative", "median_relative", "rms_relative"
            )}
            record["profile"] = radius_profile(
                reference["points_native"], row["force_error"]["relative_errors"]
            )
        force_rows.append(record)
    row_fields = (
        "case_id", "variant", "diagnostics", "warm_seconds", "warm_median_seconds",
        "lowering_and_compilation_seconds", "first_call_seconds",
        "split_half_variation",
    )
    summary = {
        "forces": force_rows, "cases": orbits["cases"],
        "orbits": [{key: row[key] for key in row_fields} for row in orbits["rows"]],
        "controls": orbits["control_comparisons"],
        "timing": data["grid_timing.json"]["records"], "awake": awake["records"],
        "protectedCase": awake["case_id"], "environment": orbits["environment"],
    }
    template = (ROOT / "force_grid_report.html.in").read_text()
    manifest = {name: {k: v for k, v in item.items() if k != "base64"}
                for name, item in assets.items()}
    replacements = {
        "@@SUMMARY@@": script_json(summary), "@@ASSETS@@": script_json(assets),
        "@@MANIFEST@@": html.escape(json.dumps(manifest, indent=2)),
        "@@REPORT@@": render_report(markdown, set(assets)),
        "@@FIGURE@@": "data:image/png;base64," + base64.b64encode(
            (ROOT / "figures/force-grid-20261018/comparison.png").read_bytes()
        ).decode(),
    }
    for marker, value in replacements.items():
        if template.count(marker) != 1:
            raise ValueError(f"Template must contain exactly one {marker}.")
        template = template.replace(marker, value)
    if re.search(r"@@[A-Z]+@@", template):
        raise ValueError("Unresolved template marker.")
    with output.open("x") as stream:
        stream.write(template)
    print(f"Wrote {output} ({output.stat().st_size / 1024**2:.2f} MiB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    export(parser.parse_args().output)
