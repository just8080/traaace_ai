#!/usr/bin/env python3
"""Builds the TRAAACE AI add-on: the files to upload and the manifest.json that lists them.

  python3 scripts/make_pack.py --models <models_dir> --version 1.0.0 --min-app-version 0.6.0 --out <out_dir> [--local]

<models_dir> holds detector.onnx, classifier.onnx, class_names.json and name_map.csv.
ONNX Runtime comes from Microsoft's GitHub release (checked against GitHub's SHA-256) and FFmpeg from the latest
successful "Build FFmpeg" run in this repo. Both are fetched with the GitHub CLI (`gh`, signed in).

Without --local the download addresses point at this repo's release `ai-pack-v<version>`. With --local they point at
the files in <out_dir>, to test the download on this computer before anything is published.

Afterwards, sign the manifest (it asks for the key's password):
  minisign -S -s ~/.minisign/traaace_ai.key -m <out_dir>/manifest.json
"""
import argparse
import datetime as dt
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

REPO = "just8080/traaace_ai"
ORT_VERSION = "1.31.0"

# Platform (as the app names it) -> Microsoft's ONNX Runtime archive and the library inside it.
# Microsoft publishes no Intel Mac build after 1.23, so macos-x86_64 is not offered.
PLATFORMS = {
    "macos-aarch64": (f"onnxruntime-osx-arm64-{ORT_VERSION}.tgz", f"libonnxruntime.{ORT_VERSION}.dylib"),
    "windows-x86_64": (f"onnxruntime-win-x64-{ORT_VERSION}.zip", "onnxruntime.dll"),
    "linux-x86_64": (f"onnxruntime-linux-x64-{ORT_VERSION}.tgz", f"libonnxruntime.so.{ORT_VERSION}"),
}

NOTICE = {
    "title": "AI species detection",
    "requires_acceptance": True,
    "credits": [
        "Alita (New Zealand species classifier), version 3.03: Olly and wekaResearch, with the Department of Conservation "
        "Predator Science team. Everyone involved is listed at https://github.com/Wologman/Alita",
        "MegaDetector v5a (animal detection): Dan Morris and contributors, https://github.com/agentmorris/MegaDetector",
        "ONNX Runtime: Microsoft, MIT licence",
        "FFmpeg (video tool): https://ffmpeg.org, LGPL",
    ],
    "licenses": [
        "GPL-3.0 (the models, inherited from MegaDetector / YOLOv5)",
        "MIT (ONNX Runtime)",
        "LGPL v2.1 or later (FFmpeg)",
    ],
}

MODEL = {
    "id": "alita-3.03",
    "display_name": "Alita (New Zealand species)",
    "classes": 81,
    "detector_component": "detector",
    "classifier_component": "classifier",
    "class_names_component": "class-names",
    "name_map_component": "name-map",
    "defaults": {"classifier_threshold": 0.5, "detector_empty_threshold": 0.15, "count_threshold": 0.2},
    "score_kind": "uncalibrated",
    "calibration": None,
    "source": {"name": "Alita 3.03 official release (Exp_60, run_01), converted to ONNX by TRAAACE with the author's permission"},
}


def sha256_and_size(path: Path):
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def run(*cmd: str) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def fetch_onnxruntime(work: Path, platform: str) -> tuple[Path, Path]:
    """Downloads Microsoft's archive, checks it, and returns (runtime library, licence file)."""
    archive_name, library = PLATFORMS[platform]
    release = json.loads(run("gh", "api", f"repos/microsoft/onnxruntime/releases/tags/v{ORT_VERSION}"))
    asset = next(a for a in release["assets"] if a["name"] == archive_name)
    archive = work / archive_name
    if not archive.exists():
        print(f"  downloading {archive_name} ...", flush=True)
        run("gh", "release", "download", f"v{ORT_VERSION}", "-R", "microsoft/onnxruntime", "-p", archive_name, "-D", str(work))
    got, _ = sha256_and_size(archive)
    if f"sha256:{got}" != asset["digest"]:
        sys.exit(f"{archive_name}: checksum {got} does not match GitHub's {asset['digest']}")
    out = work / f"ort-{platform}"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    if archive_name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(out)
    else:
        with tarfile.open(archive) as t:
            t.extractall(out, filter="data")
    root = next(out.iterdir())
    return root / "lib" / library, root / "LICENSE"


def fetch_ffmpeg(work: Path) -> Path:
    """Downloads every platform's FFmpeg from the latest successful build in this repo."""
    run_id = json.loads(run("gh", "run", "list", "-R", REPO, "--workflow", "ffmpeg.yml", "--status", "success",
                            "--limit", "1", "--json", "databaseId"))[0]["databaseId"]
    out = work / "ffmpeg"
    shutil.rmtree(out, ignore_errors=True)
    print(f"  FFmpeg from build {run_id}", flush=True)
    for platform in PLATFORMS:
        run("gh", "run", "download", str(run_id), "-R", REPO, "-n", f"ffmpeg-{platform}", "-D", str(out / platform))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--version", required=True, help="add-on version, x.y.z")
    ap.add_argument("--min-app-version", required=True, help="oldest TRAAACE that can use it, x.y.z")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--local", action="store_true", help="file:// addresses, for testing on this computer")
    args = ap.parse_args()

    out = args.out.resolve()
    files, work = out / "files", out / "work"
    files.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    base = f"file://{files}" if args.local else f"https://github.com/{REPO}/releases/download/ai-pack-v{args.version}"

    # (id, label, kind, source file, install path, platforms, required, compress)
    parts = []
    print("ONNX Runtime", ORT_VERSION)
    licence = None
    for platform in PLATFORMS:
        library, licence = fetch_onnxruntime(work, platform)
        parts.append((f"onnxruntime-{platform}", "ONNX Runtime (runs the models)", "runtime", library,
                      f"runtime/{library.name}", [platform], True, True))
    parts.append(("onnxruntime-licence", "ONNX Runtime licence", "data", licence, "runtime/ONNXRUNTIME-LICENSE.txt", ["*"], True, False))

    print("FFmpeg")
    ffmpeg = fetch_ffmpeg(work)
    for platform in PLATFORMS:
        folder = ffmpeg / platform
        exe = "ffmpeg.exe" if platform.startswith("windows") else "ffmpeg"
        parts.append((f"ffmpeg-{platform}", "Video tool (FFmpeg)", "tool", folder / exe, f"tools/{exe}", [platform], False, True))
        parts.append((f"ffmpeg-source-{platform}", "Video tool source details", "data", folder / "SOURCE.txt",
                      "tools/FFMPEG-SOURCE.txt", [platform], False, False))
    parts.append(("ffmpeg-licence", "Video tool licence", "data", ffmpeg / "linux-x86_64" / "LICENSE.txt", "tools/FFMPEG-LICENSE.txt", ["*"], False, False))

    models = args.models.resolve()
    for cid, label, kind, name, required, compress in [
        ("detector", "Animal detector", "model", "detector.onnx", True, True),
        ("classifier", "Species classifier", "model", "classifier.onnx", True, True),
        ("class-names", "Species list", "data", "class_names.json", True, False),
        ("name-map", "Species names", "data", "name_map.csv", True, False),
    ]:
        parts.append((cid, label, kind, models / name, f"models/{name}", ["*"], required, compress))

    print("Packing")
    components = []
    for cid, label, kind, source, install_path, platforms, required, compress in parts:
        if not source.is_file():
            sys.exit(f"missing {source}")
        upload = files / (cid + (".gz" if compress else Path(install_path).suffix))
        print(f"  {cid}", flush=True)
        if compress:
            with source.open("rb") as src, gzip.open(upload, "wb", compresslevel=6) as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
        else:
            shutil.copyfile(source, upload)
        d_sha, d_size = sha256_and_size(upload)
        u_sha, u_size = sha256_and_size(source)
        components.append({
            "id": cid, "label": label, "kind": kind, "platforms": platforms, "install_path": install_path,
            "required": required,
            "download": {"url": f"{base}/{upload.name}", "compression": "gzip" if compress else "none", "size": d_size, "sha256": d_sha},
            "unpacked": {"size": u_size, "sha256": u_sha},
        })

    manifest = {
        "schema": 1, "pack": "traaace-ai", "version": args.version,
        "published": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "min_app_version": args.min_app_version, "notice": NOTICE, "model": MODEL, "components": components,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    shutil.rmtree(work)

    print(f"\nWrote {out / 'manifest.json'}")
    for platform in PLATFORMS:
        chosen = [c for c in components if "*" in c["platforms"] or platform in c["platforms"]]
        print(f"  {platform}: {sum(c['download']['size'] for c in chosen) / 1048576:.0f} MB download, "
              f"{sum(c['unpacked']['size'] for c in chosen) / 1048576:.0f} MB installed")
    print(f"\nNow sign it (asks for the key's password):\n  minisign -S -s ~/.minisign/traaace_ai.key -m {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
