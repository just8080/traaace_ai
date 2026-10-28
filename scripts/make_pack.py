#!/usr/bin/env python3
"""Builds the TRAAACE AI add-on: the files to upload and the manifest.json that lists them.

  python3 scripts/make_pack.py --models <models_dir> --version 1.0.0 --min-app-version 0.6.0 --out <out_dir> [--local]
  python3 scripts/make_pack.py --edition gpu --models <models_dir> --version 1.0.0 --min-app-version 0.7.0 --out <out_dir>

<models_dir> holds detector.onnx, classifier.onnx, class_names.json and name_map.csv.
ONNX Runtime comes from Microsoft's GitHub release (checked against GitHub's SHA-256) and FFmpeg from the latest
successful "Build FFmpeg" run in this repo. Both are fetched with the GitHub CLI (`gh`, signed in).

Without --local the download addresses point at this repo's release `ai-pack-v<version>` (`ai-pack-gpu-v<version>` for
the GPU add-on). With --local they point at the files in <out_dir>, to test the download on this computer before
anything is published.

--edition gpu builds the NVIDIA graphics card add-on (Windows only): Microsoft's CUDA 12 build of ONNX Runtime, which we
host, plus NVIDIA's CUDA and cuDNN libraries, which are never uploaded here. The manifest points at NVIDIA's own
downloads instead, and the app takes the libraries out of them. The script downloads them once (about 2.7 GB) to
check them against NVIDIA's checksums, record each library's checksum, and make sure no library needs a file that is
left out.

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

# Platform (as the app names it) -> Microsoft's ONNX Runtime archive and the libraries to take from it.
# Microsoft publishes no Intel Mac build after 1.23, so macos-x86_64 is not offered.
PLATFORMS = {
    "macos-aarch64": (f"onnxruntime-osx-arm64-{ORT_VERSION}.tgz", [f"libonnxruntime.{ORT_VERSION}.dylib"]),
    "windows-x86_64": (f"onnxruntime-win-x64-{ORT_VERSION}.zip", ["onnxruntime.dll"]),
    "linux-x86_64": (f"onnxruntime-linux-x64-{ORT_VERSION}.tgz", [f"libonnxruntime.so.{ORT_VERSION}"]),
}

# The GPU add-on: CUDA 12 (CUDA 13 drops GeForce GTX 900/1000 cards and older drivers). TensorRT is left out.
GPU_PLATFORMS = {
    "windows-x86_64": (f"onnxruntime-win-x64-gpu_cuda12-{ORT_VERSION}.zip",
                       ["onnxruntime.dll", "onnxruntime_providers_shared.dll", "onnxruntime_providers_cuda.dll"]),
}

# NVIDIA's libraries, from https://developer.download.nvidia.com/compute/{cuda,cudnn}/redist/redistrib_<version>.json:
# (list, its version, package, label, libraries to take; a name ending in * takes every match).
NVIDIA_REDIST = "https://developer.download.nvidia.com/compute"
NVIDIA_PACKAGES = [
    ("cuda", "12.9.1", "cuda_cudart", "NVIDIA CUDA runtime (from NVIDIA)", ["cudart64_12.dll"]),
    ("cuda", "12.9.1", "libcublas", "NVIDIA cuBLAS (from NVIDIA)", ["cublas64_12.dll", "cublasLt64_12.dll"]),
    ("cuda", "12.9.1", "libcufft", "NVIDIA cuFFT (from NVIDIA)", ["cufft64_11.dll"]),
    # cuDNN loads most of its own libraries while it runs, so take all of them.
    ("cudnn", "9.27.0", "cudnn", "NVIDIA cuDNN (from NVIDIA)", ["cudnn*64_9.dll"]),
]
# Windows' own files, the Microsoft C++ runtime (the standard add-on needs it too) and the NVIDIA driver's own files,
# which every computer with an NVIDIA card already has. Anything else a library needs must be in the add-on.
SYSTEM_DLLS = {
    "kernel32.dll", "user32.dll", "advapi32.dll", "ws2_32.dll", "shell32.dll", "ole32.dll", "oleaut32.dll", "setupapi.dll",
    "dbghelp.dll", "ntdll.dll", "bcrypt.dll", "crypt32.dll", "version.dll", "shlwapi.dll", "psapi.dll", "winmm.dll",
    "gdi32.dll", "cfgmgr32.dll", "dxgi.dll", "d3d12.dll", "dxcore.dll", "msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll",
    # The atomic_wait and codecvt_ids parts came with the 2022 C++ runtime; Microsoft's CUDA library needs the first.
    "msvcp140_atomic_wait.dll", "msvcp140_codecvt_ids.dll",
    "vcruntime140.dll", "vcruntime140_1.dll", "ucrtbase.dll", "nvcuda.dll", "nvml.dll", "nvapi64.dll",
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

GPU_NOTICE = {
    "title": "AI species detection (NVIDIA graphics card)",
    "requires_acceptance": True,
    "credits": NOTICE["credits"] + [
        "NVIDIA CUDA runtime, cuBLAS and cuFFT 12.9, and cuDNN 9.27: NVIDIA Corporation. Downloaded straight from NVIDIA "
        "and used only by TRAAACE.",
    ],
    "licenses": NOTICE["licenses"] + [
        "NVIDIA CUDA Toolkit End User License Agreement (https://docs.nvidia.com/cuda/eula/)",
        "NVIDIA cuDNN Software License Agreement (https://docs.nvidia.com/deeplearning/cudnn/latest/reference/eula.html)",
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


def fetch_onnxruntime(work: Path, platforms: dict, platform: str) -> tuple[list[Path], Path]:
    """Downloads Microsoft's archive, checks it, and returns (runtime libraries, licence file)."""
    archive_name, libraries = platforms[platform]
    release = json.loads(run("gh", "api", f"repos/microsoft/onnxruntime/releases/tags/v{ORT_VERSION}"))
    asset = next(a for a in release["assets"] if a["name"] == archive_name)
    archive = work / archive_name
    if not archive.exists():
        print(f"  downloading {archive_name} ...", flush=True)
        run("gh", "release", "download", f"v{ORT_VERSION}", "-R", "microsoft/onnxruntime", "-p", archive_name, "-D", str(work))
    got, _ = sha256_and_size(archive)
    if f"sha256:{got}" != asset["digest"]:
        sys.exit(f"{archive_name}: checksum {got} does not match GitHub's {asset['digest']}")
    out = work / f"ort-{archive_name}"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    if archive_name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(out)
    else:
        with tarfile.open(archive) as t:
            t.extractall(out, filter="data")
    root = next(out.iterdir())
    return [root / "lib" / library for library in libraries], root / "LICENSE"


def curl(*args: str) -> bytes:
    # curl rather than urllib: python.org's Python on a Mac often has no certificates set up.
    return subprocess.run(["curl", "-fsSL", *args], check=True, capture_output=True).stdout


def dll_imports(path: Path) -> set[str]:
    """The DLLs a Windows library loads at start-up, normal and delay-loaded, from its PE import tables."""
    data = path.read_bytes()
    u16 = lambda o: int.from_bytes(data[o:o + 2], "little")
    u32 = lambda o: int.from_bytes(data[o:o + 4], "little")
    pe = u32(0x3C)
    sections, opt = u16(pe + 6), pe + 24
    if u16(opt) != 0x20B:
        sys.exit(f"{path.name} is not a 64-bit Windows library")
    table = opt + u16(pe + 20)
    spans = [(u32(table + i * 40 + 12), u32(table + i * 40 + 8), u32(table + i * 40 + 20)) for i in range(sections)]

    def offset(rva: int) -> int:
        for start, size, raw in spans:
            if start <= rva < start + size:
                return raw + rva - start
        raise ValueError(f"{path.name}: address {rva:#x} is in no section")

    def name_at(rva: int) -> str:
        o = offset(rva)
        return data[o:data.index(b"\0", o)].decode("ascii").lower()

    names = set()
    # Data directory 1 is the import table (20-byte entries, name at +12); 13 is delay-loaded imports (32-byte, +4).
    for index, entry, at in ((1, 20, 12), (13, 32, 4)):
        rva = u32(opt + 112 + index * 8)
        if not rva:
            continue
        o = offset(rva)
        while u32(o + at):
            names.add(name_at(u32(o + at)))
            o += entry
    return names


def fetch_nvidia(work: Path) -> list[dict]:
    """Downloads NVIDIA's archives once, checks them against NVIDIA's own checksums, and describes each library to
    take from them: where the app downloads it, and its size and checksum once taken out."""
    found = []
    for kind, version, package, label, wanted in NVIDIA_PACKAGES:
        listing = json.loads(curl(f"{NVIDIA_REDIST}/{kind}/redist/redistrib_{version}.json"))
        entry = listing[package]["windows-x86_64"]
        entry = entry.get("cuda12", entry)  # cuDNN lists a build per CUDA version
        url = f"{NVIDIA_REDIST}/{kind}/redist/{entry['relative_path']}"
        archive = work / url.rsplit("/", 1)[1]
        got = sha256_and_size(archive) if archive.exists() else None
        if got != (entry["sha256"], int(entry["size"])):
            print(f"  downloading {archive.name} ({int(entry['size']) / 1e6:.0f} MB) from NVIDIA ...", flush=True)
            subprocess.run(["curl", "-fL", "--retry", "3", "-o", str(archive), url], check=True)
            got = sha256_and_size(archive)
            if got != (entry["sha256"], int(entry["size"])):
                sys.exit(f"{archive.name}: checksum {got[0]} does not match NVIDIA's {entry['sha256']}")
        out = work / f"nvidia-{package}"
        shutil.rmtree(out, ignore_errors=True)
        with zipfile.ZipFile(archive) as z:
            members = [m for m in z.namelist() if not m.endswith("/")]
            picked = []
            for name in wanted:
                match = [m for m in members if Path(m).match(f"*/bin/{name}") or Path(m).match(f"*/bin/x64/{name}")]
                if not match:
                    sys.exit(f"{archive.name} has no {name}")
                picked += match
            # NVIDIA's licence goes with its libraries.
            picked.append(next(m for m in members if m.count("/") == 1 and m.endswith("/LICENSE")))
            for member in picked:
                source = out / Path(member).name
                source.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member) as src, source.open("wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
                licence = member.endswith("/LICENSE")
                found.append({
                    "id": f"nvidia-{package}-licence" if licence else f"nvidia-{Path(member).stem.lower()}",
                    "label": label, "source": source, "member": member, "url": url, "licence": licence,
                    "install_path": f"runtime/NVIDIA-{package.upper()}-LICENSE.txt" if licence else f"runtime/{Path(member).name}",
                    "download": (int(entry["size"]), entry["sha256"]),
                })
    return found


def check_dependencies(libraries: list[Path]) -> None:
    """Stops if a library in the add-on needs a file that is neither in it nor part of Windows or the NVIDIA driver."""
    shipped = {p.name.lower() for p in libraries}
    for library in libraries:
        missing = {d for d in dll_imports(library) - shipped - SYSTEM_DLLS if not d.startswith(("api-ms-win-", "ext-ms-"))}
        if missing:
            sys.exit(f"{library.name} needs {', '.join(sorted(missing))}, which the add-on does not include")
    print(f"  all {len(libraries)} libraries have what they need", flush=True)


def fetch_ffmpeg(work: Path, platforms: dict) -> Path:
    """Downloads every platform's FFmpeg from the latest successful build in this repo."""
    run_id = json.loads(run("gh", "run", "list", "-R", REPO, "--workflow", "ffmpeg.yml", "--status", "success",
                            "--limit", "1", "--json", "databaseId"))[0]["databaseId"]
    out = work / "ffmpeg"
    shutil.rmtree(out, ignore_errors=True)
    print(f"  FFmpeg from build {run_id}", flush=True)
    for platform in platforms:
        run("gh", "run", "download", str(run_id), "-R", REPO, "-n", f"ffmpeg-{platform}", "-D", str(out / platform))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--version", required=True, help="add-on version, x.y.z")
    ap.add_argument("--min-app-version", required=True, help="oldest TRAAACE that can use it, x.y.z")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--local", action="store_true", help="file:// addresses, for testing on this computer")
    ap.add_argument("--edition", choices=["standard", "gpu"], default="standard",
                    help="gpu: the NVIDIA graphics card add-on (Windows)")
    args = ap.parse_args()
    gpu = args.edition == "gpu"
    platforms = GPU_PLATFORMS if gpu else PLATFORMS

    out = args.out.resolve()
    files, work = out / "files", out / "work"
    files.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    tag = f"ai-pack-gpu-v{args.version}" if gpu else f"ai-pack-v{args.version}"
    base = f"file://{files}" if args.local else f"https://github.com/{REPO}/releases/download/{tag}"

    # (id, label, kind, source file, install path, platforms, required, compress)
    parts = []
    print("ONNX Runtime", ORT_VERSION)
    licence = None
    windows_libraries = []
    for platform in platforms:
        libraries, licence = fetch_onnxruntime(work, platforms, platform)
        for library in libraries:
            # The first stays `onnxruntime-<platform>`, as in add-ons built before the GPU one.
            cid = f"onnxruntime-{platform}" if library == libraries[0] else f"{library.stem.lower().replace('_', '-')}-{platform}"
            parts.append((cid, "ONNX Runtime (runs the models)", "runtime", library, f"runtime/{library.name}", [platform], True, True))
        if platform == "windows-x86_64":
            windows_libraries += libraries
    parts.append(("onnxruntime-licence", "ONNX Runtime licence", "data", licence, "runtime/ONNXRUNTIME-LICENSE.txt", ["*"], True, False))

    nvidia = []
    if gpu:
        print("NVIDIA CUDA and cuDNN")
        nvidia = fetch_nvidia(work)
        check_dependencies(windows_libraries + [n["source"] for n in nvidia if not n["licence"]])

    print("FFmpeg")
    ffmpeg = fetch_ffmpeg(work, platforms)
    for platform in platforms:
        folder = ffmpeg / platform
        exe = "ffmpeg.exe" if platform.startswith("windows") else "ffmpeg"
        parts.append((f"ffmpeg-{platform}", "Video tool (FFmpeg)", "tool", folder / exe, f"tools/{exe}", [platform], False, True))
        parts.append((f"ffmpeg-source-{platform}", "Video tool source details", "data", folder / "SOURCE.txt",
                      "tools/FFMPEG-SOURCE.txt", [platform], False, False))
    parts.append(("ffmpeg-licence", "Video tool licence", "data", ffmpeg / next(iter(platforms)) / "LICENSE.txt", "tools/FFMPEG-LICENSE.txt", ["*"], False, False))

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
    for cid, label, kind, source, install_path, part_platforms, required, compress in parts:
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
            "id": cid, "label": label, "kind": kind, "platforms": part_platforms, "install_path": install_path,
            "required": required,
            "download": {"url": f"{base}/{upload.name}", "compression": "gzip" if compress else "none", "size": d_size, "sha256": d_sha},
            "unpacked": {"size": u_size, "sha256": u_sha},
        })

    # NVIDIA's files are never uploaded: the app downloads NVIDIA's archive and takes each one out of it.
    for n in nvidia:
        print(f"  {n['id']} (from NVIDIA)", flush=True)
        u_sha, u_size = sha256_and_size(n["source"])
        components.append({
            "id": n["id"], "label": n["label"], "kind": "data" if n["licence"] else "library", "platforms": ["windows-x86_64"],
            "install_path": n["install_path"], "extract": n["member"], "required": True,
            "download": {"url": n["url"], "compression": "zip", "size": n["download"][0], "sha256": n["download"][1]},
            "unpacked": {"size": u_size, "sha256": u_sha},
        })

    manifest = {
        "schema": 1, "pack": "traaace-ai", "edition": args.edition, "version": args.version,
        "published": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "min_app_version": args.min_app_version, "notice": GPU_NOTICE if gpu else NOTICE, "model": MODEL, "components": components,
    }
    name = "manifest-gpu.json" if gpu else "manifest.json"
    (out / name).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    shutil.rmtree(work)

    print(f"\nWrote {out / name}")
    for platform in platforms:
        chosen = [c for c in components if "*" in c["platforms"] or platform in c["platforms"]]
        # An archive several files come from is downloaded once, as the app does.
        downloads = {c["download"]["sha256"]: c["download"]["size"] for c in chosen}
        print(f"  {platform}: {sum(downloads.values()) / 1048576:.0f} MB download, "
              f"{sum(c['unpacked']['size'] for c in chosen) / 1048576:.0f} MB installed")
    if gpu:
        print(f"  upload the files in {files} to the release {tag}; NVIDIA's are not in there and are never uploaded")
    print(f"\nNow sign it (asks for the key's password):\n  minisign -S -s ~/.minisign/traaace_ai.key -m {out / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
