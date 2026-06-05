#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import os
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


def read_container_url(args):
    if args.container_url:
        return args.container_url.strip()
    return Path(args.container_url_file).read_text().strip()


def build_url(container_url, extra_query):
    parts = urllib.parse.urlsplit(container_url)
    query = dict(urllib.parse.parse_qsl(parts.query, keep_blank_values=True))
    query.update(extra_query)
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, urllib.parse.urlencode(query), ""))


def blob_url(container_url, blob_name):
    parts = urllib.parse.urlsplit(container_url)
    path = parts.path.rstrip("/") + "/" + urllib.parse.quote(blob_name, safe="/")
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, path, parts.query, ""))


def list_blobs(container_url, prefix):
    marker = ""
    blobs = []
    while True:
        query = {"restype": "container", "comp": "list", "maxresults": "5000", "prefix": prefix}
        if marker:
            query["marker"] = marker
        with urllib.request.urlopen(build_url(container_url, query), timeout=120) as resp:
            root = ET.fromstring(resp.read())
        for blob in root.findall(".//Blob"):
            name = blob.findtext("Name")
            size_text = blob.findtext("Properties/Content-Length") or "0"
            if name:
                blobs.append({"name": name, "size": int(size_text)})
        print(f"listed {len(blobs)} blobs", flush=True)
        marker = root.findtext("NextMarker") or ""
        if not marker:
            return blobs


def download_one(container_url, out_dir, blob):
    name = str(blob["name"])
    size = int(blob["size"])
    target = out_dir / name
    if target.exists() and target.stat().st_size == size:
        return "skipped", size
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(blob_url(container_url, name), timeout=300) as resp, tmp.open("wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    got = tmp.stat().st_size
    if got != size:
        raise RuntimeError(f"size mismatch for {name}: got {got}, expected {size}")
    os.replace(tmp, target)
    return "downloaded", size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--container-url")
    ap.add_argument("--container-url-file")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args()
    if not (args.container_url or args.container_url_file):
        ap.error("provide --container-url or --container-url-file")
    container_url = read_container_url(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    blobs = list_blobs(container_url, args.prefix)
    total_bytes = sum(int(b["size"]) for b in blobs)
    print(f"found {len(blobs)} blobs under {args.prefix} ({total_bytes / 1024**3:.2f} GiB)", flush=True)
    if args.manifest:
        manifest = Path(args.manifest)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "size"])
            writer.writeheader()
            writer.writerows(blobs)
    done = skipped = failed = bytes_done = 0
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(download_one, container_url, out_dir, b) for b in blobs]
        for fut in concurrent.futures.as_completed(futs):
            try:
                status, size = fut.result()
                bytes_done += size
                if status == "skipped":
                    skipped += 1
                else:
                    done += 1
            except Exception as exc:
                failed += 1
                print(f"ERROR {exc}", file=sys.stderr, flush=True)
            completed = done + skipped + failed
            if completed % 25 == 0 or completed == len(blobs):
                elapsed = max(time.time() - start, 1)
                rate = completed / elapsed
                eta = (len(blobs) - completed) / rate if rate else 0
                print(f"progress {completed}/{len(blobs)} downloaded={done} skipped={skipped} failed={failed} bytes={bytes_done / 1024**3:.2f}GiB eta={eta/60:.1f}m", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
