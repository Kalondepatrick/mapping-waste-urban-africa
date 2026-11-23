"""Upload a staged data package to Zenodo as an UNPUBLISHED draft.

This script deliberately does not publish. A published Zenodo record is
permanent: the DOI is minted, the files become public, and the deposit cannot
be withdrawn without contacting Zenodo. The final publish is a decision for a
person, made in the Zenodo web interface after reviewing the draft.

What it does
------------
1. creates a draft deposition from tools/zenodo_metadata.json
2. uploads every file in the staged package, resuming anything already there
3. prints the draft URL for you to review and publish yourself

Usage
-----
Set your token as an environment variable first, so it never appears on the
command line or in shell history. Create one at
https://zenodo.org/account/settings/applications/tokens/new/ with the scopes
`deposit:write` and `deposit:actions`.

    $env:ZENODO_TOKEN = "..."
    python tools/upload_to_zenodo.py --package ../zenodo_waste_ndirande

Test against the sandbox first; it is a separate site with separate tokens:

    python tools/upload_to_zenodo.py --package ... --sandbox

Resuming after an interrupted upload:

    python tools/upload_to_zenodo.py --package ... --deposition 1234567
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(HERE, "zenodo_metadata.json")


def api_base(sandbox: bool) -> str:
    return "https://sandbox.zenodo.org/api" if sandbox else "https://zenodo.org/api"


def request(url: str, token: str, method: str = "GET", payload=None,
            data_file: str | None = None, content_type: str | None = None):
    headers = {"Authorization": f"Bearer {token}"}
    body = None
    if payload is not None:
        body = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    if data_file is not None:
        body = open(data_file, "rb")
        headers["Content-Type"] = content_type or "application/octet-stream"
        headers["Content-Length"] = str(os.path.getsize(data_file))
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as r:
            raw = r.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:600]
        raise SystemExit(f"\nZenodo returned {e.code} for {method} {url}\n{detail}\n")
    finally:
        if data_file is not None and body is not None:
            body.close()


def md5(path: str, buf: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(buf)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def collect(package: str) -> list[tuple[str, str]]:
    """(absolute path, path relative to the package root), largest last."""
    out = []
    for root, _, files in os.walk(package):
        for f in sorted(files):
            p = os.path.join(root, f)
            out.append((p, os.path.relpath(p, package).replace(os.sep, "/")))
    out.sort(key=lambda t: os.path.getsize(t[0]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", required=True,
                    help="staged directory from tools/make_zenodo_package.py")
    ap.add_argument("--sandbox", action="store_true",
                    help="use sandbox.zenodo.org (separate account and token)")
    ap.add_argument("--deposition", help="resume into an existing draft id")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be uploaded and exit")
    a = ap.parse_args()

    package = os.path.abspath(a.package)
    if not os.path.isdir(package):
        raise SystemExit(f"No such package directory: {package}")

    files = collect(package)
    total = sum(os.path.getsize(p) for p, _ in files)
    print(f"package : {package}")
    print(f"files   : {len(files)}  ({total/1e9:.2f} GB)")
    if total > 50e9:
        print("\nWARNING: this exceeds Zenodo's standard 50 GB per-record quota.")
        print("Request a quota increase first, or deposit a smaller tier.\n")
    if a.dry_run:
        for p, rel in files[-10:]:
            print(f"  {os.path.getsize(p)/1e6:10.1f} MB  {rel}")
        print("  (ten largest shown)")
        return 0

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        raise SystemExit(
            "ZENODO_TOKEN is not set.\n"
            "  $env:ZENODO_TOKEN = \"<your token>\"\n"
            "Create one at https://zenodo.org/account/settings/applications/tokens/new/\n"
            "with scopes deposit:write and deposit:actions.")

    base = api_base(a.sandbox)

    if a.deposition:
        dep = request(f"{base}/deposit/depositions/{a.deposition}", token)
        print(f"resuming draft {dep['id']}")
    else:
        meta = json.load(open(META, encoding="utf-8"))
        dep = request(f"{base}/deposit/depositions", token, "POST", meta)
        print(f"created draft {dep['id']}")

    bucket = dep["links"]["bucket"]
    existing = {f["filename"]: f.get("checksum", "").replace("md5:", "")
                for f in dep.get("files", [])}

    done = 0
    for path, rel in files:
        size = os.path.getsize(path)
        if rel in existing:
            print(f"  skip (already uploaded)  {rel}")
            done += 1
            continue
        print(f"  uploading {rel}  ({size/1e6:.1f} MB) ...", flush=True)
        request(f"{bucket}/{rel}", token, "PUT", data_file=path)
        done += 1

    url = dep["links"].get("html", f"https://zenodo.org/deposit/{dep['id']}")
    print(f"\n{done}/{len(files)} files in draft {dep['id']}")
    print(f"\nDraft (NOT published): {url}")
    print("\nReview the draft, then publish it yourself from that page.")
    print("Publishing mints a permanent DOI and cannot be undone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
