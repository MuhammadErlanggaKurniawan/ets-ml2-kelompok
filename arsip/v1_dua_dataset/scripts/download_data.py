"""Acquisition step: download the two public datasets and verify their checksums.

Sources
- AqSolDB (Sorkun et al., 2019, Sci. Data 6:143), mirror: huggingface.co/datasets/maomlab/AqSolDB (MIT)
- Beef e-nose (Wijaya et al., 2018, Data in Brief 21:2414-2420), Mendeley Data doi:10.17632/mwmhh766fc (CC BY 4.0)
"""
import hashlib
import json
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path

import certifi

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

ENOSE_URL = ("https://data.mendeley.com/public-files/datasets/mwmhh766fc/files/"
             "2e65305f-5e53-4395-9b28-91dfc73ff490/file_downloaded")
ENOSE_SHA256 = "772bc2c75f2ac4b785538ea38311bdefaee65059b3391b080b4da5ea619963e3"
AQSOL_API = "https://huggingface.co/api/datasets/maomlab/AqSolDB/tree/main?recursive=true"
CTX = ssl.create_default_context(cafile=certifi.where())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        print(f"downloading {dest.name}")
        req = urllib.request.Request(url, headers={"User-Agent": "ets-ml2"})
        with urllib.request.urlopen(req, context=CTX) as r, open(dest, "wb") as f:
            f.write(r.read())
    return dest


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = {}

    zpath = fetch(ENOSE_URL, RAW / "enose_beef.zip")
    if sha256(zpath) != ENOSE_SHA256:
        sys.exit("e-nose checksum mismatch")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(RAW / "enose_beef")
    manifest["enose_beef.zip"] = ENOSE_SHA256

    with urllib.request.urlopen(AQSOL_API, context=CTX) as r:
        tree = json.load(r)
    csvs = [t["path"] for t in tree if t["path"].endswith(".csv")]
    for p in csvs:
        dest = fetch(f"https://huggingface.co/datasets/maomlab/AqSolDB/resolve/main/{p}",
                     RAW / f"aqsoldb_{Path(p).name}")
        manifest[dest.name] = sha256(dest)

    (RAW / "checksums_sha256.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
