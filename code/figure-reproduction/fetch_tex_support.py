#!/usr/bin/env python3
"""Download the exact publisher-owned files required by the paper build.

Run explicitly when internet access is available. Existing files are verified
and never overwritten. Downloaded files retain the publisher's license notices.
"""
from pathlib import Path
import argparse
import hashlib
import io
import urllib.request
import zipfile

URL = "https://cms-resources.apps.public.k8s.springernature.io/springer-cms/rest/v1/content/18782940/data/v12"
ARCHIVE_SHA256 = "812e76dcaa9c28dc1bff1fb6065d51729b67d4ea140552a05088317414a3ecae"
FILES = {
    "sn-jnl.cls": ("sn-article-template/sn-jnl.cls", "36d0c3273a59d48dc6a9c7b080dfa1ec50dc10229d8751568d1f2e490ffa5ecc"),
    "sn-nature.bst": ("sn-article-template/bst/sn-nature.bst", "638d68f5b0e92ffb25d64d29c7334e1bac6bbf54574b602bc633af8d96e1fd9b"),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "paper")
    args = parser.parse_args()
    paper = args.paper_dir.resolve()
    if not paper.is_dir():
        raise SystemExit(f"Paper directory does not exist: {paper}")
    missing = []
    for name, (_, expected) in FILES.items():
        target = paper / name
        if target.exists():
            if target.is_symlink() or digest(target.read_bytes()) != expected:
                raise SystemExit(f"Refusing to overwrite an existing or mismatched file: {target}")
        else:
            missing.append(name)
    if not missing:
        print("Both publisher support files already match their recorded hashes.")
        return
    with urllib.request.urlopen(URL, timeout=30) as response:
        archive = response.read(2_000_001)
    if len(archive) > 2_000_000 or digest(archive) != ARCHIVE_SHA256:
        raise SystemExit("Publisher archive checksum mismatch; no files were written.")
    verified = {}
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        for name in missing:
            member, expected = FILES[name]
            content = bundle.read(member)
            if digest(content) != expected:
                raise SystemExit(f"Checksum mismatch for {name}; no files were written.")
            verified[name] = content
    for name, content in verified.items():
        with (paper / name).open("xb") as target:
            target.write(content)
        print(f"Installed and verified {name}")
    print("These files retain their original LPPL terms; the author CC BY 4.0/MIT licenses do not apply to them.")


if __name__ == "__main__":
    main()
