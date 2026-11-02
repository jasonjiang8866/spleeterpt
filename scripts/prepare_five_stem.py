"""Prepare a pinned research checkpoint; never add it to the released-model catalog."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from build_pretrained_release import unpack_checkpoint

from spleeterpt.checkpoint import read_checkpoint
from spleeterpt.pretrained import digest, verified_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(Path(__file__).with_name("five_stem_source.json").read_text())
    root = args.work.resolve()
    root.mkdir(parents=True, exist_ok=True)
    archive = verified_download(
        source["url"], root / "5stems.tar.gz", sha256=source["sha256"], size=source["bytes"]
    )
    unpack_checkpoint(archive, root / "5stems")
    target = root / "5stems.pt"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "spleeterpt.cli",
            "convert",
            "--checkpoint",
            str(root / "5stems/model"),
            "--stems",
            "5",
            "--output",
            str(target),
        ],
        check=True,
    )
    payload = read_checkpoint(target)
    if payload["config"]["instruments"] != source["instruments"]:
        raise ValueError("Unexpected official stem order")
    (root / "provenance.json").write_text(
        json.dumps(
            {
                "source": source,
                "checkpoint_sha256": digest(target),
                "source_tensor_files": payload["source_sha256"],
                "config": payload["config"],
                "conversion_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "release_eligible": False,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
