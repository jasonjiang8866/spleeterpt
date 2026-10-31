"""Bounded streaming HTTPS transfer; account credential never enters the Pod."""

import http.client
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from recovery_result import digest, validate_bundle


def download_bundle(url, token, descriptor, output, *, complete):
    from control import NoRedirect

    output = Path(output)
    temporary = output.with_suffix(".part")
    try:
        with temporary.open("wb") as dest:
            for start in range(0, descriptor["bytes"], 8 * 1024 * 1024):
                end = min(start + 8 * 1024 * 1024, descriptor["bytes"]) - 1
                req = urllib.request.Request(
                    url,
                    headers={
                        "Authorization": "Bearer " + token,
                        "User-Agent": "spleeterpt-runpod-actions/0.1",
                        "Content-Type": "application/json",
                        "Range": f"bytes={start}-{end}",
                    },
                )
                try:
                    response = urllib.request.build_opener(NoRedirect).open(req, timeout=120)
                except urllib.error.HTTPError as error:
                    raise RuntimeError(f"Checkpoint download HTTP {error.code}") from None
                with response:
                    if (
                        response.status != 206
                        or int(response.headers.get("Content-Length", "-1")) != end - start + 1
                        or response.headers.get("Content-Range")
                        != f"bytes {start}-{end}/{descriptor['bytes']}"
                    ):
                        raise RuntimeError("Checkpoint response range mismatch")
                    data = response.read(end - start + 2)
                    if len(data) != end - start + 1:
                        raise RuntimeError("Checkpoint response truncated or oversized")
                    dest.write(data)
        validate_bundle(temporary, descriptor, complete=complete)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def upload_bundle(url, token, path, descriptor):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname.endswith(".proxy.runpod.net"):
        raise RuntimeError("Unexpected restore destination")
    validate_bundle(path, descriptor, complete=False)
    connection = http.client.HTTPSConnection(parsed.hostname, timeout=120)
    try:
        offset = 0
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                connection.putrequest("PUT", parsed.path)
                connection.putheader("Authorization", "Bearer " + token)
                connection.putheader("User-Agent", "spleeterpt-runpod-actions/0.1")
                connection.putheader("Content-Type", "application/octet-stream")
                connection.putheader("Content-Length", str(len(chunk)))
                connection.putheader("X-Offset", str(offset))
                connection.endheaders(chunk)
                response = connection.getresponse()
                if response.status != 201:
                    raise RuntimeError(f"Restore upload HTTP {response.status}")
                reply = json.loads(response.read(1024))
                offset += len(chunk)
                if reply.get("received") != offset:
                    raise RuntimeError("Restore upload offset mismatch")
        if reply.get("sha256") != digest(path):
            raise RuntimeError("Restore upload checksum mismatch")
    finally:
        connection.close()
