"""Bounded streaming HTTPS transfer; account credential never enters the Pod."""

import http.client
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

from recovery_result import MAX_BUNDLE, digest, validate_bundle


def download_bundle(url, token, descriptor, output, *, complete):
    from control import NoRedirect

    output = Path(output)
    temporary = output.with_suffix(".part")
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + token})
    try:
        with (
            urllib.request.build_opener(NoRedirect).open(req, timeout=120) as response,
            temporary.open("wb") as dest,
        ):
            if int(response.headers.get("Content-Length", "-1")) != descriptor["bytes"]:
                raise RuntimeError("Checkpoint response size mismatch")
            received = 0
            while b := response.read(1024 * 1024):
                received += len(b)
                if received > descriptor["bytes"] or received > MAX_BUNDLE:
                    raise RuntimeError("Checkpoint exceeded transfer limit")
                dest.write(b)
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
                connection.putheader("Content-Length", str(len(chunk)))
                connection.putheader("X-Offset", str(offset))
                connection.endheaders(chunk)
                response = connection.getresponse()
                if response.status != 201:
                    raise RuntimeError("Restore upload rejected")
                reply = json.loads(response.read(1024))
                offset += len(chunk)
                if reply.get("received") != offset:
                    raise RuntimeError("Restore upload offset mismatch")
        if reply.get("sha256") != digest(path):
            raise RuntimeError("Restore upload checksum mismatch")
    finally:
        connection.close()
