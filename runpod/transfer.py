"""Bounded streaming HTTPS transfer; account credential never enters the Pod."""

import base64
import hashlib
import http.client
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from recovery_result import digest, validate_bundle

PROBE_BYTES = b"spleeterpt-transfer-probe" * 512


def read_chunk(url, token, offset, length):
    from control import request

    separator = "&" if "?" in url else "?"
    reply = request(
        url + separator + urllib.parse.urlencode({"offset": offset, "length": length}), token=token
    )
    try:
        data = base64.b64decode(reply["data"], validate=True)
    except (ValueError, KeyError) as error:
        raise RuntimeError("Invalid checkpoint chunk encoding") from error
    if (
        reply.get("offset") != offset
        or reply.get("bytes") != length
        or len(data) != length
        or hashlib.sha256(data).hexdigest() != reply.get("sha256")
    ):
        raise RuntimeError("Checkpoint chunk offset, size or checksum mismatch")
    return data


def probe_transfer(url, token):
    if read_chunk(url + "?probe=1", token, 0, len(PROBE_BYTES)) != PROBE_BYTES:
        raise RuntimeError("Checkpoint transport probe failed")


def download_bundle(url, token, descriptor, output, *, complete):
    output = Path(output)
    temporary = output.with_suffix(".part")
    try:
        with temporary.open("wb") as dest:
            for offset in range(0, descriptor["bytes"], 4 * 1024 * 1024):
                length = min(4 * 1024 * 1024, descriptor["bytes"] - offset)
                dest.write(read_chunk(url, token, offset, length))
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
                connection.putrequest("PUT", parsed.path + "?offset=" + str(offset))
                connection.putheader("Authorization", "Bearer " + token)
                connection.putheader("User-Agent", "spleeterpt-runpod-actions/0.1")
                connection.putheader("Content-Type", "application/octet-stream")
                connection.putheader("Content-Length", str(len(chunk)))
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
