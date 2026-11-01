"""Finished failed workers may export evidence; running/unauthenticated requests may not."""

import base64
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

import worker


class Tests(unittest.TestCase):
    def test_finished_export_retained_after_validation_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "checkpoint.tar"
            data = b"bounded completed export"
            path.write_bytes(data)
            with patch.object(worker, "TOKEN", "x" * 32), patch.object(worker, "EXPORT_FILE", path):
                server = worker.ThreadingHTTPServer(("127.0.0.1", 0), worker.Handler)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    url = f"http://127.0.0.1:{server.server_port}/checkpoint?offset=0&length={len(data)}"

                    def get(token="x" * 32):
                        request = urllib.request.Request(
                            url, headers={"Authorization": "Bearer " + token}
                        )
                        with urllib.request.urlopen(request, timeout=3) as response:
                            return json.load(response)

                    with patch.object(worker, "RESULT", {"status": "running"}):
                        with self.assertRaises(urllib.error.HTTPError) as caught:
                            get()
                        self.assertEqual(caught.exception.code, 404)
                    for status in ("passed", "failed"):
                        with patch.object(worker, "RESULT", {"status": status}):
                            self.assertEqual(base64.b64decode(get()["data"]), data)
                            with self.assertRaises(urllib.error.HTTPError) as caught:
                                get("wrong")
                            self.assertEqual(caught.exception.code, 401)
                finally:
                    server.shutdown()
                    thread.join(timeout=3)
                    server.server_close()


if __name__ == "__main__":
    unittest.main()
