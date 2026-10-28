"""Serve this folder with the two headers browsers need for multi-threaded WebAssembly (SharedArrayBuffer):
Cross-Origin-Opener-Policy: same-origin and Cross-Origin-Embedder-Policy: credentialless.

    python3 serve.py [PORT]
"""
import functools, http.server, os, sys

class H(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        super().end_headers()

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8811
handler = functools.partial(H, directory=os.path.dirname(os.path.abspath(__file__)))
http.server.ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()
