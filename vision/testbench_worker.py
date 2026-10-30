"""Persistent local JSON-lines testbench worker. Standard output is protocol only."""
from __future__ import annotations

from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

from vision.testbench import RequestError, compare, decode_image, settings_for, pass_count

ROOT = Path(__file__).resolve().parents[1]
MAX_REQUEST_LINE = 12 * 1024 * 1024


def load_detector():
    from adapters.scout_frame import LocalDetector
    return LocalDetector(ROOT / "models/grounding-dino-tiny", device="cpu", threads=4,
                         box_threshold=.25, text_threshold=.25)


def serve(input_stream, output_stream, detector_factory=load_detector):
    detector = None

    def emit(payload):
        output_stream.write(json.dumps(payload, allow_nan=False, separators=(",", ":")) + "\n")
        output_stream.flush()

    while True:
        line = input_stream.readline(MAX_REQUEST_LINE + 1)
        if not line:
            break
        request_id = None
        oversized = len(line) > MAX_REQUEST_LINE
        try:
            if oversized or not line.endswith(b"\n"):
                raise RequestError("Request line is too long or unterminated.")
            try:
                request = json.loads(line)
            except (ValueError, UnicodeError, RecursionError) as exc:
                raise RequestError("Request is not valid JSON.") from exc
            if not isinstance(request, dict):
                raise RequestError("Request must be an object.")
            candidate_id = request.get("id")
            if not isinstance(candidate_id, str) or not 1 <= len(candidate_id) <= 128 or any(ord(c) < 32 for c in candidate_id):
                raise RequestError("Request ID must contain 1 to 128 printable characters.")
            request_id = candidate_id
            settings = settings_for(request.get("settings", {}))
            image, image_hash = decode_image(request.get("imageBase64"), request.get("mimeType"))
            total = pass_count(image.size, settings)

            def progress(value):
                emit({"id": request_id, "status": "progress", "progress": value})

            # Model/library diagnostics must never become JSON protocol messages.
            with redirect_stdout(sys.stderr):
                if detector is None:
                    progress({"stage": "loading", "completed": 0, "total": total})
                    detector = detector_factory()
                result = compare(image, image_hash, settings, detector, progress)
            emit({"id": request_id, "status": "ok", "result": result})
        except RequestError as exc:
            emit({"id": request_id, "status": "error", "error": str(exc)})
        except (MemoryError, RuntimeError, OSError, ValueError, TypeError, KeyError, AttributeError):
            # Never reveal filesystem paths, arbitrary exception strings or user image bytes.
            emit({"id": request_id, "status": "error", "error": "Local image analysis failed. Check the local model and retry with a smaller photo."})
        if oversized:
            break


def main():
    serve(sys.stdin.buffer, sys.stdout)


if __name__ == "__main__":
    main()
