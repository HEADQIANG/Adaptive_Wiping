"""Bounded event recording with file I/O owned by a background thread."""

import queue
import sys
import threading


class EventWriter:
    def __init__(self, path):
        self._queue = queue.Queue(maxsize=1024)
        self._stop = threading.Event()
        self.error = None
        self._file = path.open("x", encoding="utf-8")
        self._worker = threading.Thread(target=self._run, name="deployment-events", daemon=True)
        try:
            self._worker.start()
        except BaseException:
            self._file.close()
            raise

    def write(self, text):
        self.flush()
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            self.error = RuntimeError("Event queue full; recording incomplete")
            raise self.error
        return len(text)

    def flush(self):
        """Non-blocking health check; the worker flushes accepted events."""
        if self.error is not None:
            raise self.error
        if self._stop.is_set() or not self._worker.is_alive():
            raise RuntimeError("Event writer stopped; recording incomplete")

    def _run(self):
        try:
            while not self._stop.is_set() or not self._queue.empty():
                try:
                    text = self._queue.get(timeout=0.01)
                except queue.Empty:
                    continue
                self._file.write(text)
                self._file.flush()
        except BaseException as exc:
            if self.error is None:
                self.error = exc
        finally:
            try:
                self._file.close()
            except BaseException as exc:
                if self.error is None:
                    self.error = exc

    def close(self):
        self._stop.set()
        self._worker.join(timeout=2.0)
        if self._worker.is_alive():
            raise RuntimeError("Event writer shutdown timed out; recording incomplete")
        if self.error is not None:
            raise self.error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except Exception as cleanup:
            if exc_type is None:
                raise
            print(f"Event cleanup failed; recording may be incomplete: {cleanup}", file=sys.stderr)
