"""Remote policy client for RobotWin batch evaluation.

This module keeps the local evaluator and simulator lightweight: it sends the
minimal RobotWin observation needed by a policy server and receives an action
chunk shaped like (T, 14).
"""

from __future__ import annotations

import base64
import json
import socket
import time
from typing import Any

import numpy as np


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that preserves numpy arrays without pickle."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                "__numpy_array__": True,
                "data": base64.b64encode(obj.tobytes()).decode("ascii"),
                "dtype": str(obj.dtype),
                "shape": obj.shape,
            }
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def numpy_to_json(data: Any) -> str:
    return json.dumps(data, cls=NumpyJSONEncoder)


def json_to_numpy(raw: str) -> Any:
    def object_hook(dct):
        if dct.get("__numpy_array__"):
            buf = base64.b64decode(dct["data"])
            return np.frombuffer(buf, dtype=np.dtype(dct["dtype"])).reshape(dct["shape"])
        return dct

    return json.loads(raw, object_hook=object_hook)


class RemoteModelClient:
    """Length-prefixed TCP client compatible with RobotWin policy_model_server.py."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8999,
        timeout: float = 120.0,
        connect_retries: int = 20,
        retry_delay: float = 1.0,
    ):
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)
        self.connect_retries = int(connect_retries)
        self.retry_delay = float(retry_delay)
        self.sock: socket.socket | None = None
        self._connect()

    def _connect(self) -> None:
        last_error = None
        for attempt in range(1, self.connect_retries + 1):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self.timeout)
                sock.connect((self.host, self.port))
                self.sock = sock
                print(f"[RemotePolicy] connected to {self.host}:{self.port}")
                return
            except OSError as exc:
                last_error = exc
                try:
                    sock.close()
                except Exception:
                    pass
                if attempt < self.connect_retries:
                    print(
                        f"[RemotePolicy] connect attempt {attempt}/{self.connect_retries} "
                        f"failed: {exc}; retrying in {self.retry_delay}s"
                    )
                    time.sleep(self.retry_delay)

        raise ConnectionError(f"Failed to connect to remote policy at {self.host}:{self.port}: {last_error}")

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None

    def _recv_exact(self, n_bytes: int) -> bytes:
        if self.sock is None:
            raise ConnectionError("Remote policy socket is closed")

        chunks = []
        remaining = n_bytes
        while remaining > 0:
            chunk = self.sock.recv(remaining)
            if not chunk:
                raise ConnectionError("Remote policy connection closed while receiving data")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def call(self, command: str, obs: Any | None = None) -> Any:
        if self.sock is None:
            self._connect()

        request = numpy_to_json({"cmd": command, "obs": obs}).encode("utf-8")
        if len(request) > 128 * 1024 * 1024:
            raise ValueError("Remote policy request exceeds 128 MiB")
        # A lost reply does not mean the server did not execute the command.
        # Replaying predict can advance a stateful policy twice or lose the
        # episode's instruction after a server restart. Only reset is safe.
        return self._call_encoded(request, command, allow_retry=command == "reset")

    def _call_encoded(self, request: bytes, command: str, allow_retry: bool) -> Any:
        try:
            assert self.sock is not None
            self.sock.sendall(len(request).to_bytes(4, "big"))
            self.sock.sendall(request)

            response_len = int.from_bytes(self._recv_exact(4), "big")
            if not 0 < response_len <= 128 * 1024 * 1024:
                self.close()
                raise ValueError(f"Invalid remote policy response length: {response_len}")
            response = json_to_numpy(self._recv_exact(response_len).decode("utf-8"))
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, ConnectionError, OSError) as exc:
            self.close()
            if allow_retry:
                print(f"[RemotePolicy] connection lost during '{command}' ({exc}); reconnecting once")
                self._connect()
                return self._call_encoded(request, command, allow_retry=False)
            raise ConnectionError(
                f"Remote policy connection failed during '{command}'. "
                "Check the server log for a traceback or protocol/method-name mismatch."
            ) from exc

        if isinstance(response, dict) and "error" in response:
            tb = response.get("traceback", "")
            raise RuntimeError(f"Remote policy server error: {response['error']}\n{tb}")
        if not isinstance(response, dict) or "res" not in response:
            raise RuntimeError(f"Invalid remote policy response: {response!r}")
        return response["res"]


class RemotePolicy:
    """TCP adapter implementing the RoboFollow policy interface."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8999,
        timeout: float = 120.0,
        connect_retries: int = 20,
        retry_delay: float = 1.0,
        reset_command: str = "reset",
        set_instruction_command: str = "set_instruction",
        predict_command: str = "predict",
    ):
        self.client = RemoteModelClient(
            host=host,
            port=port,
            timeout=timeout,
            connect_retries=connect_retries,
            retry_delay=retry_delay,
        )
        self.reset_command = reset_command
        self.set_instruction_command = set_instruction_command
        self.predict_command = predict_command
        self.instruction = ""

    def reset(self):
        self.instruction = ""
        return self.client.call(self.reset_command)

    def set_instruction(self, text):
        self.instruction = text or ""
        return self.client.call(self.set_instruction_command, self.instruction)

    def predict(self, obs):
        # Keep this payload structurally close to the local RobotWin observation,
        # while omitting unused depth/segmentation fields to keep network traffic low.
        payload = {
            "observation": {
                "head_camera": {"rgb": obs["observation"]["head_camera"]["rgb"]},
                "right_camera": {"rgb": obs["observation"]["right_camera"]["rgb"]},
                "left_camera": {"rgb": obs["observation"]["left_camera"]["rgb"]},
            },
            "joint_action": {
                "vector": obs["joint_action"]["vector"],
            },
            "instruction": self.instruction,
        }
        actions = self.client.call(self.predict_command, payload)
        return np.asarray(actions)

    def close(self):
        self.client.close()
