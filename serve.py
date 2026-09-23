"""Serve one policy over RoboFollow's length-prefixed TCP protocol.

The server processes one client at a time; run one instance per evaluation
worker when using stateful models.
"""
import argparse
import json
import socketserver
import traceback

from .policies import ValidatedPolicy, load_factory
from .remote_policy import json_to_numpy, numpy_to_json


class Handler(socketserver.BaseRequestHandler):
    def receive(self, length):
        data = bytearray()
        while len(data) < length:
            chunk = self.request.recv(length - len(data))
            if not chunk:
                raise EOFError
            data.extend(chunk)
        return bytes(data)

    def handle(self):
        policy = self.server.policy
        while True:
            try:
                size = int.from_bytes(self.receive(4), "big")
                if not 0 < size <= 128 * 1024 * 1024:
                    return
                request = json_to_numpy(self.receive(size).decode())
            except EOFError:
                return
            try:
                command, obs = request["cmd"], request.get("obs")
                if command == "reset":
                    value = policy.reset()
                elif command == "set_instruction":
                    value = policy.set_instruction(obs)
                elif command == "predict":
                    value = policy.predict(obs)
                else:
                    raise ValueError(f"Unknown policy command: {command}")
                response = {"res": value}
            except Exception as exc:
                response = {"error": str(exc), "traceback": traceback.format_exc()}
            raw = numpy_to_json(response).encode()
            self.request.sendall(len(raw).to_bytes(4, "big") + raw)


class Server(socketserver.TCPServer):
    allow_reuse_address = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory", required=True)
    parser.add_argument("--kwargs", default="{}")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8999)
    args = parser.parse_args()
    policy = ValidatedPolicy(load_factory(args.factory, json.loads(args.kwargs)))
    try:
        with Server((args.host, args.port), Handler) as server:
            server.policy = policy
            print(f"Policy server: {args.host}:{server.server_address[1]}", flush=True)
            server.serve_forever()
    finally:
        policy.close()


if __name__ == "__main__":
    main()
