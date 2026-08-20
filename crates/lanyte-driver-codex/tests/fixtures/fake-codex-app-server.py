#!/usr/bin/env python3
import json
import sys


def main() -> None:
    if "--version" in sys.argv:
        sys.stdout.write("fake-codex 0.0.1\n")
        return
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        message = json.loads(line)
        method = message.get("method")
        if method == "initialize":
            sys.stdout.write(
                json.dumps({"id": message["id"], "result": {"protocolVersion": "1"}}) + "\n"
            )
            sys.stdout.flush()
        elif method == "thread/start":
            sys.stdout.write(
                json.dumps({"id": message["id"], "result": {"thread": {"id": "thr_test"}}})
                + "\n"
            )
            sys.stdout.write(
                json.dumps(
                    {
                        "method": "turn/started",
                        "params": {"threadId": "thr_test", "turnId": "turn_test"},
                    }
                )
                + "\n"
            )
            sys.stdout.write(
                json.dumps(
                    {
                        "method": "item/started",
                        "params": {"item": {"type": "command_execution"}, "name": "shell"},
                    }
                )
                + "\n"
            )
            sys.stdout.flush()
        elif method == "turn/interrupt":
            params = message.get("params") or {}
            sys.stdout.write(json.dumps({"id": message["id"], "result": {}}) + "\n")
            sys.stdout.write(
                json.dumps(
                    {
                        "method": "turn/completed",
                        "params": {
                            "threadId": params.get("threadId"),
                            "turnId": params.get("turnId"),
                            "status": "interrupted",
                        },
                    }
                )
                + "\n"
            )
            sys.stdout.flush()
        elif method == "thread/unsubscribe":
            sys.stdout.write(
                json.dumps({"method": "thread/exited", "params": {"threadId": "thr_test"}})
                + "\n"
            )
            sys.stdout.flush()
            return


if __name__ == "__main__":
    main()
