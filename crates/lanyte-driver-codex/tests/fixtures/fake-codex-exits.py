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
            sys.stdout.flush()
            return


if __name__ == "__main__":
    main()
