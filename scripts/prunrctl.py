#!/usr/bin/env python3
"""Drive a running Prunr through its control socket.

Start the app with PRUNR_CONTROL_PORT=7340 (any free port), then:

    prunrctl.py controls              # every control: role, name, value
    prunrctl.py tree                  # the whole tree as JSON
    prunrctl.py find Settings
    prunrctl.py click Settings        # by readable name; --secondary for right-click
    prunrctl.py key Mod+Shift+Z       # a chord in the settings form
    prunrctl.py type "hello"
    prunrctl.py intent ToggleQueue    # an action by name
    prunrctl.py open photo.png more.jpg
    prunrctl.py state
    prunrctl.py screenshot            # prints the directory the PNG lands in
    prunrctl.py wait 10               # let N frames run

Set PRUNR_CONTROL_PORT (or --port) to match the app. Replies are JSON.
"""

import argparse
import json
import os
import socket
import sys


def send(port: int, request: dict) -> dict:
    with socket.create_connection(("127.0.0.1", port), timeout=600) as s:
        s.sendall((json.dumps(request) + "\n").encode())
        f = s.makefile("r", encoding="utf-8")
        line = f.readline()
    if not line:
        raise SystemExit("error: the app closed the connection")
    return json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=int(os.environ.get("PRUNR_CONTROL_PORT", "7340")))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ping")
    sub.add_parser("controls")
    sub.add_parser("tree")
    sub.add_parser("find").add_argument("name")
    click = sub.add_parser("click")
    click.add_argument("name")
    click.add_argument("--secondary", action="store_true")
    sub.add_parser("key").add_argument("chord")
    sub.add_parser("type").add_argument("text")
    sub.add_parser("intent").add_argument("action")
    sub.add_parser("open").add_argument("paths", nargs="+")
    sub.add_parser("state")
    sub.add_parser("screenshot")
    sub.add_parser("wait").add_argument("frames", type=int, nargs="?", default=1)
    a = ap.parse_args()

    request = {
        "ping": lambda: {"cmd": "ping"},
        "controls": lambda: {"cmd": "tree"},
        "tree": lambda: {"cmd": "tree", "all": True},
        "find": lambda: {"cmd": "find", "name": a.name},
        "click": lambda: {"cmd": "click", "name": a.name, "secondary": a.secondary},
        "key": lambda: {"cmd": "key", "chord": a.chord},
        "type": lambda: {"cmd": "type", "text": a.text},
        "intent": lambda: {"cmd": "intent", "action": a.action},
        "open": lambda: {"cmd": "open", "paths": [os.path.abspath(p) for p in a.paths]},
        "state": lambda: {"cmd": "state"},
        "screenshot": lambda: {"cmd": "screenshot"},
        "wait": lambda: {"cmd": "wait", "frames": a.frames},
    }[a.cmd]()

    reply = send(a.port, request)
    if not reply.get("ok"):
        print(f"error: {reply.get('error')}", file=sys.stderr)
        return 1
    result = reply.get("result")
    if a.cmd == "controls":
        for c in result:
            value = c.get("value") if c.get("value") is not None else c.get("number")
            flags = " (disabled)" if c.get("disabled") else ""
            toggled = "" if c.get("toggled") is None else (" [x]" if c["toggled"] else " [ ]")
            print(f"{c['role']:<12} {c.get('name', '')!s:<40} {'' if value is None else value}{toggled}{flags}")
    else:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
