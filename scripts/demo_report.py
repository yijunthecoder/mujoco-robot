#!/usr/bin/env python3
"""One line (plus details) per zebra run log: full zebra or not, how long, each flip
(how it was done, how long the planner thought), how far off each brick was placed,
and every failure or retry. Used by scripts/check_demos.sh; works on any saved log
of scripts/run_zebra.sh too.

usage: python3 scripts/demo_report.py LOG...
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def report(path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="replace")
    stamps = [float(t) for t in re.findall(r"\[(\d+\.\d+)\] \[zebra_skill_bridge\]: (?:<-|->)", text)]
    minutes = (stamps[-1] - stamps[0]) / 60 if len(stamps) > 1 else 0.0
    start = re.search(r"(place|drop|upright) seed (\d+)", text)
    totals = re.findall(r"(\d+) placed, (\d+) escalated, (\d+) total", text)
    full = "Zebra fully assembled" in text or bool(totals and totals[-1][0] == totals[-1][2])
    result = ("FULL ZEBRA" if full else
              f"{totals[-1][0]} placed, {totals[-1][1]} escalated" if totals else "did not finish")
    print(f"{path.stem}{f' (seed {start.group(2)})' if start else ''}: {result} in {minutes:.1f} min")
    for m in re.finditer(r"flip plan \(([\d.]+) s\): (\w+): (.*)", text):
        print(f"   flip {m.group(2):11} planned in {float(m.group(1)):4.1f} s (sim): {m.group(3)[:75]}")
    places = re.findall(r"arm placed \w+ \(checked with \w+: ([\d.]+) cm to the side", text)
    if places:
        print("   placed " + ", ".join(f"{p} cm" for p in places) + " off target")
    for m in re.finditer(r"\[mjrobots\] (.*turns the brick upright, \d+ cm away from the stack)", text):
        print(f"   note: {m.group(1)}")
    for m in re.finditer(r"zebra_skill_bridge\]: (\w+ zebra-\d+ failed: .*)", text):
        print(f"   PROBLEM: {m.group(1)[:150]}")
    return full


if __name__ == "__main__":
    logs = [Path(p) for p in sys.argv[1:]]
    full = [report(p) for p in logs]
    print(f"\n{sum(full)} of {len(logs)} built a full zebra")
    sys.exit(0 if all(full) else 1)
