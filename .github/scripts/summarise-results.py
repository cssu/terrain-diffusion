"""Turns the test result files into the markdown that goes in the reply comment.

Both pytest and vitest write their results as JUnit XML,
the workflow collects them, and this reads and prints a summary.

Usage:
    python3 summarise-results.py <directory of xml files>
"""

import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path

MAX_FAILURES_LISTED = 20


def counts_for(root: ElementTree.Element) -> tuple[int, int, int]:
    """Total, failed and skipped, added up over every suite in one file."""
    total = failed = skipped = 0

    suites = root.iter("testsuite")

    for suite in suites:
        total += int(suite.get("tests", 0))
        failed += int(suite.get("failures", 0)) + int(suite.get("errors", 0))
        skipped += int(suite.get("skipped", 0))

    return total, failed, skipped


def failures_in(root: ElementTree.Element, group: str) -> list[str]:
    """One markdown bullet per failing test, saying where it lives."""
    bullets = []

    for case in root.iter("testcase"):
        problem = case.find("failure")
        if problem is None:
            problem = case.find("error")
        if problem is None:
            continue

        where = case.get("file") or case.get("classname") or ""
        line = case.get("line")
        if where and line:
            where = f"{where}:{line}"

        name = case.get("name", "unknown test")
        location = f" — `{where}`" if where else ""

        message = (problem.get("message") or "").strip().splitlines()
        summary = f" — {message[0]}" if message else ""

        bullets.append(f"- **{group}** `{name}`{location}{summary}")

    return bullets


def main() -> int:
    directory = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    files = sorted(directory.rglob("junit-*.xml"))

    if not files:
        return 0

    rows = []
    failures: list[str] = []

    for path in files:
        group = path.stem.removeprefix("junit-")

        try:
            root = ElementTree.parse(path).getroot()
        except ElementTree.ParseError:
            # A half written file means the job died mid-run. Say so rather
            # than losing the whole comment to an exception.
            rows.append(f"| {group} | results unreadable |")
            continue

        total, failed, skipped = counts_for(root)
        passed = total - failed - skipped

        parts = [f"{passed}/{total} passed"]
        if failed:
            parts.append(f"{failed} failed")
        if skipped:
            parts.append(f"{skipped} skipped")

        rows.append(f"| {group} | {', '.join(parts)} |")
        failures.extend(failures_in(root, group))

    print("| Group | Result |")
    print("| --- | --- |")
    for row in rows:
        print(row)

    if failures:
        print()
        print("**Failures**")
        print()
        for bullet in failures[:MAX_FAILURES_LISTED]:
            print(bullet)

        remaining = len(failures) - MAX_FAILURES_LISTED
        if remaining > 0:
            print(f"- ...and {remaining} more, see the full log")

    return 0


if __name__ == "__main__":
    sys.exit(main())
