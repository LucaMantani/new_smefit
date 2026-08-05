"""Reproduce the Codecov *patch* status locally.

``codecov.yml`` requires 100% coverage on the lines a pull request adds or
modifies, which is the check that usually fails in CI over one or two stray
lines.  This script answers the same question before pushing: run the test
suite under coverage, diff the working tree against the merge base with
``main``, and report every added/modified statement that no test executed.

Typical use::

    python scripts/check_patch_coverage.py             # run pytest, then check
    python scripts/check_patch_coverage.py --no-run    # reuse existing .coverage
    python scripts/check_patch_coverage.py -m "not slow"

Note that skipping tests (``-m "not slow"``) makes the check *stricter* than
CI: lines only reached by the skipped tests will be reported as uncovered.
"""

import argparse
import pathlib
import re
import subprocess
import sys

import coverage

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
HUNK_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def git(*args):
    """Run a git command in the repository and return its stdout."""
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def merge_base(base):
    """Resolve the commit the branch diverged from, trying remote then local."""
    for ref in (base, base.split("/")[-1]):
        try:
            return git("merge-base", ref, "HEAD").strip()
        except subprocess.CalledProcessError:
            continue
    sys.exit(f"cannot resolve base ref '{base}' - fetch it or pass --base")


def changed_lines(since):
    """Map each changed Python file to the set of lines it adds or modifies."""
    diff = git("diff", "--unified=0", since, "--", "*.py")
    changed = {}
    lines = None
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            lines = changed.setdefault(line[len("+++ b/") :], set())
        elif line.startswith("@@"):
            match = HUNK_HEADER.match(line)
            if match is not None and lines is not None:
                start = int(match.group(1))
                count = 1 if match.group(2) is None else int(match.group(2))
                lines.update(range(start, start + count))
    return {path: nums for path, nums in changed.items() if nums}


def run_tests(pytest_args):
    """Run the test suite under coverage, writing a .coverage data file."""
    command = [
        sys.executable,
        "-m",
        "pytest",
        "--cov=smefit",
        "--cov-report=",
        *pytest_args,
    ]
    print(f"$ {' '.join(command)}\n")
    if subprocess.run(command, cwd=REPO_ROOT).returncode != 0:
        sys.exit("tests failed - fix them before checking patch coverage")


def uncovered(changed, cov):
    """Intersect the changed lines with the statements coverage saw missed."""
    measured = {pathlib.Path(f).resolve() for f in cov.get_data().measured_files()}
    report = {}
    for path, nums in sorted(changed.items()):
        absolute = (REPO_ROOT / path).resolve()
        if not absolute.exists():
            continue
        try:
            _, statements, _, missing, _ = cov.analysis2(str(absolute))
        except coverage.misc.CoverageException:
            continue
        # A file with no data at all is either omitted by .coveragerc or simply
        # never imported; only the latter counts as untested new code.
        if absolute not in measured and not set(statements) & nums:
            continue
        missed = sorted(set(missing) & nums)
        if missed:
            report[path] = missed
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        default="origin/main",
        help="branch the patch is measured against (default: origin/main)",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="reuse the existing .coverage file instead of running pytest",
    )
    parser.add_argument(
        "pytest_args",
        nargs="*",
        help="extra arguments forwarded to pytest, e.g. -- -m 'not slow'",
    )
    args = parser.parse_args()

    since = merge_base(args.base)
    changed = changed_lines(since)
    if not changed:
        print(f"no Python changes against {args.base} - nothing to check")
        return 0

    if not args.no_run:
        run_tests(args.pytest_args)

    cov = coverage.Coverage()
    cov.load()
    report = uncovered(changed, cov)

    total = sum(len(nums) for nums in changed.values())
    if not report:
        print(f"patch coverage OK: all changed lines in {len(changed)} file(s) covered")
        return 0

    missed = sum(len(nums) for nums in report.values())
    print(f"\nuncovered lines added against {args.base} ({missed} of {total}):\n")
    for path, nums in report.items():
        print(f"  {path}: {format_ranges(nums)}")
    print("\nCodecov requires 100% patch coverage - add tests for the lines above.")
    return 1


def format_ranges(nums):
    """Collapse a sorted line list into compact ``12-15, 20`` style ranges."""
    ranges = []
    start = previous = nums[0]
    for num in nums[1:]:
        if num == previous + 1:
            previous = num
            continue
        ranges.append((start, previous))
        start = previous = num
    ranges.append((start, previous))
    return ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in ranges)


if __name__ == "__main__":
    sys.exit(main())
