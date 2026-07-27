"""Tests for .github/scripts/summarise-results.py"""

import importlib.util
import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / ".github" / "scripts" / "summarise-results.py"


def load_script():
    spec = importlib.util.spec_from_file_location("summarise_results", SCRIPT)
    assert spec and spec.loader, f"could not load {SCRIPT}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


summarise = load_script()


def xml_of(text: str) -> ElementTree.Element:
    return ElementTree.fromstring(text)


PYTEST_XML = """
<testsuites>
  <testsuite name="pytest" errors="0" failures="1" skipped="1" tests="4">
    <testcase classname="tests.test_a" file="tests/test_a.py" line="3" name="test_ok"/>
    <testcase classname="tests.test_a" file="tests/test_a.py" line="9" name="test_bad">
      <failure message="assert 1 == 2">long traceback
second line</failure>
    </testcase>
    <testcase classname="tests.test_a" file="tests/test_a.py" line="14" name="test_skipped">
      <skipped message="no reason"/>
    </testcase>
    <testcase classname="tests.test_a" file="tests/test_a.py" line="20" name="test_ok_two"/>
  </testsuite>
</testsuites>
"""

# What vitest writes: the path is in classname and there is no line number.
VITEST_XML = """
<testsuites name="vitest tests" tests="2" failures="1" errors="0">
  <testsuite name="tests/App.test.tsx" tests="2" failures="1" errors="0" skipped="0">
    <testcase classname="tests/App.test.tsx" name="renders"/>
    <testcase classname="tests/App.test.tsx" name="breaks">
      <failure message="expected 1 to be 2"/>
    </testcase>
  </testsuite>
</testsuites>
"""


class TestCountsFor:
    def test_counts_a_pytest_suite(self):
        assert summarise.counts_for(xml_of(PYTEST_XML)) == (4, 1, 1)

    def test_counts_a_vitest_suite(self):
        assert summarise.counts_for(xml_of(VITEST_XML)) == (2, 1, 0)

    def test_errors_count_as_failures(self):
        """A test that errored did not pass, so it belongs with the failures."""
        xml = '<testsuite tests="3" failures="1" errors="1" skipped="0"/>'

        assert summarise.counts_for(xml_of(xml)) == (3, 2, 0)

    def test_missing_attributes_are_zero(self):
        """vitest omits attributes it has nothing to say about."""
        assert summarise.counts_for(xml_of('<testsuite tests="2"/>')) == (2, 0, 0)

    def test_adds_up_several_suites_in_one_file(self):
        xml = """
        <testsuites>
          <testsuite tests="2" failures="1" skipped="0"/>
          <testsuite tests="3" failures="0" skipped="1"/>
        </testsuites>
        """

        assert summarise.counts_for(xml_of(xml)) == (5, 1, 1)


class TestFailuresIn:
    def test_names_the_file_and_line_when_pytest_gives_them(self):
        bullets = summarise.failures_in(xml_of(PYTEST_XML), "python")

        assert len(bullets) == 1
        assert "`test_bad`" in bullets[0]
        assert "`tests/test_a.py:9`" in bullets[0]

    def test_uses_only_the_first_line_of_the_message(self):
        """The rest is a traceback, and the comment links to the full log."""
        bullets = summarise.failures_in(xml_of(PYTEST_XML), "python")

        assert "assert 1 == 2" in bullets[0]
        assert "second line" not in bullets[0]

    def test_falls_back_to_classname_when_there_is_no_file(self):
        bullets = summarise.failures_in(xml_of(VITEST_XML), "web")

        assert "`tests/App.test.tsx`" in bullets[0]
        # No line number is available, so none should be invented.
        assert "tests/App.test.tsx:" not in bullets[0]

    def test_says_which_group_a_failure_came_from(self):
        bullets = summarise.failures_in(xml_of(VITEST_XML), "web")

        assert bullets[0].startswith("- **web**")

    def test_passing_tests_produce_nothing(self):
        xml = '<testsuite tests="1"><testcase name="fine"/></testsuite>'

        assert summarise.failures_in(xml_of(xml), "python") == []

    def test_an_errored_test_is_listed_too(self):
        xml = """
        <testsuite tests="1">
          <testcase name="blew_up"><error message="ImportError"/></testcase>
        </testsuite>
        """

        bullets = summarise.failures_in(xml_of(xml), "python")

        assert len(bullets) == 1
        assert "ImportError" in bullets[0]


class TestMain:
    """main() reads a directory and prints markdown, which is the whole job."""

    @pytest.fixture
    def run_on(self, tmp_path, monkeypatch, capsys):
        def run(files: dict[str, str]) -> str:
            for name, text in files.items():
                (tmp_path / name).write_text(text)
            monkeypatch.setattr(sys, "argv", ["summarise-results.py", str(tmp_path)])

            assert summarise.main() == 0

            return capsys.readouterr().out

        return run

    def test_builds_a_row_per_group(self, run_on):
        out = run_on({"junit-python.xml": PYTEST_XML, "junit-web.xml": VITEST_XML})

        assert "| python | 2/4 passed, 1 failed, 1 skipped |" in out
        assert "| web | 1/2 passed, 1 failed |" in out

    def test_lists_the_failures_under_a_heading(self, run_on):
        out = run_on({"junit-python.xml": PYTEST_XML})

        assert "**Failures**" in out
        assert "`tests/test_a.py:9`" in out

    def test_says_nothing_about_failures_when_all_pass(self, run_on):
        xml = '<testsuite tests="2" failures="0" skipped="0"/>'
        out = run_on({"junit-python.xml": xml})

        assert "| python | 2/2 passed |" in out
        assert "**Failures**" not in out

    def test_prints_nothing_when_there_are_no_files(self, run_on):
        """A run cancelled before any tests started still gets a comment."""
        assert run_on({}) == ""

    def test_a_broken_file_does_not_lose_the_whole_comment(self, run_on):
        out = run_on({"junit-python.xml": "<testsuite", "junit-web.xml": VITEST_XML})

        assert "| python | results unreadable |" in out
        # The group that did work still gets reported.
        assert "| web | 1/2 passed, 1 failed |" in out

    def test_long_failure_lists_are_cut_short(self, run_on):
        cases = "".join(
            f'<testcase name="t{i}"><failure message="boom"/></testcase>' for i in range(25)
        )
        out = run_on(
            {"junit-python.xml": f'<testsuite tests="25" failures="25">{cases}</testsuite>'}
        )

        assert out.count("- **python**") == summarise.MAX_FAILURES_LISTED
        assert "...and 5 more" in out
