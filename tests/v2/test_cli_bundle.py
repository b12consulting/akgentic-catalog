"""Tests for the v2 ``ak-catalog`` CLI bundle verbs — Story 17.3.

Exercises ``export``, ``import`` and ``import --dry-run`` through
:class:`typer.testing.CliRunner` against a YAML-backed tmp-dir ``Catalog``
seeded with a self-contained namespace (team + agent + tool, agent
referencing the tool).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from akgentic.catalog.cli import main as cli_main

from .conftest import base_args as _base_args
from .conftest import seed_namespace

_FIXTURE_MODULE = "akgentic.catalog.tests_fixture_17_3"

pytestmark = pytest.mark.usefixtures("cli_fixture_models")


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def catalog_root(tmp_path: Path, cli_fixture_models: str) -> Path:
    root = tmp_path / "catalog"
    root.mkdir()
    seed_namespace(root, fixture_module=_FIXTURE_MODULE)
    return root


# --------------------------------------------------------------------------- #
# `export` verb
# --------------------------------------------------------------------------- #


class TestExportVerb:
    """AC3-AC6 + AC23 — export verb shape, stdout fidelity, error paths."""

    def test_export_stdout_is_parseable_bundle(self, runner: CliRunner, catalog_root: Path) -> None:
        result = runner.invoke(
            cli_main.app, _base_args(catalog_root) + ["export", "--namespace", "ns-a"]
        )
        assert result.exit_code == 0, result.stderr
        payload = yaml.safe_load(result.stdout)
        # Story 18.2 — eight top-level keys including the header (adds
        # ``public`` after ``shareable``).
        assert set(payload.keys()) == {
            "namespace",
            "user_id",
            "name",
            "description",
            "properties",
            "shareable",
            "public",
            "entries",
        }
        assert payload["namespace"] == "ns-a"
        assert "team-a" in payload["entries"]
        assert "tool-a" in payload["entries"]
        assert "agent-a" in payload["entries"]

    def test_export_missing_namespace_is_usage_error(
        self, runner: CliRunner, catalog_root: Path
    ) -> None:
        result = runner.invoke(cli_main.app, _base_args(catalog_root) + ["export"])
        assert result.exit_code == 2
        assert result.stdout == ""

    def test_export_empty_namespace_string_is_usage_error(
        self, runner: CliRunner, catalog_root: Path
    ) -> None:
        result = runner.invoke(
            cli_main.app, _base_args(catalog_root) + ["export", "--namespace", ""]
        )
        assert result.exit_code == 2

    def test_export_unknown_namespace_is_validation_error(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        # AC23 — point at an empty catalog dir; exporting any namespace fails.
        root = tmp_path / "catalog"
        root.mkdir()
        result = runner.invoke(
            cli_main.app,
            ["--backend", "yaml", "--root", str(root), "export", "--namespace", "nope"],
        )
        assert result.exit_code == 1
        assert "validation error:" in result.stderr

    def test_export_format_flag_ignored(self, runner: CliRunner, catalog_root: Path) -> None:
        # --format is a no-op on export; bundle is always YAML bytes.
        result_yaml = runner.invoke(
            cli_main.app,
            _base_args(catalog_root) + ["--format", "json", "export", "--namespace", "ns-a"],
        )
        assert result_yaml.exit_code == 0
        # Output must still parse as YAML (the bundle format).
        payload = yaml.safe_load(result_yaml.stdout)
        assert payload["namespace"] == "ns-a"


# --------------------------------------------------------------------------- #
# `import` verb — persistence mode
# --------------------------------------------------------------------------- #


class TestImportPersistence:
    """AC7-AC11 + AC17-AC18 — round-trip + atomicity."""

    def test_round_trip_mutation(
        self, runner: CliRunner, catalog_root: Path, tmp_path: Path
    ) -> None:
        # AC17 — export → edit → import → re-export → verify.
        export = runner.invoke(
            cli_main.app, _base_args(catalog_root) + ["export", "--namespace", "ns-a"]
        )
        assert export.exit_code == 0
        bundle = yaml.safe_load(export.stdout)
        assert bundle["namespace"] == "ns-a"
        bundle["entries"]["team-a"]["description"] = "edited description"
        bundle_path = tmp_path / "bundle.yaml"
        bundle_path.write_text(yaml.safe_dump(bundle, sort_keys=False))

        imp = runner.invoke(cli_main.app, _base_args(catalog_root) + ["import", str(bundle_path)])
        assert imp.exit_code == 0, imp.stderr
        assert imp.stdout == ""
        assert "imported" in imp.stderr
        assert "ns-a" in imp.stderr

        re_export = runner.invoke(
            cli_main.app, _base_args(catalog_root) + ["export", "--namespace", "ns-a"]
        )
        assert re_export.exit_code == 0
        re_bundle = yaml.safe_load(re_export.stdout)
        assert re_bundle["entries"]["team-a"]["description"] == "edited description"
        # Other entries untouched — byte-equivalent dumps.
        for k in ("tool-a", "agent-a"):
            assert re_bundle["entries"][k] == bundle["entries"][k]

    def test_atomic_failure_leaves_namespace_untouched(
        self, runner: CliRunner, catalog_root: Path, tmp_path: Path
    ) -> None:
        # AC18 — broken bundle with dangling ref → exit 1 → export byte-equal to pre.
        before = runner.invoke(
            cli_main.app, _base_args(catalog_root) + ["export", "--namespace", "ns-a"]
        )
        assert before.exit_code == 0
        pre_bundle_text = before.stdout

        # Break the bundle: point agent-a.linked.__ref__ at a missing id.
        bundle = yaml.safe_load(pre_bundle_text)
        bundle["entries"]["agent-a"]["payload"]["linked"] = {"__ref__": "does-not-exist"}
        broken_path = tmp_path / "broken.yaml"
        broken_path.write_text(yaml.safe_dump(bundle, sort_keys=False))

        imp = runner.invoke(cli_main.app, _base_args(catalog_root) + ["import", str(broken_path)])
        assert imp.exit_code == 1
        assert "validation error:" in imp.stderr

        after = runner.invoke(
            cli_main.app, _base_args(catalog_root) + ["export", "--namespace", "ns-a"]
        )
        assert after.exit_code == 0
        assert after.stdout == pre_bundle_text


# --------------------------------------------------------------------------- #
# `import --dry-run` — success + failure
# --------------------------------------------------------------------------- #


class TestImportDryRun:
    """AC12-AC13 + AC19-AC21 — dry-run delegation, report rendering, exit-code map."""

    def test_dry_run_happy_json(
        self, runner: CliRunner, catalog_root: Path, tmp_path: Path
    ) -> None:
        # AC19 — export → dry-run --format json → ok=True, empty errors.
        export = runner.invoke(
            cli_main.app, _base_args(catalog_root) + ["export", "--namespace", "ns-a"]
        )
        bundle_path = tmp_path / "bundle.yaml"
        bundle_path.write_text(export.stdout)

        result = runner.invoke(
            cli_main.app,
            _base_args(catalog_root)
            + ["--format", "json", "import", str(bundle_path), "--dry-run"],
        )
        assert result.exit_code == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload["ok"] is True
        assert payload["global_errors"] == []
        assert payload["entry_issues"] == []
        assert result.stderr == ""

    def test_dry_run_failure_json(
        self, runner: CliRunner, catalog_root: Path, tmp_path: Path
    ) -> None:
        # AC20 — broken bundle → exit 1, stdout valid JSON with ok=false,
        # stderr carries the AC13 summary line.
        export = runner.invoke(
            cli_main.app, _base_args(catalog_root) + ["export", "--namespace", "ns-a"]
        )
        bundle = yaml.safe_load(export.stdout)
        bundle["entries"]["agent-a"]["payload"]["linked"] = {"__ref__": "does-not-exist"}
        broken_path = tmp_path / "broken.yaml"
        broken_path.write_text(yaml.safe_dump(bundle, sort_keys=False))

        result = runner.invoke(
            cli_main.app,
            _base_args(catalog_root)
            + ["--format", "json", "import", str(broken_path), "--dry-run"],
        )
        assert result.exit_code == 1
        payload = json.loads(result.stdout)
        assert payload["ok"] is False
        has_error = bool(payload["global_errors"]) or any(
            i["errors"] for i in payload["entry_issues"]
        )
        assert has_error
        assert re.search(
            r"validation failed: \d+ global error\(s\), \d+ entry issue\(s\)", result.stderr
        )
        # AC20 — no writes performed: re-export equals pre-export.
        before_re = runner.invoke(
            cli_main.app, _base_args(catalog_root) + ["export", "--namespace", "ns-a"]
        )
        assert before_re.stdout == export.stdout

    @pytest.mark.parametrize("fmt", ["table", "json", "yaml"])
    def test_dry_run_happy_format_coverage(
        self, runner: CliRunner, catalog_root: Path, tmp_path: Path, fmt: str
    ) -> None:
        # AC21 — parametrised over all formats.
        export = runner.invoke(
            cli_main.app, _base_args(catalog_root) + ["export", "--namespace", "ns-a"]
        )
        bundle_path = tmp_path / "bundle.yaml"
        bundle_path.write_text(export.stdout)

        result = runner.invoke(
            cli_main.app,
            _base_args(catalog_root) + ["--format", fmt, "import", str(bundle_path), "--dry-run"],
        )
        assert result.exit_code == 0, result.stderr
        if fmt == "json":
            payload = json.loads(result.stdout)
            assert payload["ok"] is True
            assert payload["namespace"] == "ns-a"
        elif fmt == "yaml":
            payload = yaml.safe_load(result.stdout)
            assert payload["ok"] is True
            assert payload["namespace"] == "ns-a"
        else:
            assert "ok: True" in result.stdout
            assert "ns-a" in result.stdout


# --------------------------------------------------------------------------- #
# Usage errors
# --------------------------------------------------------------------------- #


class TestImportUsageErrors:
    """AC9 + AC11 + AC22 — path/parse/encoding failure shapes."""

    def test_missing_file(self, runner: CliRunner, catalog_root: Path, tmp_path: Path) -> None:
        missing = tmp_path / "does-not-exist.yaml"
        result = runner.invoke(cli_main.app, _base_args(catalog_root) + ["import", str(missing)])
        assert result.exit_code == 2
        assert "file not found" in result.stderr

    def test_directory_not_file(
        self, runner: CliRunner, catalog_root: Path, tmp_path: Path
    ) -> None:
        d = tmp_path / "a-directory"
        d.mkdir()
        result = runner.invoke(cli_main.app, _base_args(catalog_root) + ["import", str(d)])
        assert result.exit_code == 2
        assert "file not found" in result.stderr

    def test_non_utf8_file(self, runner: CliRunner, catalog_root: Path, tmp_path: Path) -> None:
        bad = tmp_path / "garbage.yaml"
        bad.write_bytes(b"\xff\xfe\x00\x00\xff\xff")
        result = runner.invoke(cli_main.app, _base_args(catalog_root) + ["import", str(bad)])
        assert result.exit_code == 2
        assert "not valid UTF-8" in result.stderr

    def test_malformed_yaml(self, runner: CliRunner, catalog_root: Path, tmp_path: Path) -> None:
        bad = tmp_path / "malformed.yaml"
        bad.write_text(":\n-[[[\n")
        result = runner.invoke(cli_main.app, _base_args(catalog_root) + ["import", str(bad)])
        assert result.exit_code == 2
        assert "YAML parse error" in result.stderr


# --------------------------------------------------------------------------- #
# Stdout discipline
# --------------------------------------------------------------------------- #


class TestStdoutDiscipline:
    """AC24 — stream routing pins."""

    def test_persistence_stdout_empty_success_on_stderr(
        self, runner: CliRunner, catalog_root: Path, tmp_path: Path
    ) -> None:
        export = runner.invoke(
            cli_main.app, _base_args(catalog_root) + ["export", "--namespace", "ns-a"]
        )
        bundle_path = tmp_path / "bundle.yaml"
        bundle_path.write_text(export.stdout)
        result = runner.invoke(
            cli_main.app, _base_args(catalog_root) + ["import", str(bundle_path)]
        )
        assert result.exit_code == 0, result.stderr
        assert result.stdout == ""
        assert "imported" in result.stderr

    def test_dry_run_json_stderr_empty_stdout_has_report(
        self, runner: CliRunner, catalog_root: Path, tmp_path: Path
    ) -> None:
        export = runner.invoke(
            cli_main.app, _base_args(catalog_root) + ["export", "--namespace", "ns-a"]
        )
        bundle_path = tmp_path / "bundle.yaml"
        bundle_path.write_text(export.stdout)
        result = runner.invoke(
            cli_main.app,
            _base_args(catalog_root)
            + ["--format", "json", "import", str(bundle_path), "--dry-run"],
        )
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["ok"] is True
        assert result.stderr == ""
