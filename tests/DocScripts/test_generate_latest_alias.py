import sys
from pathlib import Path

import pytest

# Add docs/scripts to path so we can import the modules
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs" / "scripts"))

from generate_latest_alias import find_latest_stable_version, main


@pytest.fixture
def site_dir(tmp_path: Path) -> Path:
    site = tmp_path / "_site"
    site.mkdir()
    return site


def _create_version_dir(site_dir: Path, version: str) -> None:
    version_dir = site_dir / version
    version_dir.mkdir()
    (version_dir / "index.html").write_text(f"<html>{version}</html>")
    (version_dir / "api.html").write_text(f"<html>API {version}</html>")


class TestFindLatestStableVersion:
    def test_no_versions(self, site_dir: Path) -> None:
        assert find_latest_stable_version(site_dir) is None

    def test_single_stable_version(self, site_dir: Path) -> None:
        _create_version_dir(site_dir, "v1.0.0")
        assert find_latest_stable_version(site_dir) == "v1.0.0"

    def test_multiple_stable_versions(self, site_dir: Path) -> None:
        _create_version_dir(site_dir, "v1.0.0")
        _create_version_dir(site_dir, "v1.2.0")
        _create_version_dir(site_dir, "v1.1.0")
        assert find_latest_stable_version(site_dir) == "v1.2.0"

    def test_only_rc_versions_falls_back(self, site_dir: Path) -> None:
        _create_version_dir(site_dir, "v1.5.0rc1")
        _create_version_dir(site_dir, "v1.5.0rc2")
        assert find_latest_stable_version(site_dir) == "v1.5.0rc2"

    def test_stable_preferred_over_rc(self, site_dir: Path) -> None:
        _create_version_dir(site_dir, "v1.4.0")
        _create_version_dir(site_dir, "v1.5.0rc1")
        assert find_latest_stable_version(site_dir) == "v1.4.0"

    def test_ignores_non_version_dirs(self, site_dir: Path) -> None:
        _create_version_dir(site_dir, "v1.0.0")
        (site_dir / "latest").mkdir()
        (site_dir / "main").mkdir()
        assert find_latest_stable_version(site_dir) == "v1.0.0"


class TestMain:
    def test_moves_latest_and_leaves_redirect(
        self, site_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _create_version_dir(site_dir, "v1.3.0")
        _create_version_dir(site_dir, "v1.4.0")
        monkeypatch.setattr(sys, "argv", ["generate_latest_alias.py", str(site_dir)])

        result = main()

        assert result == 0
        # Content moved to latest/
        latest_dir = site_dir / "latest"
        assert latest_dir.exists()
        assert (latest_dir / "index.html").read_text() == "<html>v1.4.0</html>"
        assert (latest_dir / "api.html").read_text() == "<html>API v1.4.0</html>"
        # Old path has a redirect, not the original content
        old_dir = site_dir / "v1.4.0"
        assert old_dir.exists()
        assert "latest" in (old_dir / "index.html").read_text()
        assert not (old_dir / "api.html").exists()

    def test_replaces_existing_latest(
        self, site_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _create_version_dir(site_dir, "v1.3.0")
        # Create a stale latest directory
        latest_dir = site_dir / "latest"
        latest_dir.mkdir()
        (latest_dir / "index.html").write_text("<html>old</html>")

        monkeypatch.setattr(sys, "argv", ["generate_latest_alias.py", str(site_dir)])

        result = main()

        assert result == 0
        assert (latest_dir / "index.html").read_text() == "<html>v1.3.0</html>"

    def test_missing_site_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            sys, "argv", ["generate_latest_alias.py", str(tmp_path / "nonexistent")]
        )
        assert main() == 1

    def test_empty_site_dir(self, site_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["generate_latest_alias.py", str(site_dir)])
        assert main() == 1
