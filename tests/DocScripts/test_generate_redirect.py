import sys
from pathlib import Path

import pytest

# Add docs/scripts to path so we can import the modules
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs" / "scripts"))

from generate_redirect import generate_redirect_html, main


@pytest.fixture
def site_dir(tmp_path: Path) -> Path:
    site = tmp_path / "_site"
    site.mkdir()
    return site


def _create_version_dir(site_dir: Path, version: str) -> None:
    version_dir = site_dir / version
    version_dir.mkdir()
    (version_dir / "index.html").write_text(f"<html>{version}</html>")


class TestGenerateRedirectHtml:
    def test_redirect_target_in_meta(self) -> None:
        html = generate_redirect_html("latest")
        assert "url=./latest/" in html

    def test_redirect_target_in_link(self) -> None:
        html = generate_redirect_html("latest")
        assert 'href="./latest/index.html"' in html

    def test_specific_version_target(self) -> None:
        html = generate_redirect_html("v1.4.0")
        assert "url=./v1.4.0/" in html


class TestMain:
    def test_redirects_to_latest(self, site_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _create_version_dir(site_dir, "v1.3.0")
        _create_version_dir(site_dir, "v1.4.0")
        monkeypatch.setattr(sys, "argv", ["generate_redirect.py", str(site_dir)])

        result = main()

        assert result == 0
        index_html = (site_dir / "index.html").read_text()
        assert "url=./latest/" in index_html

    def test_missing_site_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["generate_redirect.py", str(tmp_path / "nonexistent")])
        assert main() == 1

    def test_empty_site_dir(self, site_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["generate_redirect.py", str(site_dir)])
        assert main() == 1
