import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs" / "scripts"))

from version_utils import find_latest_rc_tag, find_latest_stable_tag, parse_version


class TestParseVersion:
    def test_stable_sorts_higher_than_rc(self) -> None:
        assert parse_version("v1.5.0") > parse_version("v1.5.0rc10")

    def test_rc10_sorts_higher_than_rc9(self) -> None:
        assert parse_version("v1.5.0rc10") > parse_version("v1.5.0rc9")

    def test_rc2_sorts_higher_than_rc1(self) -> None:
        assert parse_version("v1.5.0rc2") > parse_version("v1.5.0rc1")

    def test_higher_minor_wins(self) -> None:
        assert parse_version("v1.5.0") > parse_version("v1.4.0")

    def test_higher_patch_wins(self) -> None:
        assert parse_version("v1.4.2") > parse_version("v1.4.1")


class TestFindLatestRcTag:
    def test_rc10_over_rc9(self) -> None:
        tags = ["v1.5.0rc7", "v1.5.0rc9", "v1.5.0rc10", "v1.4.0"]
        assert find_latest_rc_tag(tags) == "v1.5.0rc10"

    def test_no_rc_tags(self) -> None:
        tags = ["v1.4.0", "v1.3.0"]
        assert find_latest_rc_tag(tags) is None


class TestFindLatestStableTag:
    def test_ignores_rc(self) -> None:
        tags = ["v1.5.0rc10", "v1.4.0", "v1.3.0"]
        assert find_latest_stable_tag(tags) == "v1.4.0"
