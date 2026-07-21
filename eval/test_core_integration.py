"""Contract tests for the pinned ptcg-agent-core integration (SOT-1810)."""

import configparser
import pathlib
import subprocess
import tarfile
import unittest


REPO = pathlib.Path(__file__).resolve().parents[1]
CORE = REPO / "vendor" / "ptcg-agent-core"


class CoreIntegrationTest(unittest.TestCase):
    def test_submodule_uses_the_canonical_core_repository(self):
        config = configparser.ConfigParser()
        config.read(REPO / ".gitmodules")
        section = 'submodule "vendor/ptcg-agent-core"'
        self.assertEqual(config[section]["path"], "vendor/ptcg-agent-core")
        self.assertEqual(
            config[section]["url"],
            "https://github.com/sota1111/ptcg-agent-core.git",
        )

    def test_required_core_contracts_are_available(self):
        self.assertTrue((CORE / "package.json").is_file())
        guide = CORE / "docs" / "kaggle-submission.md"
        self.assertTrue(guide.is_file())
        self.assertIn("submission.tar.gz", guide.read_text(encoding="utf-8"))

    def test_submission_builder_excludes_development_and_core_files(self):
        subprocess.run(
            ["bash", "scripts/build_submission.sh"],
            cwd=REPO,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        archive = REPO / "submission.tar.gz"
        try:
            with tarfile.open(archive, "r:gz") as bundle:
                names = bundle.getnames()
            self.assertIn("main.py", names)
            self.assertIn("deck.csv", names)
            forbidden = ("vendor/", ".git/", "eval/", "venv/", "__pycache__/")
            self.assertFalse(any(name.startswith(forbidden) for name in names))
        finally:
            archive.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
