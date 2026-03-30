"""Tests for CLI."""

from click.testing import CliRunner

from llm_quanta.cli import main


class TestCLI:
    """Tests for CLI commands."""

    def test_version(self) -> None:
        """Test --version flag."""
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "llm-quanta" in result.output

    def test_help(self) -> None:
        """Test --help flag."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "LLM-Quanta" in result.output
        assert "quantize" in result.output
        assert "benchmark" in result.output
        assert "compare" in result.output

    def test_quantize_help(self) -> None:
        """Test quantize --help."""
        runner = CliRunner()
        result = runner.invoke(main, ["quantize", "--help"])
        assert result.exit_code == 0
        assert "methods" in result.output.lower() or "Methods" in result.output

    def test_compare_help(self) -> None:
        """Test compare --help."""
        runner = CliRunner()
        result = runner.invoke(main, ["compare", "--help"])
        assert result.exit_code == 0
