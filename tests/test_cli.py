from lognexus import cli


def test_cli_returns_nonzero_for_missing_input_dir(tmp_path, capsys, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    monkeypatch.setattr(
        "sys.argv",
        ["lognexus", "--input_dir", str(tmp_path / "missing"), "--model_dir", str(model_dir)],
    )

    assert cli.main() == 1
    assert "Input directory" in capsys.readouterr().err


def test_cli_help_exits_successfully(capsys, monkeypatch):
    monkeypatch.setattr("sys.argv", ["lognexus", "--help"])

    try:
        cli.main()
    except SystemExit as exc:
        assert exc.code == 0

    assert "LogNexus" in capsys.readouterr().out
