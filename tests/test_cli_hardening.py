from yake_sum.cli import main

TEXT = "Graph databases store nodes and edges natively. They suit fraud detection and knowledge graphs."


def test_check_backend_mock_ok(capsys):
    assert main(["--check-backend", "-b", "mock"]) == 0
    assert "OK" in capsys.readouterr().out


def test_check_backend_unreachable_exit_3(capsys):
    assert main(["--check-backend", "-b", "ollama", "--host", "http://127.0.0.1:1"]) == 3
    err = capsys.readouterr().err
    assert "NOT READY" in err and "ollama serve" in err


def test_missing_input_is_a_usage_error(capsys):
    assert main([]) == 2


def test_mock_backend_warns_loudly_on_stderr(tmp_path, capsys):
    f = tmp_path / "d.txt"
    f.write_text(TEXT, encoding="utf-8")
    assert main(["-i", str(f), "-m", "abstractive", "-b", "mock"]) == 0
    assert "NOT a real summary" in capsys.readouterr().err


def test_json_output_contains_warnings_and_faithfulness(tmp_path, capsys):
    import json

    f = tmp_path / "d.txt"
    f.write_text(TEXT, encoding="utf-8")
    assert main(["-i", str(f), "-m", "abstractive", "-b", "mock", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "faithfulness" in payload["metrics"] and payload["metrics"]["warnings"]
