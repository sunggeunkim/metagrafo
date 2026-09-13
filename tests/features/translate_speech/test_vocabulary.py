from pathlib import Path

from features.translate_speech.vocabulary import load_vocabulary


def test_vocabulary_file_becomes_initial_prompt(tmp_path: Path) -> None:
    path = tmp_path / "church_vocabulary.txt"
    path.write_text("Matthew, Mark, Luke\nJesus, Gospel, Amen.\n", encoding="utf-8")
    prompt = load_vocabulary(path)
    assert prompt == "Matthew, Mark, Luke Jesus, Gospel, Amen."


def test_missing_vocabulary_file_is_empty(tmp_path: Path) -> None:
    assert load_vocabulary(tmp_path / "missing.txt") == ""
