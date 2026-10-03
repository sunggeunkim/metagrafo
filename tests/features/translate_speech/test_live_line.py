from features.translate_speech.live_line import LiveLine


def test_fragments_grow_one_open_line() -> None:
    line = LiveLine()
    assert line.extend("Today's topical") == "Today's topical"
    assert line.extend(" sermon, \"The Prayer") == "Today's topical sermon, \"The Prayer"
    assert not line.closes_sentence()
    assert line.commit() == "Today's topical sermon, \"The Prayer"


def test_sentence_punctuation_marks_the_line_ready_to_commit() -> None:
    line = LiveLine()
    line.extend("Today's topical sermon.")
    assert line.closes_sentence()
    assert line.commit() == "Today's topical sermon."
    assert line.commit() is None


def test_commit_without_punctuation_keeps_the_words_and_clears() -> None:
    line = LiveLine()
    line.extend("Originally")
    assert not line.closes_sentence()
    assert line.commit() == "Originally"
    assert line.extend("I was") == "I was"
