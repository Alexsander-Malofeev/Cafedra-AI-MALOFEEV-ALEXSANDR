from src.preprocessing import clean_text, count_words, is_too_short


def test_clean_text_removes_punctuation_and_normalizes_case():
    assert clean_text("ЁЖИК!!!  Ошибка???") == "ежик ошибка"


def test_count_words():
    assert count_words("Не могу войти в аккаунт") == 5


def test_short_text_detection():
    assert is_too_short("Ошибка") is True
    assert is_too_short("Не могу войти") is False
