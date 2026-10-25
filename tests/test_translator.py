
from translator import translate_text


def test_empty_text():
    """Пустая строка не должна отправляться на перевод."""

    result = translate_text("")

    assert result == ""


def test_spaces_only():
    """Строка из пробелов должна возвращать пустой результат."""

    result = translate_text("   ")

    assert result == ""
