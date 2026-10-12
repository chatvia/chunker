from app.engines.base import Document, Page


def test_text_joins_non_empty_pages() -> None:
    document = Document(pages=[Page("one", 1), Page(""), Page("two", 3)])

    assert document.text == "one\n\ntwo"


def test_page_numbers_cover_overlapping_pages() -> None:
    document = Document.from_pages([("a" * 10, 1), ("b" * 10, 2), ("c" * 10, 3)])
    # Layout: page 1 = [0, 10), page 2 = [12, 22), page 3 = [24, 34)

    assert document.page_numbers(0, 10) == [1]
    assert document.page_numbers(5, 15) == [1, 2]
    assert document.page_numbers(12, 34) == [2, 3]
    assert document.page_numbers(30, 34) == [3]


def test_page_numbers_empty_for_plain_text() -> None:
    assert Document.from_text("plain text").page_numbers(0, 5) == []
