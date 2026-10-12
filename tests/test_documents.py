from app.engines.base import Element, ParsedDocument


def test_text_joins_non_empty_elements() -> None:
    document = ParsedDocument(
        elements=[Element("one", page_number=1), Element(""), Element("two", page_number=2)]
    )

    assert document.text == "one\n\ntwo"


def test_page_numbers_cover_overlapping_elements() -> None:
    document = ParsedDocument(
        elements=[
            Element("a" * 10, page_number=1),
            Element("b" * 10, page_number=2),
            Element("c" * 10, page_number=3),
        ]
    )
    # Layout: page 1 = [0, 10), page 2 = [12, 22), page 3 = [24, 34)

    assert document.page_numbers(0, 10) == [1]
    assert document.page_numbers(5, 15) == [1, 2]
    assert document.page_numbers(12, 34) == [2, 3]
    assert document.page_numbers(30, 34) == [3]


def test_page_numbers_empty_without_pages() -> None:
    assert ParsedDocument.from_text("plain text").page_numbers(0, 5) == []
