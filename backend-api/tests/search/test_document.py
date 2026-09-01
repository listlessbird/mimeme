from __future__ import annotations

from mimeme.search import document


def test_projection_without_source_facts_keeps_image_annotation() -> None:
    value = document.project(
        7,
        caption="  surprised   cat ",
        ocr_text=" TOP\nTEXT ",
    )

    assert value == document.SearchDocument(
        image_id=7,
        captions=("surprised cat",),
        ocr_texts=("TOP TEXT",),
    )


def test_projection_aggregates_aliases_without_collapsing_conflicts() -> None:
    value = document.project(
        9,
        sources=(
            document.SourceFacts(
                item_title="  Distracted   Boyfriend ",
                title="Distracted Boyfriend",
                tags=("reaction", " stock photo "),
                categories=("Relationship",),
                types=("Image macro",),
                origin="Instagram",
                year="2017",
                description="A man looks back at another woman.",
            ),
            document.SourceFacts(
                item_title="Disloyal Man Walking",
                title="Distracted Boyfriend",
                tags=("Reaction", "reaction", "stock\nphoto"),
                categories=("relationship",),
                origin="Stock photography",
                year=" 2017 ",
                description="A man looks back at another woman.",
            ),
        ),
    )

    assert value.titles == ("Disloyal Man Walking", "Distracted Boyfriend")
    assert value.tags == ("Reaction", "reaction", "stock photo")
    assert value.categories == ("Relationship", "relationship")
    assert value.types == ("Image macro",)
    assert value.origins == ("Instagram", "Stock photography")
    assert value.years == ("2017",)
    assert value.descriptions == ("A man looks back at another woman.",)


def test_projection_ignores_null_and_blank_values() -> None:
    value = document.project(
        11,
        sources=(
            document.SourceFacts(
                item_title=" \n ",
                tags=("", "\t"),
                categories=(),
                origin=None,
            ),
        ),
        caption=" ",
    )

    assert value == document.SearchDocument(image_id=11)


def test_canonical_serialization_is_stable_across_input_order() -> None:
    first = document.project(
        13,
        sources=(
            document.SourceFacts(item_title="Zulu", tags=("two", "one")),
            document.SourceFacts(item_title="Alpha", tags=("one",)),
        ),
    )
    second = document.project(
        13,
        sources=(
            document.SourceFacts(item_title="Alpha", tags=("one",)),
            document.SourceFacts(item_title="Zulu", tags=("one", "two")),
        ),
    )

    assert document.canonical_json(first) == document.canonical_json(second)
    assert document.canonical_json(first) == document.canonical_json(first)
