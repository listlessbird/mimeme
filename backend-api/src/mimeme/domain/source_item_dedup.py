from pydantic import BaseModel

from mimeme.domain.adapters.base import DiscoveredItem


class SourceItemDedup(BaseModel, frozen=True):
    new: list[DiscoveredItem]
    already_seen: list[DiscoveredItem]


def dedup_source_items(discovered: list[DiscoveredItem], *, seen_ids: set[str]) -> SourceItemDedup:
    new: list[DiscoveredItem] = []
    already_seen: list[DiscoveredItem] = []
    collapsed: set[str] = set()

    for item in discovered:
        external_item_id = item.external_item_id

        if external_item_id in collapsed:
            continue

        collapsed.add(external_item_id)

        if external_item_id in seen_ids:
            already_seen.append(item)
        else:
            new.append(item)

    return SourceItemDedup(new=new, already_seen=already_seen)
