from activities.embedding import embed_batch_activity
from activities.indexing import (
    build_index_activity,
    garbage_collect_indexes_activity,
    swap_index_activity,
)
from activities.sources import create_source_run_activity, fetch_source_items_activity
from activities.storage import (
    cleanup_temp_file_activity,
    download_image_activity,
    process_image_activity,
)
from activities.vision import caption_activity, ocr_activity
from activities.workflow_state import (
    complete_ingest_job_activity,
    complete_rebuild_job_activity,
    fail_rebuild_job_activity,
    ingest_initialize_activity,
    mark_ingest_url_done_activity,
    mark_ingest_url_failed_activity,
    save_annotations_activity,
    save_embedding_info_activity,
    start_rebuild_job_activity,
    update_job_progress_activity,
)

ACTIVITIES = [
    create_source_run_activity,
    fetch_source_items_activity,
    download_image_activity,
    process_image_activity,
    cleanup_temp_file_activity,
    caption_activity,
    ocr_activity,
    embed_batch_activity,
    build_index_activity,
    swap_index_activity,
    garbage_collect_indexes_activity,
    ingest_initialize_activity,
    mark_ingest_url_failed_activity,
    mark_ingest_url_done_activity,
    save_annotations_activity,
    save_embedding_info_activity,
    update_job_progress_activity,
    complete_ingest_job_activity,
    start_rebuild_job_activity,
    fail_rebuild_job_activity,
    complete_rebuild_job_activity,
]


__all__ = [
    "ACTIVITIES",
    "create_source_run_activity",
    "fetch_source_items_activity",
    "download_image_activity",
    "process_image_activity",
    "cleanup_temp_file_activity",
    "caption_activity",
    "ocr_activity",
    "embed_batch_activity",
    "build_index_activity",
    "swap_index_activity",
    "garbage_collect_indexes_activity",
    "ingest_initialize_activity",
    "mark_ingest_url_failed_activity",
    "mark_ingest_url_done_activity",
    "save_annotations_activity",
    "save_embedding_info_activity",
    "update_job_progress_activity",
    "complete_ingest_job_activity",
    "start_rebuild_job_activity",
    "fail_rebuild_job_activity",
    "complete_rebuild_job_activity",
]
