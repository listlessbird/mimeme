from activities.embedding import embed_batch_activity, encode_query_activity
from activities.indexing import (
    build_index_activity,
    collect_embeddings_activity,
    garbage_collect_indexes_activity,
    swap_index_activity,
)
from activities.storage import (
    cleanup_temp_file_activity,
    download_image_activity,
    process_image_activity,
)
from activities.vision import caption_activity, ocr_activity
from activities.workflow_state import (
    complete_ingest_job_activity,
    complete_rebuild_job_activity,
    ingest_initialize_activity,
    mark_ingest_url_done_activity,
    mark_ingest_url_failed_activity,
    save_annotations_activity,
    save_embedding_info_activity,
    start_rebuild_job_activity,
    update_job_progress_activity,
)

CPU_ACTIVITIES = [
    download_image_activity,
    process_image_activity,
    cleanup_temp_file_activity,
    collect_embeddings_activity,
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
    complete_rebuild_job_activity,
]

GPU_ACTIVITIES = [
    caption_activity,
    ocr_activity,
    embed_batch_activity,
    encode_query_activity,
]

__all__ = [
    "CPU_ACTIVITIES",
    "GPU_ACTIVITIES",
    "download_image_activity",
    "process_image_activity",
    "cleanup_temp_file_activity",
    "caption_activity",
    "ocr_activity",
    "embed_batch_activity",
    "encode_query_activity",
    "collect_embeddings_activity",
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
    "complete_rebuild_job_activity",
]
