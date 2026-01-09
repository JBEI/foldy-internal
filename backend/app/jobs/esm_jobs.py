import json
import logging
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from io import BytesIO, StringIO
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from Bio import SeqIO
from flask import current_app
from werkzeug.exceptions import BadRequest

if TYPE_CHECKING:
    import torch

from app.helpers.boltz_yaml_helper import BoltzYamlHelper
from app.helpers.esm_client import FoldyPLMClient
from app.helpers.esm_util import (
    MSA_A3M_STORAGE_PATH,
    MSA_CONTEXT_STORAGE_PATH,
    get_naturalness,
    normalize_msa_a3m_contents,
    validate_msa_a3m_contents,
)
from app.helpers.fold_storage_manager import FoldStorageManager
from app.helpers.gpu_util import clean_up_torch_memory, log_memory_usage
from app.helpers.jobs_util import (
    LoggingRecorder,
    _live_update_tail,
    _psql_tail,
    get_torch_cuda_is_available_and_add_logs,
)
from app.helpers.sequence_util import (
    VALID_AMINO_ACIDS,
    get_loci_set,
    get_seq_ids_for_deep_mutational_scan,
    is_homolog_seq_id,
    maybe_get_seq_id_error_message,
    process_and_validate_evolve_input_files,
    seq_id_to_seq,
)
from app.models import Dock, Embedding, FewShot, Fold, Invokation, Naturalness


def load_fasta_to_dict(homolog_fasta: str) -> dict[str, str]:
    homolog_id_to_seq_map = {}
    if homolog_fasta:
        try:
            fasta_io = SeqIO.parse(StringIO(homolog_fasta), "fasta")
            for record in fasta_io:
                homolog_id_to_seq_map[record.id] = str(record.seq)
        except Exception as e:
            raise ValueError(f"Invalid homolog fasta: {e}")
    return homolog_id_to_seq_map


def validate_embedding_inputs(
    wt_aa_seq, extra_seq_ids, dms_starting_seq_ids, homolog_id_to_seq_map: dict[str, str]
) -> None:
    if ":" in wt_aa_seq or ";" in wt_aa_seq:
        raise KeyError(
            f"Fold seems to be a multimer which is not supported for ESM embeddings yet."
        )

    # Validate homolog IDs.
    for seq_id, homolog_seq in homolog_id_to_seq_map.items():
        if not is_homolog_seq_id(seq_id):
            raise ValueError(f"Invalid homolog ID {seq_id}: must start with HOM-")
        for aa in homolog_seq:
            if aa not in VALID_AMINO_ACIDS:
                raise ValueError(
                    f"The Fasta is invalid: {seq_id} with sequence {homolog_seq}: {aa} is not a valid amino acid"
                )

    for extra_seq_id in extra_seq_ids:
        if is_homolog_seq_id(extra_seq_id):
            if extra_seq_id not in homolog_id_to_seq_map:
                raise ValueError(f"Homolog sequence {extra_seq_id} not found in homolog fasta.")

        seq_id_errors = maybe_get_seq_id_error_message(wt_aa_seq, extra_seq_id)
        if seq_id_errors:
            raise ValueError(f"Invalid extra seq id {extra_seq_id}: {seq_id_errors}")
    for dms_starting_seq_id in dms_starting_seq_ids:
        seq_id_errors = maybe_get_seq_id_error_message(wt_aa_seq, dms_starting_seq_id)
        if seq_id_errors:
            raise ValueError(f"Invalid DMS starting seq id {dms_starting_seq_id}: {seq_id_errors}")


def has_stored_msa_context(storage_manager, fold_id: int) -> bool:
    if storage_manager is None:
        return False
    try:
        storage_manager.get_blob(fold_id, MSA_CONTEXT_STORAGE_PATH)
        return True
    except Exception:
        return False


def has_stored_msa_a3m(storage_manager, fold_id: int) -> bool:
    if storage_manager is None:
        return False
    try:
        storage_manager.get_blob(fold_id, MSA_A3M_STORAGE_PATH)
        return True
    except Exception:
        return False


def stored_msa_context_exists(fold_id: int) -> bool:
    fsm = FoldStorageManager()
    fsm.setup()
    return has_stored_msa_context(fsm.storage_manager, fold_id)


def stored_msa_a3m_exists(fold_id: int) -> bool:
    fsm = FoldStorageManager()
    fsm.setup()
    return has_stored_msa_a3m(fsm.storage_manager, fold_id)


def load_stored_msa_a3m(storage_manager, fold_id: int) -> str | None:
    if storage_manager is None:
        return None
    try:
        stored_msa = storage_manager.get_binary(fold_id, MSA_A3M_STORAGE_PATH)
        return stored_msa.decode()
    except Exception as exc:
        logging.info("No stored MSA A3M found at %s: %s", MSA_A3M_STORAGE_PATH, exc)
        return None


def load_or_sample_msa_context(
    fsm: FoldStorageManager,
    fold_id: int,
    msa_a3m_path: str | None,
    device: "torch.device",
) -> str:
    storage_manager = fsm.storage_manager
    if storage_manager is None:
        raise RuntimeError("Storage manager is not initialized.")
    try:
        stored_context = storage_manager.get_binary(fold_id, MSA_CONTEXT_STORAGE_PATH)
        context = stored_context.decode()
        if context.strip():
            if msa_a3m_path:
                logging.info(
                    "Loaded stored MSA context from %s (ignoring provided msa_a3m)",
                    MSA_CONTEXT_STORAGE_PATH,
                )
            else:
                logging.info("Loaded stored MSA context from %s", MSA_CONTEXT_STORAGE_PATH)
            return context
    except Exception as exc:
        logging.info("No stored MSA context found at %s: %s", MSA_CONTEXT_STORAGE_PATH, exc)

    if not msa_a3m_path:
        raise BadRequest("msa_a3m is required when use_msa_context=true.")

    import random

    from E1.msa_sampling import ContextSpecification, sample_context

    context_spec = ContextSpecification()
    seed = random.SystemRandom().randrange(0, 2**32 - 1)
    context, context_ids = sample_context(
        msa_a3m_path,
        max_num_samples=context_spec.max_num_samples,
        max_token_length=context_spec.max_token_length,
        max_query_similarity=context_spec.max_query_similarity,
        min_query_similarity=context_spec.min_query_similarity,
        neighbor_similarity_lower_bound=context_spec.neighbor_similarity_lower_bound,
        seed=seed,
        device=device,
    )
    if not context:
        raise ValueError("MSA context sampling returned no context sequences")
    storage_manager.write_file(fold_id, MSA_CONTEXT_STORAGE_PATH, context)
    logging.info(
        "Stored MSA context at %s (sampled %d sequences)",
        MSA_CONTEXT_STORAGE_PATH,
        len(context_ids),
    )
    return context


def get_esm_embeddings(
    embed_id: int,
    msa_a3m: str | None = None,
):
    """Compute the ESM embeddings and store them with the storage manager.

    Arguments:
        embed_id: ID of the embedding record to run.
    """
    # 1. Get records.
    embed_record = Embedding.get_by_id(embed_id)
    if not embed_record:
        raise KeyError(f"Embedding ID {embed_id} not found!")
    embed_name = embed_record.name
    embedding_model = embed_record.embedding_model
    use_msa_context = bool(embed_record.use_msa_context)

    invokation = Invokation.get_by_id(embed_record.invokation_id)
    if not invokation:
        raise KeyError(
            f"Embedding ID {embed_id} ({embed_name}) does not have an associated invokation!"
        )
    with LoggingRecorder(invokation):
        logging.info(
            "Starting embedding...",
        )

        fsm = FoldStorageManager()
        fsm.setup()
        storage_manager = fsm.storage_manager
        if storage_manager is None:
            raise RuntimeError("Storage manager is not initialized.")

        dms_starting_seq_ids = (
            embed_record.dms_starting_seq_ids.split(",")
            if embed_record.dms_starting_seq_ids
            else []
        )
        extra_seq_ids = embed_record.extra_seq_ids.split(",") if embed_record.extra_seq_ids else []
        extra_layers = (
            [int(ii) for ii in embed_record.extra_layers.split(",")]
            if embed_record.extra_layers
            else []
        )
        domain_boundaries = (
            [int(ii) for ii in embed_record.domain_boundaries.split(",")]
            if embed_record.domain_boundaries
            else []
        )
        homolog_fasta = embed_record.homolog_fasta

        fold = embed_record.fold
        if not fold:
            raise KeyError(
                f"Embedding ID {embed_id} ({embed_name}) does not have an associated fold!"
            )

        # 3. Validate seq_ids.
        if not fold.yaml_config:
            raise ValueError("Fold does not have a YAML config!")
        boltz_yaml_helper = BoltzYamlHelper(fold.yaml_config)
        if len(boltz_yaml_helper.get_protein_sequences()) > 1:
            raise ValueError(
                "Fold has multiple protein sequences, which is not supported for ESM embeddings yet."
            )
        wt_aa_seq = boltz_yaml_helper.get_protein_sequences()[0][1]

        stored_context_exists = False
        stored_a3m_exists = False
        if use_msa_context:
            stored_context_exists = has_stored_msa_context(storage_manager, fold.id)
            stored_a3m_exists = has_stored_msa_a3m(storage_manager, fold.id)
            if not embedding_model.startswith("e1_"):
                raise BadRequest("MSA context is only supported for E1 embedding models.")
            if msa_a3m:
                msa_a3m = normalize_msa_a3m_contents(msa_a3m, wt_aa_seq)
                validate_msa_a3m_contents(msa_a3m, wt_aa_seq)
            elif not (stored_context_exists or stored_a3m_exists):
                raise BadRequest("msa_a3m is required when use_msa_context=true.")

        homolog_id_to_seq_map = load_fasta_to_dict(homolog_fasta)
        validate_embedding_inputs(
            wt_aa_seq, extra_seq_ids, dms_starting_seq_ids, homolog_id_to_seq_map
        )

        logging.info(
            f"Getting all sequence IDs (dms_starting_seq_ids: {dms_starting_seq_ids}; extra_seq_ids: {extra_seq_ids})"
        )
        dms_seq_ids = get_seq_ids_for_deep_mutational_scan(
            wt_aa_seq, dms_starting_seq_ids, extra_seq_ids
        )
        logging.info(f"Will be embedding {len(dms_seq_ids)} sequences")

        # 5. Import ESM and create client.
        logging.info(f"Importing ESM and creating client for {embedding_model}")

        gpu_available = get_torch_cuda_is_available_and_add_logs(logging.info)

        foldy_esm_client = FoldyPLMClient.get_client(embedding_model)

        def build_embedding_dict(seq_id, seq, embedding_list):
            output_dict = {
                "seq_id": seq_id,
                "seq": seq,
                "embedding": json.dumps(embedding_list[-1]),
            }
            for extra_layer_idx, extra_layer_embedding in zip(extra_layers, embedding_list[:-1]):
                output_dict[f"embedding_layer_{extra_layer_idx}"] = json.dumps(
                    extra_layer_embedding
                )
            return output_dict

        sequence_cache: dict[str, str] = {}

        def resolve_sequence(seq_id: str) -> str:
            cached = sequence_cache.get(seq_id)
            if cached is not None:
                return cached
            if is_homolog_seq_id(seq_id):
                if seq_id not in homolog_id_to_seq_map:
                    raise ValueError(
                        f"Full seq id {seq_id} not found in full_seq_id_to_seq_map populated from extra_seq_ids."
                    )
                sequence = homolog_id_to_seq_map[seq_id]
            else:
                sequence = seq_id_to_seq(wt_aa_seq, seq_id)
            sequence_cache[seq_id] = sequence
            return sequence

        def get_cuda_memory_ratio() -> float | None:
            try:
                import torch

                if not torch.cuda.is_available():
                    return None
                torch.cuda.synchronize()
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                if total_bytes <= 0:
                    return None
                return (total_bytes - free_bytes) / total_bytes
            except Exception as exc:
                logging.debug("Could not read CUDA memory utilization: %s", exc)
                return None

        def reset_cuda_peak_memory_stats() -> bool:
            try:
                import torch

                if not torch.cuda.is_available():
                    return False
                torch.cuda.reset_peak_memory_stats()
                return True
            except Exception as exc:
                logging.debug("Could not reset CUDA peak memory stats: %s", exc)
                return False

        def get_cuda_peak_memory_ratio() -> float | None:
            try:
                import torch

                if not torch.cuda.is_available():
                    return None
                torch.cuda.synchronize()
                total_bytes = torch.cuda.get_device_properties(
                    torch.cuda.current_device()
                ).total_memory
                if total_bytes <= 0:
                    return None
                peak_reserved = torch.cuda.max_memory_reserved()
                peak_allocated = torch.cuda.max_memory_allocated()
                peak_bytes = max(peak_reserved, peak_allocated)
                if peak_bytes <= 0:
                    return None
                return peak_bytes / total_bytes
            except Exception as exc:
                logging.debug("Could not read CUDA peak memory utilization: %s", exc)
                return None

        if use_msa_context:
            if msa_a3m:
                storage_manager.write_file(fold.id, MSA_A3M_STORAGE_PATH, msa_a3m)
                embed_record.msa_a3m_path = MSA_A3M_STORAGE_PATH
                embed_record.save()
                stored_a3m_exists = True
            elif stored_a3m_exists and embed_record.msa_a3m_path != MSA_A3M_STORAGE_PATH:
                embed_record.msa_a3m_path = MSA_A3M_STORAGE_PATH
                embed_record.save()

        dynamic_batching = gpu_available and foldy_esm_client.supports_batch_embedding()
        target_gpu_util = os.environ.get("FOLDY_EMBED_TARGET_GPU_UTIL", "0.4")
        vram_limit_env = os.environ.get("FOLDY_EMBED_VRAM_LIMIT", "0.9")
        max_batch_size = os.environ.get("FOLDY_EMBED_MAX_BATCH", "1024")
        try:
            target_gpu_utilization = float(target_gpu_util)
        except ValueError:
            target_gpu_utilization = 0.9
        try:
            vram_utilization_cap = float(vram_limit_env)
        except ValueError:
            vram_utilization_cap = 0.9
        try:
            max_batch_size_int = int(max_batch_size)
        except ValueError:
            max_batch_size_int = 1024
        target_gpu_utilization = min(max(target_gpu_utilization, 0.1), 0.98)
        vram_utilization_cap = min(max(vram_utilization_cap, 0.1), 0.98)
        max_batch_size_int = max(1, max_batch_size_int)
        configured_max_batch_size = max_batch_size_int
        if not dynamic_batching:
            max_batch_size_int = 1
        if dynamic_batching:
            if target_gpu_utilization > vram_utilization_cap:
                logging.info(
                    "Clamping target GPU utilization %.2f to VRAM limit %.2f",
                    target_gpu_utilization,
                    vram_utilization_cap,
                )
                target_gpu_utilization = vram_utilization_cap
            logging.info(
                "Adaptive embedding batch size enabled (target_gpu_util=%.2f, vram_limit=%.2f, max_batch=%d)",
                target_gpu_utilization,
                vram_utilization_cap,
                max_batch_size_int,
            )

        length_aware = dynamic_batching and os.environ.get("FOLDY_EMBED_LENGTH_AWARE", "1") != "0"
        length_bucket_env = os.environ.get("FOLDY_EMBED_LEN_BUCKET", "64")
        try:
            length_bucket_size = int(length_bucket_env)
        except ValueError:
            length_bucket_size = 64
        length_bucket_size = max(1, length_bucket_size)

        def estimate_safe_batch_cap(current_batch_size: int, peak_ratio: float) -> int:
            if current_batch_size <= 0 or peak_ratio <= 0:
                return configured_max_batch_size
            per_item_ratio = peak_ratio / current_batch_size
            if per_item_ratio <= 0:
                return configured_max_batch_size
            return max(1, int(vram_utilization_cap / per_item_ratio))

        total_sequences = len(dms_seq_ids)
        sequence_order = list(range(total_sequences))
        seq_lengths = [0] * total_sequences
        if length_aware:
            for idx, seq_id in enumerate(dms_seq_ids):
                seq_lengths[idx] = len(resolve_sequence(seq_id))
            sequence_order.sort(key=lambda idx: (seq_lengths[idx], idx))
            logging.info("Length-aware batching enabled (len_bucket=%d)", length_bucket_size)

        embedding_dicts: list[dict[str, str] | None] = [None] * total_sequences

        temp_dir_context = tempfile.TemporaryDirectory() if use_msa_context else nullcontext()
        with temp_dir_context as temp_dir:
            msa_a3m_temp_path = None
            if use_msa_context:
                assert temp_dir is not None
                msa_a3m_contents = msa_a3m
                if msa_a3m_contents is None and not stored_context_exists and stored_a3m_exists:
                    msa_a3m_contents = load_stored_msa_a3m(storage_manager, fold.id)
                if msa_a3m_contents:
                    msa_a3m_temp_path = os.path.join(temp_dir, "msa.a3m")
                    with open(msa_a3m_temp_path, "w") as msa_file:
                        msa_file.write(msa_a3m_contents)

                import torch

                msa_context_contents = load_or_sample_msa_context(
                    fsm,
                    fold.id,
                    msa_a3m_temp_path,
                    torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                )
                msa_context_temp_path = os.path.join(temp_dir, "msa_context.txt")
                with open(msa_context_temp_path, "w") as msa_file:
                    msa_file.write(msa_context_contents)
                msa_a3m_temp_path = msa_context_temp_path

            processed = 0
            next_progress_log = 0
            next_memory_log = 0
            batch_size = 1
            progress_log_interval = 100
            last_progress_log_time = None
            last_progress_log_count = None
            recent_progress_samples: list[tuple[float, int]] = []
            current_bucket = None
            bucket_end = total_sequences

            while processed < total_sequences:
                if length_aware:
                    current_index = sequence_order[processed]
                    bucket = seq_lengths[current_index] // length_bucket_size
                    if bucket != current_bucket:
                        current_bucket = bucket
                        batch_size = 1
                        bucket_end = processed
                        while (
                            bucket_end < total_sequences
                            and seq_lengths[sequence_order[bucket_end]] // length_bucket_size
                            == current_bucket
                        ):
                            bucket_end += 1
                        logging.info(
                            "Length-aware batching: bucket %d (seq_len<=%d), resetting batch size",
                            current_bucket,
                            (current_bucket + 1) * length_bucket_size,
                        )
                else:
                    bucket_end = total_sequences

                current_batch_size = min(batch_size, max_batch_size_int, bucket_end - processed)
                batch_indices = sequence_order[processed : processed + current_batch_size]
                batch_seq_ids = [dms_seq_ids[idx] for idx in batch_indices]
                batch_sequences = [resolve_sequence(seq_id) for seq_id in batch_seq_ids]

                peak_memory_ratio = None
                peak_stats_enabled = False
                if dynamic_batching:
                    peak_stats_enabled = reset_cuda_peak_memory_stats()

                try:
                    batch_embeddings = foldy_esm_client.embed_batch(
                        batch_sequences,
                        extra_layers=extra_layers,
                        domain_boundaries=domain_boundaries,
                        use_msa_context=use_msa_context,
                        msa_a3m_path=msa_a3m_temp_path,
                    )
                except RuntimeError as exc:
                    if dynamic_batching and "out of memory" in str(exc).lower():
                        if current_batch_size <= 1:
                            raise
                        logging.warning(
                            "CUDA OOM at batch_size=%d; reducing and retrying.",
                            current_batch_size,
                        )
                        clean_up_torch_memory()
                        batch_size = max(1, current_batch_size // 2)
                        if batch_size < max_batch_size_int:
                            max_batch_size_int = batch_size
                            logging.info(
                                "Capping embedding batch size to %d after OOM.",
                                max_batch_size_int,
                            )
                        continue
                    raise

                if peak_stats_enabled:
                    peak_memory_ratio = get_cuda_peak_memory_ratio()
                    if peak_memory_ratio is not None:
                        safe_batch_cap = min(
                            max_batch_size_int,
                            estimate_safe_batch_cap(current_batch_size, peak_memory_ratio),
                        )
                        if safe_batch_cap < max_batch_size_int:
                            max_batch_size_int = safe_batch_cap
                            logging.info(
                                "Capping embedding batch size to %d based on peak_vram_ratio=%.2f (limit=%.2f).",
                                max_batch_size_int,
                                peak_memory_ratio,
                                vram_utilization_cap,
                            )

                for idx, seq_id, sequence, embedding_list in zip(
                    batch_indices, batch_seq_ids, batch_sequences, batch_embeddings
                ):
                    embedding_dicts[idx] = build_embedding_dict(seq_id, sequence, embedding_list)

                processed += current_batch_size

                while processed >= next_progress_log:
                    log_count = min(next_progress_log, total_sequences)
                    now = time.monotonic()
                    eta_text = None
                    if (
                        last_progress_log_time is not None
                        and last_progress_log_count is not None
                        and log_count > last_progress_log_count
                    ):
                        elapsed = now - last_progress_log_time
                        delta_count = log_count - last_progress_log_count
                        if elapsed > 0 and delta_count > 0:
                            recent_progress_samples.append((elapsed, delta_count))
                            if len(recent_progress_samples) > 3:
                                recent_progress_samples.pop(0)
                            total_elapsed = sum(sample[0] for sample in recent_progress_samples)
                            total_count = sum(sample[1] for sample in recent_progress_samples)
                            avg_seconds_per_item = (
                                total_elapsed / total_count if total_count > 0 else None
                            )
                            remaining = total_sequences - log_count
                            if avg_seconds_per_item is not None:
                                eta_seconds = remaining * avg_seconds_per_item
                                eta_text = str(timedelta(seconds=int(max(0, eta_seconds))))
                    if eta_text:
                        logging.info(
                            "Finished embedding %d/%d (ETA %s)",
                            log_count,
                            total_sequences,
                            eta_text,
                        )
                    else:
                        logging.info("Finished embedding %d/%d", log_count, total_sequences)
                    last_progress_log_time = now
                    last_progress_log_count = log_count
                    next_progress_log += progress_log_interval

                while processed >= next_memory_log:
                    log_memory_usage()
                    clean_up_torch_memory()
                    next_memory_log += 2000

                if dynamic_batching:
                    memory_ratio_source = "peak"
                    memory_ratio = peak_memory_ratio
                    if memory_ratio is None:
                        memory_ratio_source = "current"
                        memory_ratio = get_cuda_memory_ratio()
                    if memory_ratio is not None:
                        next_batch_size = current_batch_size
                        if (
                            memory_ratio < target_gpu_utilization * 0.9
                            and current_batch_size < max_batch_size_int
                        ):
                            next_batch_size = min(max_batch_size_int, current_batch_size * 2)
                        elif (
                            memory_ratio > target_gpu_utilization * 1.05 and current_batch_size > 1
                        ):
                            next_batch_size = max(1, current_batch_size // 2)
                        if next_batch_size != current_batch_size:
                            logging.info(
                                "Adjusting embedding batch size %d -> %d (gpu_mem_ratio=%.2f, source=%s)",
                                current_batch_size,
                                next_batch_size,
                                memory_ratio,
                                memory_ratio_source,
                            )
                        batch_size = next_batch_size
                    else:
                        batch_size = current_batch_size

        if any(entry is None for entry in embedding_dicts):
            missing = sum(1 for entry in embedding_dicts if entry is None)
            raise RuntimeError(f"Missing embeddings for {missing} sequences")

        embedding_df = pd.DataFrame(embedding_dicts)

        # Convert the DataFrame to a CSV string
        csv_buffer = StringIO()
        embedding_df.to_csv(csv_buffer, index=False)  # Use index=False to exclude the index
        embedding_csv_string = csv_buffer.getvalue()

        # Create a FoldStorageManager and store the embeddings.
        padded_fold_id = "%06d" % fold.id
        embedding_path = f"embed/{padded_fold_id}_embeddings_{embedding_model}_{embed_name}.csv"

        logging.info(f"Saving output to {embedding_path}")
        storage_manager.write_file(fold.id, embedding_path, embedding_csv_string)

        # Try writing homolog_fasta to file.
        if homolog_fasta:
            try:
                homolog_fasta_path = f"embed/{padded_fold_id}_embeddings_{embedding_model}_{embed_name}_homologs.fasta"
                storage_manager.write_file(fold.id, homolog_fasta_path, homolog_fasta)
            except Exception as e:
                logging.error(f"Error writing homolog fasta to file: {e}")

        embed_record.output_fpath = embedding_path
        embed_record.save()


def get_esm_naturalness(naturalness_id: int, msa_a3m: str | None = None):
    """Compute the ESM naturalness and store them with the storage manager.

    Arguments:
        naturalness_id: ID of the naturalness record to run.
    """
    naturalness_record = Naturalness.get_by_id(naturalness_id)
    if not naturalness_record:
        raise KeyError(f"Naturalness ID {naturalness_id} not found!")

    naturalness_name = naturalness_record.name
    naturalness_model = naturalness_record.logit_model
    use_msa_context = bool(naturalness_record.use_msa_context)
    fold = naturalness_record.fold
    if not fold:
        raise KeyError(
            f"Naturalness ID {naturalness_id} ({naturalness_name}) does not have an associated fold!"
        )
    invokation = Invokation.get_by_id(naturalness_record.invokation_id)
    if not invokation:
        raise KeyError(
            f"Naturalness ID {naturalness_id} ({naturalness_name}) does not have an associated invokation!"
        )

    with LoggingRecorder(invokation):
        logging.info("Starting naturalness...")

        fsm = FoldStorageManager()
        fsm.setup()
        storage_manager = fsm.storage_manager
        if storage_manager is None:
            raise RuntimeError("Storage manager is not initialized.")
        gpu_available = get_torch_cuda_is_available_and_add_logs(logging.info)

        # 3. Validate seq_ids.
        if not fold.yaml_config:
            raise ValueError("Fold does not have a YAML config!")
        boltz_yaml_helper = BoltzYamlHelper(fold.yaml_config)

        # 3. Validate seq_ids.
        if len(boltz_yaml_helper.get_protein_sequences()) > 1:
            raise ValueError(
                "Fold has multiple protein sequences, which is not supported for ESM embeddings yet."
            )
        wt_aa_seq = boltz_yaml_helper.get_protein_sequences()[0][1]

        stored_context_exists = False
        stored_a3m_exists = False
        if use_msa_context:
            stored_context_exists = has_stored_msa_context(storage_manager, fold.id)
            stored_a3m_exists = has_stored_msa_a3m(storage_manager, fold.id)
            if not naturalness_model.startswith("e1_"):
                raise BadRequest("MSA context is only supported for E1 naturalness models.")
            if msa_a3m:
                msa_a3m = normalize_msa_a3m_contents(msa_a3m, wt_aa_seq)
                validate_msa_a3m_contents(msa_a3m, wt_aa_seq)
            elif not (stored_context_exists or stored_a3m_exists):
                raise BadRequest("msa_a3m is required when use_msa_context=true.")

        protein_input = wt_aa_seq

        get_depth_two_logits = naturalness_record.get_depth_two_logits or False

        if use_msa_context:
            if msa_a3m:
                storage_manager.write_file(fold.id, MSA_A3M_STORAGE_PATH, msa_a3m)
                naturalness_record.msa_a3m_path = MSA_A3M_STORAGE_PATH
                naturalness_record.save()
                stored_a3m_exists = True
            elif stored_a3m_exists and naturalness_record.msa_a3m_path != MSA_A3M_STORAGE_PATH:
                naturalness_record.msa_a3m_path = MSA_A3M_STORAGE_PATH
                naturalness_record.save()

        with tempfile.TemporaryDirectory() as temp_dir:
            msa_a3m_temp_path = None
            if use_msa_context:
                msa_a3m_contents = msa_a3m
                if msa_a3m_contents is None and not stored_context_exists and stored_a3m_exists:
                    msa_a3m_contents = load_stored_msa_a3m(storage_manager, fold.id)
                if msa_a3m_contents:
                    msa_a3m_temp_path = os.path.join(temp_dir, "msa.a3m")
                    with open(msa_a3m_temp_path, "w") as msa_file:
                        msa_file.write(msa_a3m_contents)

                import torch

                msa_context_contents = load_or_sample_msa_context(
                    fsm,
                    fold.id,
                    msa_a3m_temp_path,
                    torch.device("cuda" if gpu_available else "cpu"),
                )
                msa_context_temp_path = os.path.join(temp_dir, "msa_context.txt")
                with open(msa_context_temp_path, "w") as msa_file:
                    msa_file.write(msa_context_contents)
                msa_a3m_temp_path = msa_context_temp_path
            if naturalness_record.use_structure:
                pdb_binary = storage_manager.get_binary(fold.id, "ranked_0.cif")
                with open(os.path.join(temp_dir, "ranked_0.cif"), "wb") as f:
                    f.write(pdb_binary)
                cif_file_path = os.path.join(temp_dir, "ranked_0.cif")
            else:
                cif_file_path = None

            if naturalness_model == "esm1v_t33_650M_UR90S_ensemble":
                logits_dicts_list = []
                melted_df_list = []
                for ii in range(1, 6):
                    submodel = f"esm1v_t33_650M_UR90S_{ii}"
                    logits_json, melted_df = get_naturalness(
                        protein_input,
                        submodel,
                        get_depth_two_logits,
                        cif_file_path,
                        use_msa_context=use_msa_context,
                        msa_a3m_path=msa_a3m_temp_path,
                    )
                    logits_dicts_list.append(json.loads(logits_json))
                    melted_df_list.append(melted_df.assign(model=ii))
                logits_json = json.dumps(logits_dicts_list)
                melted_df = pd.concat(melted_df_list)
            else:
                logits_json, melted_df = get_naturalness(
                    protein_input,
                    naturalness_model,
                    get_depth_two_logits,
                    cif_file_path,
                    use_msa_context=use_msa_context,
                    msa_a3m_path=msa_a3m_temp_path,
                )

        melted_csv_buffer = StringIO()
        melted_df.to_csv(melted_csv_buffer, index=False)
        melted_csv_string = melted_csv_buffer.getvalue()

        # Save both formats using FoldStorageManager
        logging.info("Saving naturalness to storage")
        logits_path = f"naturalness/naturalness_{naturalness_name}.json"
        melted_path = f"naturalness/naturalness_{naturalness_name}_melted.csv"

        storage_manager.write_file(fold.id, logits_path, logits_json)
        storage_manager.write_file(fold.id, melted_path, melted_csv_string)

        # Update the naturalness record with the output file path
        naturalness_record.output_fpath = melted_path
        naturalness_record.save()

        logging.info("Naturalness computation and storage complete")


def finetune_esm_model(few_shot_id: int):
    """Run the evolvepro workflow."""

    few_shot = FewShot.get_by_id(few_shot_id)
    if not few_shot:
        raise BadRequest(f"FewShot {few_shot_id} not found")
    fold = Fold.get_by_id(few_shot.fold_id)
    if not fold:
        raise BadRequest(f"Fold {few_shot.fold_id} not found")
    invokation = Invokation.get_by_id(few_shot.invokation_id)
    if not invokation:
        raise BadRequest(f"Invokation {few_shot.invokation_id} not found")

    with LoggingRecorder(invokation):
        logging.info("Starting finetuning...")

        logging.info("Loading training code.")
        import torch

        from app.helpers.finetuning.training import score_sequences, train_per_protein

        if not fold.yaml_config:
            raise ValueError("Fold does not have a YAML config!")
        boltz_yaml_helper = BoltzYamlHelper(fold.yaml_config)
        if len(boltz_yaml_helper.get_protein_sequences()) != 1:
            raise ValueError(
                f"Fold has {len(boltz_yaml_helper.get_protein_sequences())} protein sequences, which is not supported for evolvepro yet."
            )
        wt_aa_seq = boltz_yaml_helper.get_protein_sequences()[0][1]

        fsm = FoldStorageManager()
        fsm.setup()
        storage_manager = fsm.storage_manager
        if storage_manager is None:
            raise RuntimeError("Storage manager is not initialized.")

        # 1. Get the activity file.
        few_shot_directory = Path("few_shots") / few_shot.name
        activity_file_path = few_shot_directory / "activity.xlsx"
        logging.info(f"Getting the activity file {activity_file_path}")
        activity_file = storage_manager.get_binary(few_shot.fold_id, str(activity_file_path))
        raw_activity_df = pd.read_excel(BytesIO(activity_file))

        # 3. Process the activity and embedding data.
        if all([v in raw_activity_df.columns for v in ["sequence", "seq_id_w", "seq_id_l"]]):
            loss = "dpo"
            # TODO: do some validation...
            activity_df = raw_activity_df
            for seq_id in activity_df.seq_id_w.tolist() + activity_df.seq_id_l.tolist():
                for locus in get_loci_set(seq_id):
                    if locus >= 1023:
                        raise BadRequest(
                            f"One of the seq_ids is for a protein that is too big: ESM only goes up to 1024AAs, not {seq_id}."
                        )

        elif all([v in raw_activity_df.columns for v in ["seq_id", "activity"]]):
            loss = "entropy"
            activity_df, _, _ = process_and_validate_evolve_input_files(wt_aa_seq, raw_activity_df)
            # Convert activity_df, which has seq_id and activity, into train and valid dfs with an 80/20 split and columns sequence and label.
            activity_df["sequence"] = activity_df["seq_id"].apply(
                lambda seq_id: seq_id_to_seq(wt_aa_seq, seq_id)
            )
            activity_df["label"] = activity_df["activity"]
        else:
            raise ValueError(f"Activity file has invalid columns, got {raw_activity_df.columns}")
        logging.info(f"Have {activity_df.shape[0]} rows in activity_df")

        if "use_for_validation" in activity_df.columns:
            logging.info(f'Using "use_for_validation" column to split into train and valid')
            train_df = activity_df[activity_df["use_for_validation"] == False]
            valid_df = activity_df[activity_df["use_for_validation"] == True]
        else:
            logging.info(f'No "use_for_validation" column, so splitting randomly')
            train_df = activity_df.sample(frac=0.8, random_state=42)
            valid_df = activity_df.drop(train_df.index)

        logging.info(
            f"Train df has {train_df.shape[0]} rows and valid df has {valid_df.shape[0]} rows"
        )

        gpu_available = get_torch_cuda_is_available_and_add_logs(logging.info)

        epochs = 10
        learning_rate = 3e-4
        possible_params = few_shot.name.split("_")
        for possible_param in possible_params:
            parts = possible_param.split("=")
            if len(parts) == 2:
                key, value = parts
                if key == "epochs":
                    epochs = int(value)
                elif key == "learningrate":
                    learning_rate = float(value)

        # Save model outputs
        padded_fold_id = "%06d" % fold.id
        model_dir = f"few_shots/{few_shot.name}/model"

        # Declare these outside the with block
        tokenizer = None
        model = None
        history = None

        with tempfile.TemporaryDirectory() as temp_dir:
            # Make output directory.
            temp_training_subdir = Path(temp_dir) / "training"
            temp_training_subdir.mkdir(parents=True, exist_ok=True)

            # Example: enable ranking loss
            tokenizer, model, history = train_per_protein(
                checkpoint=few_shot.finetuning_model_checkpoint,
                train_df=train_df,
                valid_df=valid_df,
                device=torch.device("cuda" if gpu_available else "cpu"),
                train_batch_size=10,
                grad_accum_steps=2,
                val_batch_size=10,
                loss=loss,
                epochs=epochs,
                learning_rate=learning_rate,
                seed=42,
                mixed_precision=False,  # This causes an error "Attempting to unscale FP16 gradients" when set to "gpu_available",
                train_full=True,
                output_dir=str(temp_training_subdir),
            )
            logging.info("Finetuning complete.")

            # Save tokenizer and model
            logging.info(f"Saving tokenizer and model to {model_dir}")
            tokenizer.save_pretrained(str(Path(temp_dir) / "tokenizer"))
            model.save_pretrained(str(Path(temp_dir) / "model"))
            storage_manager.upload_folder(fold.id, temp_dir, model_dir)

        # Save training history
        history_json = json.dumps(history)
        storage_manager.write_file(fold.id, f"{model_dir}/history.json", history_json)

        # Get all sequences to score
        logging.info(f"Getting all sequences to score")
        dms_seq_ids = get_seq_ids_for_deep_mutational_scan(wt_aa_seq, ["WT"], [])

        # Score sequences and save results
        logging.info(f"Scoring {len(dms_seq_ids)} sequences")
        scores_df = score_sequences(model, tokenizer, wt_aa_seq, dms_seq_ids)
        scores_fpath = f"few_shots/{few_shot.name}/scores.csv"
        logging.info(f"Saving scores to {scores_fpath}")
        scores_csv = scores_df.to_csv(index=False)
        storage_manager.write_file(fold.id, scores_fpath, scores_csv)

        logging.info(f"Finished finetuning and scoring.")
