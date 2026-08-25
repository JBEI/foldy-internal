"""Optional native acceleration for FolDE campaign mutant-pool filtering."""

from __future__ import annotations

import ctypes
import logging
from pathlib import Path
from typing import Callable, List, Sequence, Set

from app.helpers.sequence_util import get_allele_set, is_homolog_seq_id

logger = logging.getLogger(__name__)

_LIB_PATH = Path(__file__).resolve().parent / "rust_ext" / "libfoldy_mutant_pool.so"
_NATIVE_LIB: ctypes.CDLL | None = None


def _load_native_lib() -> ctypes.CDLL | None:
    global _NATIVE_LIB
    if _NATIVE_LIB is not None:
        return _NATIVE_LIB
    if not _LIB_PATH.exists():
        return None
    try:
        lib = ctypes.CDLL(str(_LIB_PATH))
        lib.foldy_get_mutant_pool.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        lib.foldy_get_mutant_pool.restype = ctypes.c_void_p
        lib.foldy_free_string.argtypes = [ctypes.c_void_p]
        lib.foldy_free_string.restype = None
    except OSError as exc:
        logger.warning(f"Failed to load native mutant-pool library at {_LIB_PATH}: {exc}")
        return None
    _NATIVE_LIB = lib
    return lib


def native_available() -> bool:
    return _load_native_lib() is not None


def get_mutant_pool_python(
    seq_ids: Sequence[str],
    measured_seq_ids: Sequence[str],
    allele_set_getter: Callable[[str], Set[str]] = get_allele_set,
) -> List[str]:
    measured_seq_id_set = set(measured_seq_ids)
    measured_allele_sets = [allele_set_getter(s) for s in measured_seq_ids]

    mutant_pool = []
    for seq_id in seq_ids:
        if seq_id in measured_seq_id_set:
            continue

        if is_homolog_seq_id(seq_id):
            mutant_pool.append(seq_id)
            continue

        allele_set = allele_set_getter(seq_id)
        if len(allele_set) == 1:
            mutant_pool.append(seq_id)
            continue

        for measured_allele_set in measured_allele_sets:
            if len(allele_set ^ measured_allele_set) == 1:
                mutant_pool.append(seq_id)
                break

    return mutant_pool


def get_mutant_pool_native(seq_ids: Sequence[str], measured_seq_ids: Sequence[str]) -> List[str]:
    lib = _load_native_lib()
    if lib is None:
        raise RuntimeError(f"Native mutant-pool library is not built at {_LIB_PATH}")

    seq_ids_bytes = ("\n".join(seq_ids) + "\n").encode("utf-8")
    measured_seq_ids_bytes = ("\n".join(measured_seq_ids) + "\n").encode("utf-8")
    result_ptr = lib.foldy_get_mutant_pool(seq_ids_bytes, measured_seq_ids_bytes)
    if not result_ptr:
        raise RuntimeError("Native mutant-pool library returned a null result")

    try:
        result_bytes = ctypes.string_at(result_ptr)
    finally:
        lib.foldy_free_string(result_ptr)

    result = result_bytes.decode("utf-8")
    if result == "":
        return []
    return result.split("\n")


def get_mutant_pool(seq_ids: Sequence[str], measured_seq_ids: Sequence[str]) -> List[str]:
    if _load_native_lib() is None:
        return get_mutant_pool_python(seq_ids, measured_seq_ids)
    return get_mutant_pool_native(seq_ids, measured_seq_ids)
