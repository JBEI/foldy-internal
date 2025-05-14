# resource_watch_logging.py
import datetime
import logging
import os
import socket
import threading
import time

import psutil

try:
    import pynvml
    pynvml.nvmlInit()
    _GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    _GPU_OK = True
except ModuleNotFoundError:
    _GPU_OK = False                       # still logs CPU/RAM

_LOGGER = logging.getLogger("resource_watch")   # inherit parent handlers

def _fmt(val):
    """Format numbers nicely, use -1 as sentinel for 'not available'."""
    return f"{val:.1f}" if isinstance(val, float) else str(val)

def _sample(delim="\t"):
    ts = datetime.datetime.utcnow().isoformat(timespec="seconds")
    rss  = psutil.Process().memory_info().rss / 1_048_576   # MB
    swap = psutil.swap_memory().used / 1_048_576            # MB
    cpu  = psutil.cpu_percent(interval=None)

    gpu_mem = gpu_util = tcore = tmem = -1
    if _GPU_OK:
        mem  = pynvml.nvmlDeviceGetMemoryInfo(_GPU_HANDLE)
        util = pynvml.nvmlDeviceGetUtilizationRates(_GPU_HANDLE)
        tcore = pynvml.nvmlDeviceGetTemperature(
            _GPU_HANDLE, pynvml.NVML_TEMPERATURE_GPU
        )
        tmem = pynvml.nvmlDeviceGetTemperature(
            _GPU_HANDLE, pynvml.NVML_TEMPERATURE_MEMORY
        )
        gpu_mem  = mem.used / 1_048_576                     # MB
        gpu_util = util.gpu

    fields = [
        ts, socket.gethostname(),
        _fmt(rss), _fmt(swap), _fmt(cpu),
        _fmt(gpu_mem), _fmt(gpu_util),
        _fmt(tcore), _fmt(tmem),
    ]
    return delim.join(fields)

def start_resource_monitor(interval: int = 5, delim: str = "\t"):
    """
    Launch a daemon thread that logs one CSV-ish line every `interval` seconds.
    Returns a threading.Event – call `event.set()` to stop.
    """
    stop_evt = threading.Event()

    def _loop():
        # Optional: emit header once
        header = delim.join([
            "timestamp","host","rss_mb","swap_mb","cpu_pct",
            "gpu_mem_mb","gpu_util_pct","gpu_temp_core_c","gpu_temp_hbm_c"
        ])
        _LOGGER.info(header)
        while not stop_evt.is_set():
            _LOGGER.info(_sample(delim))
            time.sleep(interval)

    t = threading.Thread(target=_loop, daemon=True, name="resource-monitor")
    t.start()
    return stop_evt
