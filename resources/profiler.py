import time
import os
import psutil

try:
    import torch
except ImportError:
    torch = None

class Profiler:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.gpu_available = torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available()

    def start(self):
        self.start_time = time.time()
        self.cpu_start = self.process.cpu_percent(interval=None)
        self.mem_start = self.process.memory_info().rss / (1024 ** 2)  # in MB
        if self.gpu_available:
            self.gpu_mem_start = torch.cuda.memory_allocated()
        else:
            self.gpu_mem_start = None

    def stop(self):
        self.end_time = time.time()
        self.cpu_end = self.process.cpu_percent(interval=None)
        self.mem_end = self.process.memory_info().rss / (1024 ** 2)  # in MB
        if self.gpu_available:
            self.gpu_mem_end = torch.cuda.memory_allocated()
        else:
            self.gpu_mem_end = None

        return {
            "cpu_percent": self.cpu_end - self.cpu_start,
            "memory_mb": self.mem_end - self.mem_start,
            "inference_time_sec": self.end_time - self.start_time,
            "gpu_memory_bytes": (self.gpu_mem_end - self.gpu_mem_start) if self.gpu_mem_start is not None else None
        }