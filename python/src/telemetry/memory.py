"""Memory monitoring utilities for GPU VRAM and host RAM tracking."""

import subprocess
import ctypes
from typing import Dict, Any


class MemoryTracker:
    """Captures GPU VRAM (via nvidia-smi) and host system RAM (via Windows API / ctypes)."""

    @staticmethod
    def get_gpu_vram() -> Dict[str, Any]:
        """Queries nvidia-smi for current GPU memory utilization in MiB."""
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL,
                timeout=3,
            ).decode().strip()
            lines = out.splitlines()
            if lines:
                used_str, total_str = lines[0].split(",")
                used_mib = float(used_str.strip())
                total_mib = float(total_str.strip())
                return {
                    "used_mib": round(used_mib, 1),
                    "total_mib": round(total_mib, 1),
                    "percent": round((used_mib / total_mib * 100.0) if total_mib > 0 else 0.0, 1),
                    "available": True,
                }
        except Exception:
            pass

        return {"used_mib": 0.0, "total_mib": 0.0, "percent": 0.0, "available": False}

    @staticmethod
    def get_system_ram() -> Dict[str, Any]:
        """Queries host system RAM via Windows GlobalMemoryStatusEx."""
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))

            total_gb = stat.ullTotalPhys / (1024**3)
            avail_gb = stat.ullAvailPhys / (1024**3)
            used_gb = total_gb - avail_gb

            return {
                "used_gb": round(used_gb, 2),
                "total_gb": round(total_gb, 2),
                "percent": round((used_gb / total_gb * 100.0) if total_gb > 0 else 0.0, 1),
                "available": True,
            }
        except Exception:
            return {"used_gb": 0.0, "total_gb": 0.0, "percent": 0.0, "available": False}

    @classmethod
    def sample(cls) -> Dict[str, Any]:
        """Take a snapshot of both GPU VRAM and host RAM."""
        return {
            "gpu_vram": cls.get_gpu_vram(),
            "system_ram": cls.get_system_ram(),
        }

