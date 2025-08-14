import psutil # pip install psutil


def is_ram_usage_high(threshold_percent: float = 80) -> bool:
    """
    Checks if the system's RAM usage exceeds the given threshold percentage.

    Args:
        threshold_percent: The maximum allowed RAM usage percentage (e.g., 80).

    Returns:
        True if RAM usage is above the threshold, False otherwise.
    """
    ram_info = psutil.virtual_memory()
    used_percent = ram_info.percent
    return used_percent > threshold_percent
