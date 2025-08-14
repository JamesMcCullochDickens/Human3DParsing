import timeit
import statistics


def time_to_mins(t1: int, t2: int) -> float:
    dif = abs(t2-t1)
    mins = round(dif/60.0, 4)
    return mins


def time_to_hours(t1: int, t2: int) -> float:
    dif = abs(t2-t1)
    hours = round(dif/(60*60), 6)
    return hours
