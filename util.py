from datetime import datetime


def datetime_to_index(
    dt: datetime, step_size: int = 900, steps_per_year: int = 35135
) -> int:
    dif = dt - dt.replace(month=1, day=1, hour=0, minute=0, second=0)
    dif = dif.total_seconds()
    idx = int(dif // step_size) % steps_per_year

    return idx
