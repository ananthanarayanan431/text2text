from datetime import datetime, time
from typing import Dict, Optional, Tuple

from softrate.config.schedules import (
    FRIDAY_SCHEDULE,
    MONDAY_SCHEDULE,
    SATURDAY_SCHEDULE,
    SUNDAY_SCHEDULE,
    THURSDAY_SCHEDULE,
    TUESDAY_SCHEDULE,
    WEDNESDAY_SCHEDULE,
)


class scheduleContextGeneration:
    SCHEDULES: Dict[int, Dict[str, str]] = {
        0: MONDAY_SCHEDULE,
        1: TUESDAY_SCHEDULE,
        2: WEDNESDAY_SCHEDULE,
        3: THURSDAY_SCHEDULE,
        4: FRIDAY_SCHEDULE,
        5: SATURDAY_SCHEDULE,
        6: SUNDAY_SCHEDULE,
    }

    @staticmethod
    def _parse_time_range(time_range: str) -> Tuple[time, time]:
        """Parse a time range string (e.g., '06:00-07:00') into start and end times."""
        start_str, end_str = time_range.split("-")
        start_time: time = datetime.strptime(start_str, "%H:%M").time()
        end_time: time = datetime.strptime(end_str, "%H:%M").time()
        return start_time, end_time

    @classmethod
    def get_current_activity(cls) -> Optional[str]:
        """Get the current activity based on the current time and day of the week."""

        current_datetime = datetime.now()
        current_time: time = current_datetime.time()
        current_day = current_datetime.weekday()  # Monday is 0, Sunday is 6

        today_schedule = cls.SCHEDULES.get(current_day, {})

        for time_range, activity in today_schedule.items():
            start_time, end_time = cls._parse_time_range(time_range)
            if start_time <= current_time < end_time:
                return activity

            # (e.g., 23:00-06:00) overnight case
            if start_time > end_time:
                if current_time >= start_time or current_time <= end_time:
                    return activity

        return None

    @classmethod
    def get_schedule_for_day(cls, day: int) -> Dict[str, str]:
        """Get the schedule for a specific day of the week."""
        return cls.SCHEDULES.get(day, {})
