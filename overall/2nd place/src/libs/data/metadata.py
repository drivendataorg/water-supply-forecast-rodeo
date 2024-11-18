from datetime import date
import calendar


def day_end_of_month(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])

def date_to_doy(ref_date: date, curr_date: date):
    return (curr_date - ref_date).days


