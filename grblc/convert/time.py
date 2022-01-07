# converts decimal days to UT time.
# Ex: 0.25 to 06:00:00.000, which is a quarter of a day
def dec_to_UT(decimal: float) -> str:

    assert isinstance(decimal, float) or isinstance(decimal, int), "decimal must be of type float or int!"

    decimal -= int(decimal)

    hours = decimal * 24
    hours_decimal = hours - int(hours)
    hours = int(hours)

    minutes = hours_decimal * 60
    minutes_decimal = minutes - int(minutes)
    minutes = int(minutes)

    seconds = minutes_decimal * 60
    seconds_str = "{:.3f}".format(seconds)

    leading_hours = f"{hours}".zfill(2)
    leading_minutes = f"{minutes}".zfill(2)

    # hardcoding leading zero for seconds because of the need for 3 decimal places
    if len(seconds_str) == 5:
        leading_seconds = "0" + seconds_str
    elif len(seconds_str) == 6:
        leading_seconds = seconds_str
    else:
        raise Exception("Error with converting seconds string!")

    return f"{leading_hours}:{leading_minutes}:{leading_seconds}"


# converts UT times to decimal days
# Ex: 06:00:00.000 to 0.25.
def UT_to_dec(yr_time: str) -> str:
    # format: YYYY-MM-DD HH:MM:SS.SSS
    try:
        float(yr_time.split(" ")[1].split(":")[0])
    except:
        raise Exception("Input string must be in the format: YYYY-MM-DD HH:MM:SS.SSS")

    (date, time) = yr_time.split(" ")
    (year, month, day) = date.split("-")
    (hours, minutes, seconds) = [float(num) for num in time.split(":")]
    day = str(int(day) + (((hours * 60 * 60) + (minutes * 60) + seconds) / (24 * 60 * 60)))

    return f"{year}:{month.zfill(2)}:{day}"


# converts a GRB number to the actual date of the detection
# Ex: 010222 to 2001-02-22
def grb_to_date(GRB: str):
    import datetime

    if not GRB.isnumeric():
        GRB = GRB[:-1]
    assert len(GRB) == 6 and GRB.isnumeric(), "Incorrect GRB format; should be in format YYMMDD(X)."

    dt = datetime.datetime.strptime(GRB, "%y%m%d")
    return dt.strftime("%Y-%m-%d")
