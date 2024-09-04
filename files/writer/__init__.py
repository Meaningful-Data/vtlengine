from datetime import datetime, timedelta


def format_time_period_to_sdmx_representation(str_):
    """
    From VTL (our internal representation) to SDMX time period representation.
    'A': 'nothing to do',
    'S': 'YYYY-Sx',
    'Q': 'YYYY-Qx',
    'M': 'YYYY-MM',
    'W': 'YYYY-Wxx',
    'D': 'YYYY-MM-DD'
    """
    if "S" in str_:  # S
        year_param, semester_param = str_.split('S')
        sdmx_value = year_param + '-S' + semester_param
    elif "Q" in str_:  # Q
        year_param, quarter_param = str_.split('Q')
        sdmx_value = year_param + '-Q' + quarter_param
    elif "M" in str_:  # M
        year_param, month_param = str_.split('M')
        if len(month_param) == 1:
            month_param = '0' + month_param
        sdmx_value = year_param + '-' + month_param
    elif "W" in str_:  # W
        year_param, week_param = str_.split('W')
        if len(week_param) == 1:
            week_param = '0' + week_param
        sdmx_value = year_param + '-W' + week_param
    elif "D" in str_:  # D
        year_param, day_param = str_.split('D')
        base = datetime(int(year_param), 1, 1)
        result = base + timedelta(int(day_param) - 1)
        sdmx_value = result.strftime('%Y-%m-%d')
    else:  # year
        return str_
    return sdmx_value