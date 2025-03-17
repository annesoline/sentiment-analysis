import pandas as pd


def parse_dates(dataframe: pd.DataFrame, date_col: str, format_str: str) -> pd.DataFrame:
    """
    Parses a date column in a pandas DataFrame to datetime objects.

    Args:
        dataframe: The pandas DataFrame containing the date column.
        date_col: The name of the column containing the date strings.
        format: The format string specifying the date format. (e.g., "%Y-%m-%d")

    Returns:
        The modified pandas DataFrame with the date column converted to datetime objects.

    Raises:
        ValueError: If any date string cannot be parsed according to the specified format.
    """
    dataframe[date_col] = pd.to_datetime(dataframe[date_col], format=format_str, errors="raise")
    return dataframe
