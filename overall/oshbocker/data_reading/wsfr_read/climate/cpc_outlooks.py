import datetime
import enum
from io import StringIO
from typing import Iterator, Literal

import geopandas as gpd
import pandas as pd

from wsfr_read.config import DATA_ROOT
from wsfr_read.sites import read_geospatial, read_metadata

CPC_OUTLOOKS_DIR = DATA_ROOT / "cpc_outlooks"
CPC_CLIMATE_DIVISIONS_GEO_FILE = DATA_ROOT / "cpc_climate_divisions.gpkg"

TEMP_FILENAME_TEMPLATE = "cpcllftd.{year}.dat"
PRECIP_FILENAME_TEMPLATE = "cpcllfpd.{year}.dat"


class Variable(str, enum.Enum):
    """Enum for valid variables for outlook (temperature and precipitation)."""

    TEMP = "temp"
    PRECIP = "precip"


def _table_data_generator(
    year: int, variable: Literal["temp", "precip"] | Variable
) -> Iterator[tuple[datetime.date, StringIO]]:
    """Generator function for reading the data tables in a CPC Outlooks data file."""
    variable = Variable(variable)
    if variable == Variable.TEMP:
        path = CPC_OUTLOOKS_DIR / TEMP_FILENAME_TEMPLATE.format(year=year)
        first_header_substr = "FORECAST TEMPERATURE PERCENTILES"
    elif variable == Variable.PRECIP:
        path = CPC_OUTLOOKS_DIR / PRECIP_FILENAME_TEMPLATE.format(year=year)
        first_header_substr = "FORECAST PRECIPITATION PERCENTILES"
    with path.open("r") as fp:
        buffer = StringIO()
        issue_date = None
        for line in fp:
            if line.startswith("9999"):
                # End of data. Return buffer and exit generator
                buffer.seek(0)
                yield issue_date, buffer
                buffer.close()
                return
            elif first_header_substr in line:
                # Read issue date from first header line
                month, day, year = int(line[:2].strip()), int(line[2:4]), int(line[5:9])
                issue_date = datetime.date(year, month, day)
                if buffer.tell():
                    buffer.seek(0)
                    yield issue_date, buffer
                    buffer.close()
                    buffer = StringIO()
            elif line.startswith("YEAR"):
                # Second header line with columns
                # Skip column header row because the widths don't match the data
                # We'll pass in hardcoded column names
                continue
            else:
                # Add data line to buffer
                buffer.write(line)


TEMP_WIDTHS = (
    (
        4,  # YEAR
        4,  # MN
        4,  # LEAD
        4,  # CD
        5,  # R
    )
    + (6,) * 13  # exceedances
    + (
        6,  # F MEAN
        6,  # C MEAN
        7,  # F SD
        7,  # C SD
    )
)
TEMP_COLUMNS = (
    "YEAR",
    "MN",
    "LEAD",
    "CD",
    "R",
    "98.",
    "95.",
    "90.",
    "80.",
    "70.",
    "60.",
    "50.",
    "40.",
    "30.",
    "20.",
    "10.",
    "5.",
    "2.",
    "F MEAN",
    "C MEAN",
    "F SD",
    "C SD",
)

PRECIP_WIDTHS = (
    (
        4,  # YEAR
        4,  # MN
        4,  # LEAD
        4,  # CD
        5,  # R
    )
    + (6,) * 13  # exceedances
    + (
        6,  # FCST
        6,  # CLIM
        7,  # FCST
        7,  # CLIM
        7,  # POWER
    )
)
PRECIP_COLUMNS = (
    "YEAR",
    "MN",
    "LEAD",
    "CD",
    "R",
    "98.",
    "95.",
    "90.",
    "80.",
    "70.",
    "60.",
    "50.",
    "40.",
    "30.",
    "20.",
    "10.",
    "5.",
    "2.",
    "F MEAN",
    "C MEAN",
    "F SD",
    "C SD",
    "POWER",
)

# Some years have weird widths: 2004, 2006
PRECIP_ALT_WIDTHS = (
    (
        4,  # YEAR
        3,  # MN
        3,  # LEAD
        4,  # CD
        5,  # R
    )
    + (6,) * 13  # exceedances
    + (
        6,  # FCST
        6,  # CLIM
        8,  # FCST
        8,  # CLIM
        8,  # POWER
    )
)


def _read_outlook_for_year(
    year: int, variable: Literal["temp", "precip"] | Variable
) -> pd.DataFrame:
    """Read the full outlook data file for a given calendar year and variable. Use the function
    `read_cpc_outlooks_temp` or `read_cpc_outlooks_precip` instead to properly subset by time."""
    variable = Variable(variable)
    table_gen = _table_data_generator(year=year, variable=variable)
    if variable == Variable.TEMP:
        columns = TEMP_COLUMNS
        widths = TEMP_WIDTHS
    elif variable == Variable.PRECIP:
        columns = PRECIP_COLUMNS
        if year in {2004, 2006}:
            widths = PRECIP_ALT_WIDTHS
        else:
            widths = PRECIP_WIDTHS
    dfs = {}
    for issue_date, buffer in table_gen:
        dfs[pd.to_datetime(issue_date)] = pd.read_fwf(
            buffer, header=None, names=columns, widths=widths
        ).set_index(["YEAR", "MN", "LEAD", "CD"])
    return pd.concat(dfs, names=["issue_date"])


def read_cpc_outlooks_temp(
    issue_date: str, site_id: str | None = None, fy_start_month: int = 10
) -> pd.DataFrame:
    """Read CPC Seasonal Temperature Outlooks available as of a given issue_date. By default,
    this loads data for the water year of that issue_date (starting prior Oct 1) up to the day
    before that issue_date. See documentation from CPC for additional explanation on what columns
    in the loaded data represent.
    https://www.cpc.ncep.noaa.gov/pacdir/NFORdir/HUGEdir2/explanation_fdf.html

    The outlooks are issued for CPC "climate divisions" (a.k.a. "forecast divisions"), identified
    by the "CD" column. Geospatial vector data defining the climate divisions is available on the
    data download page as the `cpc_climate_divisions.gpkg` file and can be loaded with the
    `read_cpc_climate_divisions_geo` function.

    Args:
        issue_date (str | datetime.date | pd.Timestamp): Issue date of forecast. This is used to
            subset the data returned so that the forecast does not use future data. Must be
            something pandas.to_datetime can parse, e.g., the string "2021-03-15".
        site_id (str | None): site_id for a forecast site from the challenge. If provided, will
            subset data to climate divisions that intersect with the specified site's drainage
            basin. Default None will not subset by climate division.
        fy_start_month (int): The start month to load data for. Used to determine the lookback
            window in the previous calendar year to return outlooks starting from.

    Returns:
        pd.DataFrame: Dataframe with outlooks. See documentation for details about the data.
    """
    issue_date = pd.to_datetime(issue_date)
    prev_year_df = _read_outlook_for_year(issue_date.year - 1, Variable.TEMP)
    this_year_df = _read_outlook_for_year(issue_date.year, Variable.TEMP)
    df = pd.concat(
        [
            prev_year_df[prev_year_df.index.get_level_values("MN") >= fy_start_month],
            this_year_df[this_year_df.index.get_level_values("issue_date") < issue_date],
        ]
    )

    if site_id:
        cds = get_climate_divisions_for_site_id(site_id)
        return df.loc[pd.IndexSlice[:, :, :, :, cds]]

    return df


def read_cpc_outlooks_precip(
    issue_date: str, site_id: str | None = None, fy_start_month: int = 10
) -> pd.DataFrame:
    """Read CPC Seasonal Precipitation Outlooks available as of a given issue_date. By default,
    this loads data for the water year of that issue_date (starting prior Oct 1) up to the day
    before that issue_date. See documentation from CPC for additional explanation on what columns
    in the loaded data represent.
    https://www.cpc.ncep.noaa.gov/pacdir/NFORdir/HUGEdir2/explanation_fdf.html

    The outlooks are issued for CPC "climate divisions" (a.k.a. "forecast divisions"), identified
    by the "CD" column. Geospatial vector data defining the climate divisions is available on the
    data download page as the `cpc_climate_divisions.gpkg` file and can be loaded with the
    `read_cpc_climate_divisions_geo` function.

    Args:
        issue_date (str | datetime.date | pd.Timestamp): Issue date of forecast. This is used to
            subset the data returned so that the forecast does not use future data. Must be
            something pandas.to_datetime can parse, e.g., the string "2021-03-15".
        site_id (str | None): site_id for a forecast site from the challenge. If provided, will
            subset data to climate divisions that intersect with the specified site's drainage
            basin. Default None will not subset by climate division.
        fy_start_month (int): The start month to load data for. Used to determine the lookback
            window in the previous calendar year to return outlooks starting from.

    Returns:
        pd.DataFrame: Dataframe with outlooks. See documentation for details about the data.
    """
    issue_date = pd.to_datetime(issue_date)
    prev_year_df = _read_outlook_for_year(issue_date.year - 1, Variable.PRECIP)
    this_year_df = _read_outlook_for_year(issue_date.year, Variable.PRECIP)
    df = pd.concat(
        [
            prev_year_df[prev_year_df.index.get_level_values("MN") >= fy_start_month],
            this_year_df[this_year_df.index.get_level_values("issue_date") < issue_date],
        ]
    )

    if site_id:
        cds = get_climate_divisions_for_site_id(site_id)
        return df.loc[pd.IndexSlice[:, :, :, :, cds]]

    return df


def read_cpc_climate_divisions_geo() -> gpd.GeoDataFrame:
    """Read geospatial vector data for the CPC climate divisions, also called forecast divisions.
    These are geographical regions for which the CPC Outlooks correspond to. The climate division
    is identifier is in the "cd" column.
    """
    gdf = gpd.read_file(CPC_CLIMATE_DIVISIONS_GEO_FILE)
    return gdf


def get_climate_divisions_for_site_id(site_id: str) -> list[int]:
    """Returns a list of CPC climate divisions that spatially intersect with a site_id's drainage
    basin."""
    metadata_df = read_metadata()
    basins_gdf = read_geospatial("basins")
    climate_divisions_gdf = read_cpc_climate_divisions_geo()
    # Inner join to get gdf row for this site_id
    this_site_basin_gdf = basins_gdf.merge(
        metadata_df.loc[[site_id], []], how="inner", left_on="site_id", right_on="site_id"
    )
    # Inner spatial join to get climate divisions intersecting this drainage basin
    this_site_divisions_gdf = climate_divisions_gdf.sjoin(
        this_site_basin_gdf[["geometry"]], how="inner"
    )
    return this_site_divisions_gdf["CD"].values.tolist()
