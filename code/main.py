import xarray as xr
import numpy as np
import dask.array as da
import pandas as pd
import warnings
from pandas.tseries.offsets import DateOffset
import copy
import os
from pathlib import Path
import utils
import json


def run_avg_per(sst, w=11):
    """
    Calculate the periodic moving average of a given sequence.

    This function calculates the periodic moving average of a given input
    array `sst` using a sliding window of size `w`. The function also handles
    periodicity by extending the input array to the left and right, which
    allows for accurate calculations near the edges of the input data.

    Parameters
    ----------
    sst : numpy.ndarray or list
        The input sequence for which the periodic moving average will be calculated.
    w : int, optional, default=11
        The window size for the moving average.

    Returns
    -------
    sst : numpy.ndarray
        The calculated periodic moving average of the input sequence with the same
        shape as the input.
    """

    var = np.array(sst)
    hw = w // 2

    # Pad the input array with periodic boundary conditions
    var_padded = np.pad(var, pad_width=(hw, hw), mode="wrap")

    # Calculate the moving average using convolution
    avg = np.convolve(var_padded, np.ones(w), "valid") / w

    return avg

def get_win_clim(sst, w=11, year_length=365):
    """
    Calculate day-of-year w-day window rolling mean and smooth with
    a 31-day periodic moving average.

    Parameters
    ----------
    sst : numpy.ndarray or list
        The input sequence for which the rolling percentile will be calculated.
    w : int, optional, default=11
        The window size for the rolling mean.

    Returns
    -------
    fin_run : numpy.ndarray
        The calculated day-of-year rolling percentile smoothed with a 31-day
        periodic moving average.
    """

    if np.isnan(sst[0]):
        return np.array([10000.0] * year_length)

    n = len(sst) // year_length
    var = np.reshape(np.array(sst), (n, year_length))
    hw = w // 2

    # Extend the input array with periodic boundary conditions
    var_ext = np.pad(var, pad_width=((0, 0), (hw, hw)), mode="wrap")

    # Calculate the rolling percentile
    clim = [np.mean(var_ext[:, i : i + 2 * hw + 1]) for i in range(year_length)]

    fin = np.array(clim)
    fin_run = run_avg_per(fin, w=31)

    return fin_run

def get_win_pctl(sst, w=11, p=90, year_length=365):
    """
    Calculate day-of-year w-day window rolling p-percentile and smooth with
    a 31-day periodic moving average.

    Parameters
    ----------
    sst : numpy.ndarray or list
        The input sequence for which the rolling percentile will be calculated.
    w : int, optional, default=11
        The window size for the rolling percentile.
    p : int, optional, default=90
        The percentile value to be calculated within the rolling window.

    Returns
    -------
    fin_run : numpy.ndarray
        The calculated day-of-year rolling percentile smoothed with a 31-day
        periodic moving average.
    """

    if np.isnan(sst[0]):
        return np.array([10000.0] * year_length)

    n = len(sst) // year_length
    var = np.reshape(np.array(sst), (n, year_length))
    hw = w // 2

    # Extend the input array with periodic boundary conditions
    var_ext = np.pad(var, pad_width=((0, 0), (hw, hw)), mode="wrap")

    # Calculate the rolling percentile
    tdh = [np.percentile(var_ext[:, i : i + 2 * hw + 1], p) for i in range(year_length)]

    fin = np.array(tdh)
    fin_run = run_avg_per(fin, w=31)

    return fin_run

def compute_thresholds(
    baseline, window=11, q=90, year_length=365, var="tos", lat="lat", lon="lon"
):
    """
    Compute the rolling q-percentile of the baseline sea surface temperature
    dataset with a specified window size and smooth it using a 31-day periodic moving average.

    Parameters
    ----------
    baseline : xarray.Dataset
        The baseline dataset containing sea surface temperature data.
    window : int, optional, default=11
        The window size for the rolling percentile calculation.
    q : int, optional, default=90
        The percentile value to be calculated within the rolling window.

    Returns
    -------
    ds_thres : xarray.Dataset
        The dataset with the calculated rolling q-percentile smoothed with a 31-day
        periodic moving average.
    """
    baseline_arr = baseline[var].values
    thres = np.apply_along_axis(
        get_win_pctl, 0, baseline_arr, window, q, year_length=year_length
    )
    clim = np.apply_along_axis(
        get_win_clim, 0, baseline_arr, window, year_length=year_length
    )

    baseline = baseline.sortby("time")
    ds_thres = baseline.isel(time=slice(0, year_length))
    ds_thres = ds_thres.groupby("time.dayofyear").mean(dim="time")
    ds_thres = ds_thres.drop_vars(var)
    ds_thres["pctl"] = (("dayofyear", lat, lon), thres)
    ds_thres["clim"] = (("dayofyear", lat, lon), clim)

    ds_thres = ds_thres.where(ds_thres.pctl < 9999)

    return ds_thres

def mhs_to_mhw(mhs, min_days=5, gap=2):
    """
    :mhs: should be an array of 1s and 0s, with 1s corresponding to SST above the threshold.
    """

    mhs = np.array(mhs)
    split_indices = np.where(np.diff(mhs) != 0)[0] + 1
    split_bool = np.split(mhs, split_indices)
    split = copy.deepcopy(split_bool)

    num_splits = len(split_bool)

    # Handle the first group
    if split_bool[0][0] == 1 and len(split_bool[0]) < min_days:
        split[0] = [0] * len(split_bool[0])

    for i in range(1, num_splits - 1):
        current_group = split_bool[i]
        previous_group = split_bool[i - 1]
        next_group = split_bool[i + 1]

        if (
            current_group[0] == 0
            and len(current_group) <= gap
            and len(previous_group) >= min_days
            and len(next_group) >= min_days
        ):
            split[i] = [1] * len(current_group)
        elif current_group[0] == 1 and len(current_group) < min_days:
            split[i] = [0] * len(current_group)

    # Handle the last group
    if split_bool[-1][0] == 1 and len(split_bool[-1]) < min_days:
        split[-1] = [0] * len(split_bool[-1])

    mhw = np.concatenate(split)

    return mhw

def MHW_metrics_cmip(ds, baseline_years, baseline_type, var="tos"):
    year_length = 365
    if ds.time.dtype == "datetime64[ns]":
        # If the dataset has leapyears create a boolean mask to identify leap year February 29th
        leap_year_feb29 = (ds.time.dt.month == 2) & (ds.time.dt.day == 29)
        # Remove leap year February 29th from the dataset
        ds = ds.where(~leap_year_feb29, drop=True)

    else:
        time_index = ds.indexes["time"]
        if hasattr(time_index, "calendar"):
            calendar = time_index.calendar
            if calendar == "360_day":
                year_length = 360

    lat, lon, lat_idx, lon_idx = utils.get_1d_lat_lon(ds)

    y_i = ds.time.dt.year.min().item()
    y_f = ds.time.dt.year.max().item()

    if baseline_type == "fixed_baseline":
        baseline = ds.sel(time=ds.time.dt.year.isin(range(y_i, y_i + baseline_years)))
        thresholds = compute_thresholds(
            baseline, year_length=year_length, var=var, lat=lat_idx, lon=lon_idx
        )

    # Create simple lat and lon coordinates

    # Initialize the output dataset
    ds_out = xr.Dataset(
        {
            "MHS": (
                ["time", "lat", "lon"],
                np.zeros((y_f - y_i - baseline_years + 1, len(lat), len(lon))),
            ),
            "MHW": (
                ["time", "lat", "lon"],
                np.zeros((y_f - y_i - baseline_years + 1, len(lat), len(lon))),
            ),
            "mean_anomaly": (
                ["time", "lat", "lon"],
                np.zeros((y_f - y_i - baseline_years + 1, len(lat), len(lon))),
            ),
            "cumulative_anomaly": (
                ["time", "lat", "lon"],
                np.zeros((y_f - y_i - baseline_years + 1, len(lat), len(lon))),
            ),
        },
        coords={
            "time": np.arange(y_i + baseline_years, y_f + 1),
            "lat": lat,
            "lon": lon,
        },
    )

    # Group the input dataset by year
    grouped_years = ds.groupby("time.year")

    for year, group in grouped_years:
        if year <= y_i + baseline_years - 1:
            continue
        if ds.time.dtype == "datetime64[ns]":
            group = group.where(~(group.time.dt.dayofyear == 366), drop=True)

        print(year, end=" ")

        if baseline_type == "moving_baseline":
            baseline = ds.sel(
                time=ds.time.dt.year.isin(range(year - baseline_years, year))
            )
            thresholds = compute_thresholds(
                baseline, lat=lat_idx, lon=lon_idx, year_length=year_length
            )

        # some code to compute the MHSs per gridcell on that year
        year_thresholds = thresholds.sel(dayofyear=group.time.dt.dayofyear)

        mhs = (group[var] > year_thresholds["pctl"]).where(
            year_thresholds["pctl"].notnull()
        )
        mhw = np.apply_along_axis(mhs_to_mhw, 0, mhs)
        # Update the output dataset

        ds_out["MHS"].loc[{"time": year}] = np.sum(mhs, axis=0)
        ds_out["MHW"].loc[{"time": year}] = np.sum(mhw, axis=0)

        anomaly = (group[var] - year_thresholds["clim"]).values
        anomaly = np.where(mhw == 0, np.nan, anomaly)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_an = np.nanmean(anomaly, axis=0)
            cum_an = np.nansum(anomaly, axis=0)

        ds_out["mean_anomaly"].loc[{"time": year}] = mean_an
        ds_out["cumulative_anomaly"].loc[{"time": year}] = cum_an
        ds_out["cumulative_anomaly"] = ds_out["cumulative_anomaly"].where(
            ds_out["MHS"].notnull()
        )

    return ds_out

def mhw_duration_1d(arr_1d):
    mhw_durations = []
    mhw_duration = 0
    for day in arr_1d:
        if not np.isnan(day):
            if day == 1:
                mhw_duration += 1
            elif mhw_duration >= 5:
                mhw_durations.append(mhw_duration)
                mhw_duration = 0
            else:
                mhw_duration = 0
        else:
            nan_arr = np.empty(63)
            nan_arr[:] = np.nan
            return nan_arr

    if mhw_duration >= 5:
        mhw_durations.append(mhw_duration)

    values = np.empty(63)
    values[:] = np.nan
    values[: len(mhw_durations)] = np.array(mhw_durations)

    return np.array(values)

def mhw_mean_anomaly_1d(arr_1d):
    mhw_ans = []
    mhw_an = 0
    i = 0
    for an in arr_1d:
        if not np.isnan(an):
            i += 1
            mhw_an += an
        elif i >= 5:
            mhw_ans.append(mhw_an / i)
            mhw_an = 0
            i = 0
        else:
            mhw_an = 0
            i = 0

    values = np.empty(63)
    values[:] = np.nan
    values[: len(mhw_ans)] = np.array(mhw_ans)

    return np.array(values)

def find_mhw_durations(arr):
    duration_arr = np.apply_along_axis(mhw_duration_1d, 0, arr)
    return duration_arr

def find_mean_anomaly(arr):
    anomaly_arr = np.apply_along_axis(mhw_mean_anomaly_1d, 0, arr)
    return anomaly_arr

def nonzero_and_not_nan(arr):
    # Replace zeros with NaNs temporarily
    arr_with_nans = np.where(arr == 0, np.nan, arr)

    # Find indices of non-NaN elements
    indices = np.argwhere(~np.isnan(arr_with_nans))

    # Extract non-zero, non-NaN elements
    result = [arr_with_nans[idx[0], idx[1], idx[2]] for idx in indices]

    return result

def MHW_metrics_satellite(
    ds,
    baseline_years,
    baseline_type,
    out_folder="../../results/MHW/satellite/",
    var="tos",
    distribution=False,
    error=False,
):
    year_length = 365
    if ds.time.dtype == "datetime64[ns]":
        # If the dataset has leapyears create a boolean mask to identify leap year February 29th
        leap_year_feb29 = (ds.time.dt.month == 2) & (ds.time.dt.day == 29)
        # Remove leap year February 29th from the dataset
        ds = ds.where(~leap_year_feb29, drop=True)

    else:
        time_index = ds.indexes["time"]
        if hasattr(time_index, "calendar"):
            calendar = time_index.calendar
            if calendar == "360_day":
                year_length = 360

    y_i = ds.time.dt.year.min().item()
    y_f = ds.time.dt.year.max().item()

    if baseline_type == "fixed_baseline":
        baseline = ds.sel(time=ds.time.dt.year.isin(range(y_i, y_i + baseline_years)))
        thresholds = compute_thresholds(baseline, year_length=year_length, var=var)

    lat = ds.lat
    lon = ds.lon

    # Initialize the output dataset
    data_vars = {}
    metrics = [
        "MHS",
        "MHW",
        "MHW_cat_2",
        "MHW_cat_3",
        "MHW_cat_4",
        "mean_anomaly",
        "cumulative_anomaly",
        "mean_duration",
    ]
    for metric in metrics:
        data_vars[metric] = (
            ["time", "lat", "lon"],
            np.zeros((y_f - y_i - baseline_years + 1, len(lat), len(lon))),
        )
        ds_out = xr.Dataset(
            data_vars,
            coords={
                "time": np.arange(y_i + baseline_years, y_f + 1),
                "lat": lat,
                "lon": lon,
            },
        )

    if error != False:
        suffixes = ["pos", "neg"]
        for metric in metrics:
            if metric not in ["MHW_cat_2", "MHW_cat_3", "MHW_cat_4"]:
                for app in suffixes:
                    ds_out[f"{metric}_{app}"] = xr.DataArray(
                        np.zeros((y_f - y_i - baseline_years + 1, len(lat), len(lon))),
                        dims=["time", "lat", "lon"],
                        coords={
                            "time": np.arange(y_i + baseline_years, y_f + 1),
                            "lat": lat,
                            "lon": lon,
                        },
                    )
    grouped_years = ds.groupby("time.year")

    distribution_metrics = [
        "MHS_days_year",
        "MHW_days_year",
        "MHW_cat_2_days_year",
        "MHW_cat_3_days_year",
        "MHW_cat_4_days_year",
        "MHW_event_duration",
        "MHW_anual_cumulative_anomaly",
        "MHW_event_mean_anomaly",
    ]

    ofo = out_folder + f"{baseline_type}_{baseline_years}_year/"
    if not os.path.exists(ofo):
        os.makedirs(ofo)

    for year, group in grouped_years:
        if year <= y_i + baseline_years - 1:
            continue
        if ds.time.dtype == "datetime64[ns]":
            group = group.where(~(group.time.dt.dayofyear == 366), drop=True)

        distributions = {key: [] for key in distribution_metrics}

        print(year, end=", ")

        if os.path.exists(f"{ofo}/MHW_{year}.nc"):
            continue

        if baseline_type == "moving_baseline":
            baseline = ds.sel(
                time=ds.time.dt.year.isin(range(year - baseline_years, year))
            )
            thresholds = compute_thresholds(baseline, year_length=year_length, var=var)
        # some code to compute the MHSs per gridcell on that year
        year_thresholds = thresholds.sel(dayofyear=group.time.dt.dayofyear)

        sst = group[var]

        def get_metrics(
            sst, year_thresholds, year, ds_out, distribution=distribution, app=""
        ):
            mhs = (
                (sst > year_thresholds["pctl"])
                .where(year_thresholds["pctl"].notnull())
                .values
            )
            mhw = np.apply_along_axis(mhs_to_mhw, 0, mhs)
            ds_out[f"MHS{app}"].loc[{"time": year}] = np.sum(mhs, axis=0)
            ds_out[f"MHW{app}"].loc[{"time": year}] = np.sum(mhw, axis=0)

            dif = year_thresholds["pctl"] - year_thresholds["clim"]

            # Computing MHW categories

            mhw_cat_2 = (
                (sst > (year_thresholds["pctl"] + dif))
                .where(year_thresholds["pctl"].notnull())
                .values
            )
            ds_out[f"MHW_cat_2"].loc[{"time": year}] = np.sum(mhw_cat_2, axis=0)

            mhw_cat_3 = (
                (sst > (year_thresholds["pctl"] + 2 * dif))
                .where(year_thresholds["pctl"].notnull())
                .values
            )
            ds_out[f"MHW_cat_3"].loc[{"time": year}] = np.sum(mhw_cat_3, axis=0)

            mhw_cat_4 = (
                (sst > (year_thresholds["pctl"] + 3 * dif))
                .where(year_thresholds["pctl"].notnull())
                .values
            )
            ds_out[f"MHW_cat_4"].loc[{"time": year}] = np.sum(mhw_cat_4, axis=0)

            # Computing anomalies

            anomaly = (sst - year_thresholds["clim"]).values
            anomaly = np.where(mhw == 0, np.nan, anomaly)

            durs = find_mhw_durations(mhw)
            mean_an_event = find_mean_anomaly(anomaly)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean_anomaly = np.nanmean(anomaly, axis=0)
                cumulative_anomaly = np.nansum(anomaly, axis=0)
                mean_duration = np.nanmean(durs, axis=0)

            ds_out[f"mean_anomaly{app}"].loc[{"time": year}] = mean_anomaly
            ds_out[f"cumulative_anomaly{app}"].loc[{"time": year}] = cumulative_anomaly
            ds_out[f"cumulative_anomaly{app}"] = ds_out[
                f"cumulative_anomaly{app}"
            ].where(ds_out[f"MHS{app}"].notnull())
            ds_out[f"mean_duration{app}"].loc[{"time": year}] = mean_duration
            ds_out[f"mean_duration{app}"] = ds_out[f"mean_duration{app}"].where(
                ds_out[f"MHS{app}"].notnull()
            )

            ds_out_year = ds_out.where(ds_out.time == year, drop=True)
            ds_out_year.to_netcdf(f"{ofo}/MHW_{year}.nc")

            if distribution != False:
                distributions["MHW_event_duration"].append(nonzero_and_not_nan(durs))
                distributions["MHW_event_mean_anomaly"].append(
                    nonzero_and_not_nan(mean_an_event)
                )
                distributions["MHS_days_year"].append(
                    nonzero_and_not_nan(ds_out["MHS"].values)
                )
                distributions["MHW_days_year"].append(
                    nonzero_and_not_nan(ds_out["MHW"].values)
                )
                distributions["MHW_cat_2_days_year"].append(
                    nonzero_and_not_nan(ds_out["MHW_cat_2"].values)
                )
                distributions["MHW_cat_3_days_year"].append(
                    nonzero_and_not_nan(ds_out["MHW_cat_3"].values)
                )
                distributions["MHW_cat_4_days_year"].append(
                    nonzero_and_not_nan(ds_out["MHW_cat_4"].values)
                )
                distributions["MHW_anual_cumulative_anomaly"].append(
                    nonzero_and_not_nan(ds_out["cumulative_anomaly"].values)
                )

                histograms = {}
                bin_params = {
                    "MHS_days_year": {"range": (0, 366), "bin_width": 1},
                    "MHW_days_year": {"range": (0, 366), "bin_width": 1},
                    "MHW_cat_2_days_year": {"range": (0, 366), "bin_width": 1},
                    "MHW_cat_3_days_year": {"range": (0, 366), "bin_width": 1},
                    "MHW_cat_4_days_year": {"range": (0, 366), "bin_width": 1},
                    "MHW_event_mean_anomaly": {"range": (0, 7.01), "bin_width": 0.01},
                    "MHW_anual_cumulative_anomaly": {
                        "range": (0, 1001),
                        "bin_width": 1,
                    },
                    "MHW_event_duration": {"range": (0, 366), "bin_width": 1},
                }

                for metric in [
                    "MHS_days_year",
                    "MHW_days_year",
                    "MHW_cat_2_days_year",
                    "MHW_cat_3_days_year",
                    "MHW_cat_4_days_year",
                    "MHW_event_duration",
                    "MHW_anual_cumulative_anomaly",
                    "MHW_event_mean_anomaly",
                ]:
                    histograms[metric] = {}
                    distributions[metric] = utils.flatten(distributions[metric])
                    bin_edges = np.arange(
                        bin_params[metric]["range"][0],
                        bin_params[metric]["range"][1],
                        bin_params[metric]["bin_width"],
                    )
                    hist, bin_edges = np.histogram(
                        distributions[metric], bins=bin_edges
                    )
                    histograms[metric]["hist"] = [float(i) for i in hist]
                    histograms[metric]["bin_edges"] = [float(i) for i in bin_edges]

                fold_distr = ofo + "distributions/"
                if not os.path.exists(fold_distr):
                    os.makedirs(fold_distr)
                file_distr = fold_distr + f"distr_{year}.json"
                with open(file_distr, "w") as outfile:
                    json.dump(histograms, outfile)

        get_metrics(sst, year_thresholds, year, ds_out, distribution=distribution)

        if error != False:
            sst_pos = sst + group["analysis_error"]
            sst_neg = sst - group["analysis_error"]
            get_metrics(
                sst_pos, year_thresholds, year, ds_out, distribution=False, app="_pos"
            )
            get_metrics(
                sst_neg, year_thresholds, year, ds_out, distribution=False, app="_neg"
            )

    # if distribution == False:
    #     return ds_out, None
    # else:
    #     histograms = {}
    #     bin_params = {'MHS_days_year': {'range': (0, 366), 'bin_width': 1},
    #                   'MHW_days_year': {'range': (0, 366), 'bin_width': 1},
    #                   'MHW_cat_2_days_year': {'range': (0, 366), 'bin_width': 1},
    #                   'MHW_cat_3_days_year': {'range': (0, 366), 'bin_width': 1},
    #                   'MHW_cat_4_days_year': {'range': (0, 366), 'bin_width': 1},
    #                    'MHW_event_mean_anomaly': {'range': (0, 7.01), 'bin_width': 0.01},
    #                    'MHW_anual_cumulative_anomaly': {'range': (0, 1001), 'bin_width': 1},
    #                    'MHW_event_duration': {'range': (0, 366), 'bin_width': 1}}

    #     for metric in ['MHS_days_year', 'MHW_days_year','MHW_cat_2_days_year', 'MHW_cat_3_days_year', 'MHW_cat_4_days_year', 'MHW_event_duration', 'MHW_anual_cumulative_anomaly', 'MHW_event_mean_anomaly']:
    #         histograms[metric] = {}
    #         distributions[metric] = utils.flatten(distributions[metric])
    #         bin_edges = np.arange(bin_params[metric]['range'][0], bin_params[metric]['range'][1],
    #                               bin_params[metric]['bin_width'])
    #         hist, bin_edges = np.histogram(
    #             distributions[metric],
    #             bins=bin_edges)
    #         histograms[metric]['hist'] = [float(i) for i in hist]
    #         histograms[metric]['bin_edges'] =  [float(i) for i in bin_edges]

    #     file_distributions = ofo + 'distributions.json'
    #     with open(file_distributions, 'w') as outfile:
    #         json.dump(histograms, outfile)

    # return ds_out, histograms

def MHW_metrics_one_point(
    ds,
    baseline_years,
    baseline_type,
    year=2020,
    var="analysed_sst",
    error=False,
    window=11,
    q=90,
):
    year_length = 365
    if ds.time.dtype == "datetime64[ns]":
        # If the dataset has leapyears create a boolean mask to identify leap year February 29th
        leap_year_feb29 = (ds.time.dt.month == 2) & (ds.time.dt.day == 29)
        # Remove leap year February 29th from the dataset
        ds = ds.where(~leap_year_feb29, drop=True)

    y_i = ds.time.dt.year.min().item()
    y_f = ds.time.dt.year.max().item()

    def get_threshold(baseline):
        thres = get_win_pctl(
            baseline[var].values, window, q, year_length=year_length
        )
        clim = get_win_clim(baseline[var].values, window, year_length=year_length)
        baseline = baseline.sortby("time")
        ds_out = baseline.isel(time=slice(0, year_length))
        ds_out = ds_out.groupby("time.dayofyear").mean(dim="time")
        ds_out = ds_out.drop_vars(var)
        if error == True:
            ds_out = ds_out.drop_vars("analysis_error")
        ds_out["pctl"] = (("dayofyear"), thres)
        ds_out["clim"] = (("dayofyear"), clim)
        ds_out = ds_out.where(ds_out.pctl < 9999)
        return ds_out

    if baseline_type == "fixed_baseline":
        baseline = ds.sel(time=ds.time.dt.year.isin(range(y_i, y_i + baseline_years)))
        ds_out = get_threshold(baseline)

    if baseline_type == "moving_baseline":
        baseline = ds.sel(time=ds.time.dt.year.isin(range(year - baseline_years, year)))
        ds_out = get_threshold(baseline)

    ds_y = ds.sel(time=ds.time.dt.year.isin([year]))
    ds_y = ds_y.groupby("time.dayofyear").mean(dim="time")
    # ds_y = ds_y.where(~(ds_y.dayofyear == 366), drop=True)

    ds_out["sst"] = (("dayofyear"), ds_y[var].values)
    if error == True:
        ds_out["error"] = (("dayofyear"), ds_y["analysis_error"].values)

        for i, typ in enumerate(['pos', 'neg']):

            sign = (-1)**(i)
            sst = ds_out["sst"].values + sign * ds_out["error"].values
            mhs = (
                (sst > ds_out["pctl"]).where(ds_out["pctl"].notnull()).values
            )
            mhw = mhs_to_mhw(mhs)
            anomaly = (sst - ds_out["clim"]).values

            ds_out['MHS_' + typ] = (('dayofyear'), mhs)
            ds_out['MHW_' + typ] = (('dayofyear'), mhw)
            ds_out['anomaly_' + typ] = (('dayofyear'), anomaly)


    mhs = (ds_out["sst"] > ds_out["pctl"]).where(ds_out["pctl"].notnull()).values
    mhw = mhs_to_mhw(mhs)
    anomaly = (sst - ds_out["clim"]).values
    anomaly = np.where(mhw == 0, np.nan, anomaly)

    ds_out["MHS"] = (("dayofyear"), mhs)
    ds_out["MHW"] = (("dayofyear"), mhw)
    ds_out["anomaly"] = (("dayofyear"), anomaly)

    return ds_out