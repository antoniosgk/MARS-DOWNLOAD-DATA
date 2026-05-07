#%%
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pyreadr
import xarray as xr

# ============================================================
# SETTINGS
# ============================================================
coords_path = "/home/agkiokas/CAMS/Chile_cords.Rdata"

base_path = "/mnt/store02/agkiokas/data/GLOBE"
output_base = "/home/agkiokas/MARS/plots/"

run_names = ["0001", "icki"]
all_parameters_cifs = ["go3", "no2", "co", "so2"]

ndays = 90
n_times_per_day = 8

# 3 forecast files per initialization day: 1st, 3rd, 5th day
forecast_leads = [1, 3, 5]
n_members = len(forecast_leads)

init_start = datetime(2025, 12, 1)
cycle_folder = "00_00_00"

# R level 25 -> Python index 24
level_index = 24

# R array slot 3 -> Python index 2
var_slot_index = 2

os.makedirs(output_base, exist_ok=True)

# ============================================================
# LOAD CHILE STATION COORDINATES
# ============================================================
rdata = pyreadr.read_r(coords_path)
print("Objects in RData:", list(rdata.keys()))

if "all_cordc" not in rdata:
    raise KeyError(f"'all_cordc' not found. Available objects: {list(rdata.keys())}")

all_cordc = rdata["all_cordc"]

# R:
# x1x = all_cordc[,2]
# y1x = all_cordc[,1]
x1x = all_cordc.iloc[:, 1].to_numpy(dtype=float)
y1x = all_cordc.iloc[:, 0].to_numpy(dtype=float)

n_stations = len(x1x)
print(f"Number of Chile stations: {n_stations}")

# ============================================================
# HELPERS
# ============================================================
def detect_coord_name(ds, candidates):
    for c in candidates:
        if c in ds.coords or c in ds.variables or c in ds.dims:
            return c
    raise KeyError(
        f"Could not find any of {candidates}. "
        f"Available dims: {list(ds.dims)} | variables: {list(ds.variables)}"
    )


def normalize_longitude_da(da, lon_name):
    lon = da[lon_name]
    lon_new = xr.where(lon > 180, lon - 360, lon)
    da = da.assign_coords({lon_name: lon_new})
    return da.sortby(lon_name)


def make_nc_path(run_name, init_date, lead_day):
    target_date = init_date + timedelta(days=lead_day - 1)

    folder_date = init_date.strftime("%Y%m%d")
    start_str = init_date.strftime("%d_%m_%Y")
    target_str = target_date.strftime("%d_%m_%Y")

    filename = f"{start_str}-{target_str}.nc"

    return os.path.join(
        base_path,
        run_name,
        "FC",
        "ML",
        folder_date,
        cycle_folder,
        filename,
    )


# ============================================================
# BUILD FILE LISTS: aek_g4e2_name_nc
# ============================================================
aek_g4e2_name_nc_all = {}

for run_name in run_names:
    files_mat = np.empty((n_members, ndays), dtype=object)
    rows = []

    for day_idx in range(ndays):
        init_date = init_start + timedelta(days=day_idx)

        for member_idx, lead_day in enumerate(forecast_leads):
            path = make_nc_path(run_name, init_date, lead_day)
            files_mat[member_idx, day_idx] = path

            rows.append({
                "run_name": run_name,
                "day_index": day_idx + 1,
                "init_date": init_date.strftime("%Y-%m-%d"),
                "member": member_idx,
                "lead_day": lead_day,
                "path": path,
                "exists": os.path.exists(path),
            })

    aek_g4e2_name_nc_all[run_name] = files_mat

    df_paths = pd.DataFrame(rows)
    path_csv = os.path.join(output_base, f"{run_name}_Chile_aek_g4e2_name_nc.csv")
    df_paths.to_csv(path_csv, index=False)

    print(f"\nSaved file list for {run_name}: {path_csv}")
    print(df_paths.head(9))

    missing = df_paths.loc[~df_paths["exists"], "path"].tolist()
    print(f"Missing files for {run_name}: {len(missing)}")

    if missing:
        print("First missing files:")
        for p in missing[:10]:
            print(p)

# ============================================================
# MAIN PROCESSING LOOP
# ============================================================
for run_name in run_names:

    output_prefix = os.path.join(output_base, f"{run_name}_Chile_DJF_2026")
    nc_path = f"{output_prefix}.nc"

    if os.path.exists(nc_path):
        print(f"\nNetCDF for {run_name} already exists. Moving on...")
        continue

    print(f"\nProcessing run: {run_name}")

    aek_g4e2_name_nc = aek_g4e2_name_nc_all[run_name]

    # Equivalent to R:
    # aaek_g4o2_2 = array(NA, c(8*ndays, 6, length(x1x), 3, 4))
    aaek_g4o2_2 = np.full(
        (
            n_times_per_day * ndays,
            6,
            n_stations,
            n_members,
            len(all_parameters_cifs),
        ),
        np.nan,
        dtype=np.float32,
    )

    timeeees = np.full(
        (n_times_per_day * ndays, n_members),
        np.nan,
        dtype=np.float64,
    )

    ntime = 0

    for day_idx in range(ndays):

        print(f"\nRun {run_name} | init day {day_idx + 1}/{ndays}")

        for member_idx, lead_day in enumerate(forecast_leads):

            nc_file = aek_g4e2_name_nc[member_idx, day_idx]

            print(f"  member={member_idx} | lead={lead_day} | file={nc_file}")

            if not os.path.exists(nc_file):
                print("  Missing file, skipping.")
                continue

            ds = xr.open_dataset(nc_file)

            lon_name = detect_coord_name(ds, ["longitude", "lon", "LONGITUDE", "LON"])
            lat_name = detect_coord_name(ds, ["latitude", "lat", "LATITUDE", "LAT"])
            lev_name = detect_coord_name(ds, ["level", "lev", "LEVEL"])
            time_name = detect_coord_name(ds, ["time", "TIME"])

            time_vals = ds[time_name].values[:n_times_per_day]

            try:
                timeeees[
                    ntime:ntime + n_times_per_day,
                    member_idx,
                ] = np.asarray(time_vals, dtype=float)
            except Exception:
                timeeees[
                    ntime:ntime + n_times_per_day,
                    member_idx,
                ] = np.arange(
                    ntime,
                    ntime + n_times_per_day,
                    dtype=float,
                )

            for var_idx, varname in enumerate(all_parameters_cifs):

                if varname not in ds.data_vars:
                    print(f"  Warning: variable {varname} not found in {nc_file}")
                    continue

                da = ds[varname]

                # longitude: [0, 360] -> [-180, 180]
                da = normalize_longitude_da(da, lon_name)

                # latitude ascending for interpolation
                if da[lat_name].values[0] > da[lat_name].values[-1]:
                    da = da.sortby(lat_name)

                # Expected order:
                # time, level, latitude, longitude
                da = da.transpose(time_name, lev_name, lat_name, lon_name)

                # R equivalent:
                # ncvar_get(...)[,,,1:8]
                # z = go3_nc[..., 25, ii]
                da = da.isel({
                    time_name: slice(0, n_times_per_day),
                    lev_name: level_index,
                })

                # Interpolate to Chile station locations
                interp_da = da.interp(
                    {
                        lon_name: xr.DataArray(y1x, dims="station"),
                        lat_name: xr.DataArray(x1x, dims="station"),
                    },
                    method="linear",
                )

                out = interp_da.values  # shape: time, station

                aaek_g4o2_2[
                    ntime:ntime + n_times_per_day,
                    var_slot_index,
                    :,
                    member_idx,
                    var_idx,
                ] = out

            ds.close()

        # advance once per initialization day
        ntime += n_times_per_day

    # Match R:
    # aaek_g4o2_2[aaek_g4o2_2 < 0] = 0
    aaek_g4o2_2[aaek_g4o2_2 < 0] = 0

    # ========================================================
    # SAVE NETCDF ONLY
    # ========================================================
    ds_out = xr.Dataset(
        data_vars={
            "aaek_g4o2_2": (
                ("time_index", "var_slot", "station", "member", "variable"),
                aaek_g4o2_2,
            ),
            "aek_g4e2_name_nc": (
                ("member", "init_day"),
                np.asarray(aek_g4e2_name_nc, dtype="S500"),
            ),
        },
        coords={
            "time_index": np.arange(n_times_per_day * ndays),
            "var_slot": np.arange(6),
            "station": np.arange(n_stations),
            "member": np.arange(n_members),
            "variable": np.asarray(all_parameters_cifs, dtype="S10"),
            "init_day": np.arange(1, ndays + 1),
            "station_lat": ("station", x1x),
            "station_lon": ("station", y1x),
            "forecast_lead": ("member", forecast_leads),
            "time_value": (("time_index", "member"), timeeees),
        },
        attrs={
            "description": "Chile DJF 2026 station-interpolated model data",
            "run_name": run_name,
            "level_index_python": level_index,
            "level_index_R": level_index + 1,
            "var_slot_python": var_slot_index,
            "var_slot_R": var_slot_index + 1,
            "init_start": init_start.strftime("%Y-%m-%d"),
        },
    )

    ds_out.to_netcdf(nc_path, engine="netcdf4")
    print(f"Saved NetCDF: {nc_path}")

print("\nAll Chile processing complete.")
# %%
