#%%
import os
import pickle
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

# 3 forecast files per initialization day
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
# LOAD STATION COORDINATES
# ============================================================
rdata = pyreadr.read_r(coords_path)

print("Objects in RData:", list(rdata.keys()))

# If the object is called all_cords, use it.
# Otherwise, use the first object in the RData file.
if "all_cords" in rdata:
    coords_obj_name = "all_cords"
else:
    coords_obj_name = list(rdata.keys())[0]

print("Using coordinates object:", coords_obj_name)

all_cords = np.asarray(rdata[coords_obj_name])

# R:
# x1x = all_cords[2,1,]
# y1x = all_cords[1,1,]
x1x = all_cords[1, 0, :].astype(float)
y1x = all_cords[0, 0, :].astype(float)



n_stations = len(x1x)
print(f"Number of stations: {n_stations}")

# ============================================================
# HELPER FUNCTIONS
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
    """
    Example:
    run_name = "0001"
    init_date = datetime(2025, 12, 1)
    lead_day = 3

    returns:
    /mnt/store02/agkiokas/data/GLOBE/0001/FC/ML/20251201/00_00_00/01_12_2025-03_12_2025.nc
    """
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
    path_csv = os.path.join(output_base, f"{run_name}_aek_g4e2_name_nc.csv")
    df_paths.to_csv(path_csv, index=False)

    print(f"\nSaved file list for {run_name}: {path_csv}")
    print(df_paths.head(9))

    missing = df_paths.loc[~df_paths["exists"], "path"].tolist()
    print(f"Missing files for {run_name}: {len(missing)}")

    if missing:
        print("First missing files:")
        for p in missing[:10]:
            print(p)

paths_pkl = os.path.join(output_base, "aek_g4e2_name_nc_all.pkl")
with open(paths_pkl, "wb") as f:
    pickle.dump(aek_g4e2_name_nc_all, f)

print(f"\nSaved all file lists as pickle: {paths_pkl}")

# ============================================================
# MAIN PROCESSING LOOP
# ============================================================
for run_name in run_names:

    output_prefix = os.path.join(output_base, f"{run_name}_Chile_DJF_2026")

    pkl_path = f"{output_prefix}.pkl"
    nc_path = f"{output_prefix}.nc"
    csv_path = f"{output_prefix}_long.csv"

    if os.path.exists(pkl_path) or os.path.exists(nc_path):
        print(f"\nOutput for {run_name} already exists. Moving on...")
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

    # Time values per time index and forecast member
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
                print(f"  Missing file, skipping.")
                continue

            ds = xr.open_dataset(nc_file)

            lon_name = detect_coord_name(ds, ["longitude", "lon", "LONGITUDE", "LON"])
            lat_name = detect_coord_name(ds, ["latitude", "lat", "LATITUDE", "LAT"])
            lev_name = detect_coord_name(ds, ["level", "lev", "LEVEL"])
            time_name = detect_coord_name(ds, ["time", "TIME"])

            # Save time values
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

                # Normalize/sort longitude
                da = normalize_longitude_da(da, lon_name)

                # Sort latitude ascending for xarray interpolation
                if da[lat_name].values[0] > da[lat_name].values[-1]:
                    da = da.sortby(lat_name)

                # Force expected dimension order
                da = da.transpose(time_name, lev_name, lat_name, lon_name)

                # R:
                # go3_nc = ncvar_get(...)[,,,1:8]
                # z = go3_nc[..., level=25, time=ii]
                da = da.isel({
                    time_name: slice(0, n_times_per_day),
                    lev_name: level_index,
                })

                # Interpolate all stations at once
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

        # Advance once per initialization day, exactly like R:
        # ntime = ntime + 8 after the three files
        ntime += n_times_per_day

    # Match R:
    # aaek_g4o2_2[aaek_g4o2_2 < 0] = 0
    aaek_g4o2_2[aaek_g4o2_2 < 0] = 0
    '''
    # ========================================================
    # SAVE PKL
    # ========================================================
    with open(pkl_path, "wb") as f:
        pickle.dump(
            {
                "all_parameters_cifs": all_parameters_cifs,
                "aaek_g4o2_2": aaek_g4o2_2,
                "timeeees": timeeees,
                "aek_g4e2_name_nc": aek_g4e2_name_nc,
                "forecast_leads": forecast_leads,
                "x1x": x1x,
                "y1x": y1x,
                "run_name": run_name,
            },
            f,
        )
'''
    print(f"\nSaved PKL: {pkl_path}")

    # ========================================================
    # SAVE NETCDF
    # ========================================================
    ds_out = xr.Dataset(
        data_vars={
            "aaek_g4o2_2": (
                ("time_index", "var_slot", "station", "member", "variable"),
                aaek_g4o2_2,
            )
        },
        coords={
            "time_index": np.arange(n_times_per_day * ndays),
            "var_slot": np.arange(6),
            "station": np.arange(n_stations),
            "member": np.arange(n_members),
            "variable": all_parameters_cifs,
            "station_lat": ("station", x1x),
            "station_lon": ("station", y1x),
            "forecast_lead": ("member", forecast_leads),
            "time_value": (("time_index", "member"), timeeees),
            "init_day": ("init_day", np.arange(1, ndays + 1)),
        },
        attrs={
            "description": " DJF 2026 station-interpolated model data",
            "run_name": run_name,
            "level_index_python": level_index,
            "level_index_R": level_index + 1,
            "var_slot_python": var_slot_index,
            "var_slot_R": var_slot_index + 1,
            "init_start": init_start.strftime("%Y-%m-%d"),
        },
    )

    ds_out["aek_g4e2_name_nc"] = (
        ("member", "init_day"),
        np.asarray(aek_g4e2_name_nc, dtype=str),
    )

    ds_out.to_netcdf(nc_path)
    print(f"Saved NetCDF: {nc_path}")
    '''
    # ========================================================
    # SAVE CSV LONG FORMAT
    # ========================================================
    arr = aaek_g4o2_2[:, var_slot_index, :, :, :]

    nt, ns, nm, nv = arr.shape
    time_idx_arr, station_idx_arr, member_idx_arr, variable_idx_arr = np.indices(
        (nt, ns, nm, nv)
    )

    df = pd.DataFrame({
        "time_index": time_idx_arr.ravel(),
        "station": station_idx_arr.ravel(),
        "member": member_idx_arr.ravel(),
        "variable_index": variable_idx_arr.ravel(),
        "value": arr.ravel(),
    })

    df["variable"] = np.array(all_parameters_cifs)[df["variable_index"].values]
    df["forecast_lead"] = np.array(forecast_leads)[df["member"].values]
    df["station_lat"] = x1x[df["station"].values]
    df["station_lon"] = y1x[df["station"].values]
    df["time_value"] = timeeees[
        df["time_index"].values,
        df["member"].values,
    ]

    df = df.dropna(subset=["value"]).reset_index(drop=True)
    df.to_csv(csv_path, index=False)

    print(f"Saved CSV: {csv_path}")
    print(f"Finished run: {run_name}")
'''
print("\nAll processing complete.")
# %%
