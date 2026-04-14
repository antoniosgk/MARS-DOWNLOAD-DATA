#%%
import os
import glob
import pickle
import numpy as np
import pandas as pd
import pyreadr
import xarray as xr

# ============================================================
# USER SETTINGS
# ============================================================
rdata_path = "/home/agkiokas/CAMS/aek_ff1_EU_REAN_DJF_2025.Rdata"
base_globe_path = "/mnt/store02/agkiokas/data/GLOBE"
output_prefix = "/home/agkiokas/MARS/plots/"

all_parameters_cifs = ["go3", "co"]
modds = ["j06q", "j06r"]
n_vars = len(all_parameters_cifs)
ndays = 59
n_times_per_file = 8

# only first variable, matching your R loop for(kkk in 1:1)
varname = all_parameters_cifs[0]

# ============================================================
# HELPERS
# ============================================================
def detect_coord_name(ds, candidates):
    for c in candidates:
        if c in ds.coords or c in ds.variables or c in ds.dims:
            return c
    raise KeyError(f"Could not find any of {candidates} in dataset. Found dims/vars: {list(ds.dims)} / {list(ds.variables)}")

def normalize_longitude_da(da, lon_name="longitude"):
    lon = da[lon_name]
    lon_new = xr.where(lon > 180, lon - 360, lon)
    da = da.assign_coords({lon_name: lon_new})
    da = da.sortby(lon_name)
    return da

# ============================================================
# 1. LOAD RDATA
# ============================================================
result = pyreadr.read_r(rdata_path)
print("Objects in RData:", list(result.keys()))

if "aekkkkk_ff1" not in result:
    raise KeyError(f"'aekkkkk_ff1' not found in RData. Available objects: {list(result.keys())}")

aekkkkk_ff1 = result["aekkkkk_ff1"]

# R -> Python column mapping
# R [,3] -> Python iloc[:, 2]
# R [,2] -> Python iloc[:, 1]
# R [,11] -> Python iloc[:, 10]
x1x = aekkkkk_ff1.iloc[:, 2].to_numpy(dtype=float)         # station latitude? (as in your original logic)
y1x = aekkkkk_ff1.iloc[:, 1].to_numpy(dtype=float)         # station longitude? (as in your original logic)
stat_level = aekkkkk_ff1.iloc[:, 10].to_numpy(dtype=float)

n_stations = len(x1x)

print(f"Number of stations: {n_stations}")

# ============================================================
# 2. PREALLOCATE OUTPUTS
aaek_g4o2_2 = np.full((n_times_per_file * ndays, 6, n_stations, len(modds)), np.nan, dtype=np.float32)
timeeees = np.full(n_times_per_file * ndays, np.nan, dtype=np.float64)

# ============================================================
# 3. MAIN PROCESSING
# ============================================================
for kl, modd in enumerate(modds):
    pattern = os.path.join(base_globe_path, modd, "FC", "ML", "**", "*.nc")
    files = sorted(glob.glob(pattern, recursive=True))

    print(f"\nModel {modd}: found {len(files)} files")

    iii = 0      # index for timeeees
    ntime = 0    # index for output time axis

    for file_idx, nc_file in enumerate(files, start=1):
        print(f"Opening file {file_idx}/{len(files)}: {nc_file}")

        ds = xr.open_dataset(nc_file)

        lon_name = detect_coord_name(ds, ["longitude", "lon", "LONGITUDE", "LON"])
        lat_name = detect_coord_name(ds, ["latitude", "lat", "LATITUDE", "LAT"])
        lev_name = detect_coord_name(ds, ["level", "lev", "LEVEL"])
        time_name = detect_coord_name(ds, ["time", "TIME"])

        if varname not in ds.data_vars:
            raise KeyError(
                f"Variable '{varname}' not found in {nc_file}. "
                f"Available data vars: {list(ds.data_vars)}"
            )

        da = ds[varname]

        # Normalize longitude to [-180, 180] and sort
        da = normalize_longitude_da(da, lon_name=lon_name)

        # Sort latitude ascending for interpolation
        lat_vals = da[lat_name].values
        if lat_vals[0] > lat_vals[-1]:
            da = da.sortby(lat_name)

        # Reorder explicitly: time, level, lat, lon
        expected_dims = [time_name, lev_name, lat_name, lon_name]
        for d in expected_dims:
            if d not in da.dims:
                raise ValueError(
                    f"{nc_file}: expected dimension '{d}' not found in {da.dims}"
                )
        da = da.transpose(time_name, lev_name, lat_name, lon_name)

        # Keep first 8 times only, matching the R code
        da = da.isel({time_name: slice(0, n_times_per_file)})

        # Save raw time values
        time_vals = ds[time_name].values[:n_times_per_file]
        n_take = min(len(time_vals), len(timeeees) - iii)
        if n_take > 0:
            try:
                timeeees[iii:iii+n_take] = np.asarray(time_vals[:n_take], dtype=float)
            except Exception:
                # fallback for non-numeric time
                timeeees[iii:iii+n_take] = np.arange(iii, iii+n_take, dtype=float)
        iii += n_take

        # Interpolate all stations at once
        interp_da = da.interp(
            {
                lon_name: xr.DataArray(y1x, dims="station"),
                lat_name: xr.DataArray(x1x, dims="station"),
            },
            method="linear"
        )
        # interp_da dims: (time, level, station)

        vals = interp_da.values
        levels = interp_da[lev_name].values

        # Match each station to its corresponding model level
        # mask shape: (n_levels, n_stations)
        mask = (levels[:, None] == stat_level[None, :])

        # Broadcast over time: (time, level, station)
        vals_masked = np.where(mask[None, :, :], vals, np.nan)

        # Collapse level axis -> one value per time/station
        # This assumes each station level matches at most one model level
        out = np.nanmax(vals_masked, axis=1)   # shape: (time, station)

        # Write into var_index=2, matching Python equivalent of R's index 3
        end_time = min(ntime + out.shape[0], aaek_g4o2_2.shape[0])
        write_n = end_time - ntime
        if write_n > 0:
            aaek_g4o2_2[ntime:end_time, 2, :, kl] = out[:write_n, :]

        ntime += out.shape[0]
        ds.close()

    print(f"Completed model {modd}")


# ============================================================
# 4. SAVE PKL
# ============================================================
pkl_path = f"{output_prefix}{varname}.pkl"
with open(pkl_path, "wb") as f:
    pickle.dump(
        {
            "aaek_g4o2_2": aaek_g4o2_2,
            "timeeees": timeeees,
            "x1x": x1x,
            "y1x": y1x,
            "stat_level": stat_level,
            "modds": modds,
            "varname": varname,
        },
        f
    )
print(f"Saved PKL: {pkl_path}")

# ============================================================
# 5. SAVE NETCDF
# ============================================================
coords = {
    "time_index": np.arange(aaek_g4o2_2.shape[0]),
    "var_index": np.arange(aaek_g4o2_2.shape[1]),
    "station": np.arange(aaek_g4o2_2.shape[2]),
    "model": np.arange(aaek_g4o2_2.shape[3]),
    "time_value": ("time_index", timeeees),
    "station_lat": ("station", x1x),
    "station_lon": ("station", y1x),
    "station_level": ("station", stat_level),
}

ds_out = xr.Dataset(
    data_vars={
        "aaek_g4o2_2": (
            ("time_index", "var_index", "station", "model"),
            aaek_g4o2_2
        )
    },
    coords=coords,
    attrs={
        "description": "Processed output converted from R workflow using xarray",
        "source_rdata": rdata_path,
        "variable_processed": varname,
    }
)

ds_out["model_name"] = ("model", np.array(modds, dtype=object))

nc_path = f"{output_prefix}{varname}.nc"
ds_out.to_netcdf(nc_path)
print(f"Saved NetCDF: {nc_path}")

# ============================================================
# 6. SAVE CSV (USEFUL SLICE ONLY: var_index = 2)
# ============================================================
# This is the only populated slice from the current workflow
arr = aaek_g4o2_2[:, 2, :, :]   # shape: (time, station, model)

nt, ns, nm = arr.shape
time_idx, station_idx, model_idx = np.indices((nt, ns, nm))

df_csv = pd.DataFrame({
    "time_index": time_idx.ravel(),
    "station": station_idx.ravel(),
    "model": model_idx.ravel(),
    "value": arr.ravel(),
})

df_csv["time_value"] = timeeees[df_csv["time_index"].values]
df_csv["station_lat"] = x1x[df_csv["station"].values]
df_csv["station_lon"] = y1x[df_csv["station"].values]
df_csv["station_level"] = stat_level[df_csv["station"].values]
df_csv["model_name"] = np.array(modds)[df_csv["model"].values]
df_csv["var_index"] = 2
df_csv["variable_name"] = varname

# Optional: keep only non-missing values to reduce file size
df_csv = df_csv.dropna(subset=["value"]).reset_index(drop=True)

csv_path = f"{output_prefix}_{varname}_only.csv"
df_csv.to_csv(csv_path, index=False)
print(f"Saved CSV: {csv_path}")

print("\nAll outputs saved successfully.")
# %%
