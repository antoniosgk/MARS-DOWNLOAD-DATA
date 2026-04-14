#%%
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ============================================================
# SETTINGS
# ============================================================
varname='go3'
nc_path = f"/home/agkiokas/MARS/plots/{varname}.nc"
output_dir = "/home/agkiokas/MARS/plots"
os.makedirs(output_dir, exist_ok=True)

var_index = 2              # only this one is populated in your workflow
time_idx = 0               # snapshot time step
model_names = ["j06q", "j06r"]
save_figs = False          # set False if you only want to display figures

# Optional map extent: [lon_min, lon_max, lat_min, lat_max]
# Set to None for automatic extent from station coordinates
user_extent = None
dpi = 300
marker_size = 45

# ============================================================
# LOAD
# ============================================================
ds = xr.open_dataset(nc_path)

lats = ds["station_lat"].values
lons = ds["station_lon"].values
data = ds["aaek_g4o2_2"].isel(var_index=var_index)   # dims: time_index, station, model

# ============================================================
# HELPERS
# ============================================================
def get_extent(lons, lats, pad_frac=0.05, user_extent=None):
    if user_extent is not None:
        return user_extent

    lon_min = np.nanmin(lons)
    lon_max = np.nanmax(lons)
    lat_min = np.nanmin(lats)
    lat_max = np.nanmax(lats)

    lon_pad = max((lon_max - lon_min) * pad_frac, 1.0)
    lat_pad = max((lat_max - lat_min) * pad_frac, 1.0)

    return [lon_min - lon_pad, lon_max + lon_pad, lat_min - lat_pad, lat_max + lat_pad]


def robust_limits(arr, q_low=2, q_high=98, symmetric=False):
    vals = np.asarray(arr).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None, None

    if symmetric:
        vmax = np.nanpercentile(np.abs(vals), q_high)
        return -vmax, vmax

    return np.nanpercentile(vals, q_low), np.nanpercentile(vals, q_high)


def setup_map_ax(ax, extent):
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, alpha=0.2)
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False


def scatter_map(ax, lons, lats, vals, title, cmap, vmin=None, vmax=None, extent=None):
    setup_map_ax(ax, extent)
    sc = ax.scatter(
        lons,
        lats,
        c=vals,
        s=marker_size,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree()
    )
    ax.set_title(title, fontsize=12)
    return sc


def finish_figure(fig, filename=None):
    if save_figs and filename is not None:
        fig.savefig(os.path.join(output_dir, filename), dpi=dpi, bbox_inches="tight")
    plt.show()

# ============================================================
# PREPARE DATA
# ============================================================
extent = get_extent(lons, lats, user_extent=user_extent)

snapshot0 = data.isel(time_index=time_idx, model=0).values
snapshot1 = data.isel(time_index=time_idx, model=1).values

mean0 = data.isel(model=0).mean(dim="time_index").values
mean1 = data.isel(model=1).mean(dim="time_index").values

std0 = data.isel(model=0).std(dim="time_index").values
std1 = data.isel(model=1).std(dim="time_index").values

diff_mean = (data.isel(model=1) - data.isel(model=0)).mean(dim="time_index").values

snap_vmin, snap_vmax = robust_limits(np.concatenate([snapshot0, snapshot1]))
mean_vmin, mean_vmax = robust_limits(np.concatenate([mean0, mean1]))
std_vmin, std_vmax = robust_limits(np.concatenate([std0, std1]))
diff_vmin, diff_vmax = robust_limits(diff_mean, symmetric=True)

# ============================================================
# 1) SNAPSHOT PANEL
# ============================================================
fig, axes = plt.subplots(
    1, 2,
    figsize=(14, 5.5),
    subplot_kw={"projection": ccrs.PlateCarree()},
    constrained_layout=True
)

sc0 = scatter_map(
    axes[0], lons, lats, snapshot0,
    f"Snapshot | {model_names[0]} | time_index={time_idx}",
    "viridis", snap_vmin, snap_vmax, extent
)
sc1 = scatter_map(
    axes[1], lons, lats, snapshot1,
    f"Snapshot | {model_names[1]} | time_index={time_idx}",
    "viridis", snap_vmin, snap_vmax, extent
)

cbar = fig.colorbar(
    sc1,
    ax=axes.ravel().tolist(),
    orientation="horizontal",
    shrink=0.85,
    pad=0.08
)
cbar.set_label("Snapshot")

finish_figure(fig, "snapshot_panel_fixed.png")

# ============================================================
# 2) CLIMATOLOGICAL MEAN PANEL
# ============================================================
fig, axes = plt.subplots(
    1, 2,
    figsize=(14, 5.5),
    subplot_kw={"projection": ccrs.PlateCarree()},
    constrained_layout=True
)

sc0 = scatter_map(
    axes[0], lons, lats, mean0,
    f"Climatological mean | {model_names[0]}",
    "plasma", mean_vmin, mean_vmax, extent
)
sc1 = scatter_map(
    axes[1], lons, lats, mean1,
    f"Climatological mean | {model_names[1]}",
    "plasma", mean_vmin, mean_vmax, extent
)

cbar = fig.colorbar(
    sc1,
    ax=axes.ravel().tolist(),
    orientation="horizontal",
    shrink=0.85,
    pad=0.08
)
cbar.set_label("Mean")

finish_figure(fig, "climatological_mean_panel_fixed.png")

# ============================================================
# 3) CLIMATOLOGICAL STD PANEL
# ============================================================
fig, axes = plt.subplots(
    1, 2,
    figsize=(14, 5.5),
    subplot_kw={"projection": ccrs.PlateCarree()},
    constrained_layout=True
)

sc0 = scatter_map(
    axes[0], lons, lats, std0,
    f"Climatological std | {model_names[0]}",
    "inferno", std_vmin, std_vmax, extent
)
sc1 = scatter_map(
    axes[1], lons, lats, std1,
    f"Climatological std | {model_names[1]}",
    "inferno", std_vmin, std_vmax, extent
)

cbar = fig.colorbar(
    sc1,
    ax=axes.ravel().tolist(),
    orientation="horizontal",
    shrink=0.85,
    pad=0.08
)
cbar.set_label("Standard deviation")

finish_figure(fig, "climatological_std_panel_fixed.png")

# ============================================================
# 4) DIFFERENCE MAP
# ============================================================
fig, ax = plt.subplots(
    1, 1,
    figsize=(7.5, 5.8),
    subplot_kw={"projection": ccrs.PlateCarree()},
    constrained_layout=True
)

sc = scatter_map(
    ax, lons, lats, diff_mean,
    f"Climatological difference | {model_names[1]} - {model_names[0]}",
    "coolwarm", diff_vmin, diff_vmax, extent
)

cbar = fig.colorbar(
    sc,
    ax=ax,
    orientation="horizontal",
    shrink=0.85,
    pad=0.08
)
cbar.set_label("Difference")

finish_figure(fig, "climatological_difference_fixed.png")

# ============================================================
# 5) SUMMARY PANEL
# ============================================================
fig, axes = plt.subplots(
    2, 3,
    figsize=(16, 9),
    subplot_kw={"projection": ccrs.PlateCarree()},
    constrained_layout=True
)

sc_a = scatter_map(
    axes[0, 0], lons, lats, mean0,
    f"Mean | {model_names[0]}",
    "plasma", mean_vmin, mean_vmax, extent
)
sc_b = scatter_map(
    axes[0, 1], lons, lats, mean1,
    f"Mean | {model_names[1]}",
    "plasma", mean_vmin, mean_vmax, extent
)
sc_c = scatter_map(
    axes[0, 2], lons, lats, diff_mean,
    f"Mean difference | {model_names[1]} - {model_names[0]}",
    "coolwarm", diff_vmin, diff_vmax, extent
)
sc_d = scatter_map(
    axes[1, 0], lons, lats, std0,
    f"Std | {model_names[0]}",
    "inferno", std_vmin, std_vmax, extent
)
sc_e = scatter_map(
    axes[1, 1], lons, lats, std1,
    f"Std | {model_names[1]}",
    "inferno", std_vmin, std_vmax, extent
)
sc_f = scatter_map(
    axes[1, 2], lons, lats, snapshot0,
    f"Snapshot | {model_names[0]} | time_index={time_idx}",
    "viridis", snap_vmin, snap_vmax, extent
)

# one colorbar for each type shown
cb1 = fig.colorbar(sc_b, ax=[axes[0, 0], axes[0, 1]], orientation="horizontal", shrink=0.85, pad=0.08)
cb1.set_label("Mean")

cb2 = fig.colorbar(sc_c, ax=axes[0, 2], orientation="horizontal", shrink=0.85, pad=0.08)
cb2.set_label("Difference")

cb3 = fig.colorbar(sc_e, ax=[axes[1, 0], axes[1, 1]], orientation="horizontal", shrink=0.85, pad=0.08)
cb3.set_label("Standard deviation")

cb4 = fig.colorbar(sc_f, ax=axes[1, 2], orientation="horizontal", shrink=0.85, pad=0.08)
cb4.set_label("Snapshot")

finish_figure(fig, "summary_panel_fixed.png")

ds.close()
# %%
