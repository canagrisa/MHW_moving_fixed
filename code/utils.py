import os
import xarray as xr
import warnings
import numpy as np
from scipy import stats


def flatten(nested_list):
    return [item for sublist in nested_list for item in sublist]

def dirtodict(dirPath):
    #From a given folder, create a dictionary with a given folder & file tree structure
    d = {}
    for i in [os.path.join(dirPath, i) for i in os.listdir(dirPath)
              if os.path.isdir(os.path.join(dirPath, i))]:
        d[os.path.basename(i)] = dirtodict(i)
    d['.files'] = [os.path.join(dirPath, i) for i in os.listdir(dirPath)
                   if os.path.isfile(os.path.join(dirPath, i))]
    return d

def area(lat,lon):
    """
    Compute area of a rectilinear grid.
    """
    earth_radius = 6371e3
    omega = 7.2921159e-5

    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    f=2*omega*np.sin(lat_r)
    grad_lon=lon_r.copy()
    grad_lon.data=np.gradient(lon_r)

    dx=grad_lon*earth_radius*np.cos(lat_r)
    dy=np.gradient(lat_r)*earth_radius
    
    ds_area = xr.DataArray((dx*dy).T,
                      coords = {'lat' : lat,
                                'lon' : lon},
                      dims = ['lat','lon'])

    return ds_area

def weighted_mean(da):
    
    ds_area = area(da.lat, da.lon)
    
    ds_area = ds_area.where(da.isel(time=0).notnull(), drop=True)
    area_sum = np.nansum(np.array(ds_area))
    weighted_area = ds_area/area_sum
    result = (da*weighted_area).sum(('lat', 'lon'))
    
    return result

def get_distr(da, zeros=True, above=None):
    
    darr = np.array(da)
    values = darr[darr<1000]
        
    if zeros==True:
        values = values[values>0]

    if above!=None:
        values = values[values>above]
    
    return values

def get_1d_lat_lon(ds):
    if 'i' in ds.dims and 'j' in ds.dims:
        if 'CMCC' in  ds.attrs['source_id']:
            lat_idx='i'
            lon_idx='j'
            lat = ds.latitude.mean(dim='j')
            lon = ds.longitude.mean(dim='i')
        else:
            lat_idx='j'
            lon_idx='i'
            lat = ds.latitude.mean(dim='i')
            lon = ds.longitude.mean(dim='j')
    elif 'x' in ds.dims and 'y' in ds.dims:
        lat_idx='y'
        lon_idx='x'
        lat = ds.latitude.mean(dim='x')
        lon = ds.longitude.mean(dim='y')
    elif 'nlat' in ds.dims and 'nlon' in ds.dims:
        lat_idx='nlat'
        lon_idx='nlon'
        lat = ds.latitude.mean(dim='nlon')
        lon = ds.longitude.mean(dim='nlat')
    else:
        raise ValueError("Unsupported dataset format")
    
    lat_1d = np.unique(lat)
    lon_1d = np.unique(lon)
    
    return lat_1d, lon_1d, lat_idx, lon_idx

def compute_trend_pvalue(data):
    time_index = np.arange(len(data))
    slope, intercept, r_value, p_value, std_err = stats.linregress(time_index, data)
    return np.array([slope, p_value])

def open_and_process_dataset(files, time_slice=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ds = xr.open_mfdataset(files, combine="by_coords")
        if time_slice:
            ds = ds.sel(time=ds.time.dt.year.isin(list(range(*time_slice))))
    return ds


def trend(arr):
    #compute trend of a 1D array
    time_index = np.arange(len(arr))
    slope, intercept, r_value, p_value, std_err = stats.linregress(time_index, arr)
    return slope, p_value



def retrieve_one_point(ds, lon, lat,
                       years=[],
              lon_name = 'lon',
              lat_name = 'lat'):
    
    """
    :lon: number specifying longitude of gridpoint
    :lat: number specifying latitude of gridpoint
    :years: list specifying min and max of year range
    
    """
    
    lons = np.asarray(ds[lon_name])
    lats = np.asarray(ds[lat_name])
    
    ds = ds.sel(lon=lon, lat=lat, method = 'nearest')
    
    if years:
        if len(years)>1:
            year_range = list(range(years[0], years[1]+1))
            ds = ds.sel(time=ds.time.dt.year.isin(year_range))
        elif len(years)==1:
            ds = ds.sel(time=ds.time.dt.year.isin(years))
    
    leap_year_feb29 = (ds.time.dt.month == 2) & (ds.time.dt.day == 29)
    ds = ds.where(~leap_year_feb29, drop=True)

    return ds