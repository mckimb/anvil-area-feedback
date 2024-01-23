import xarray as xr
import numpy as np
from scipy.signal import argrelextrema, argrelmax
import scipy as sp
import cloud_constants as cc
from global_land_mask import globe
from scipy.optimize import fsolve

chunk_time = 1
chunk_lat = 1
chunk_lon = 1

def load_cf_cfmip(model,experiment):
    '''
    GOAL
        Returns the CMIP dataset for cloud fraction profile

    INPUTS
        model: choose between 'BCC-CSM2-MR','CESM2','IPSL-CM6A-LR','MRI-ESM2-0','GISS-E2-1-G'
        experiment: choose between 'amip-m4K','amip','amip-p4K
    '''
    path_to_file = '/data/bmckim/cfmip/'+experiment+'/cl_Amon_'+model+'_'+experiment+'*.nc'
    cf = xr.open_mfdataset(path_to_file,chunks={'time': chunk_time,'latitude': chunk_lat,'longitude':chunk_lon}).cl

    return cf

def load_ts_cfmip(model,experiment):
    '''
    GOAL
        Returns the CMIP dataset for surface temperature

    INPUTS
        model: choose between 'BCC-CSM2-MR','CESM2','IPSL-CM6A-LR','MRI-ESM2-0','GISS-E2-1-G'
        experiment: choose between 'amip-m4K','amip','amip-p4K
    '''
    path_to_file = '/data/bmckim/cfmip/'+experiment+'/ts_Amon_'+model+'_'+experiment+'*.nc'
    ts = xr.open_mfdataset(path_to_file,chunks={'time': chunk_time,'latitude': chunk_lat,'longitude':chunk_lon}).ts

    return ts

def process_cf_cfmip(cf):
    '''
    Returns the cloud fraction data as an array, after applying a rolling average,
    looking above 8km, and ignoring clear-sky grid points; returns the height, time, lat, and lon coordinates
    '''
    # 8 km is approx 360 hPa or 0.35 in sigma coordinates
    # applies rolling mean, ignores places below 8km (sigma <0.35), and anyplace with f=0
    foo1 = cf.transpose('lev','lat','lon', 'time') # dimensions are reorded to lev, lat, lon, and time
    # foo2 = foo1.rolling(lev=8, center=True).mean() # 480m rolling average
    foo2 = foo1
    foo3 = foo2.where(foo2.lev<0.35) # look above 8km
    foo3 = foo3.where(foo3>0) # ignore 0s
    foo4 = foo3.values
    heights = foo3.lev.values
    times = foo3.time.values
    lats = foo3.lat.values
    lons = foo3.lon.values
    return foo4, heights, times, lats, lons


def load_highcf():
    '''
    Returns the CALLID data product for clouds with an optical depth between 0.3 and 5
    '''
    # Cloud fraction for non-opaque ice clouds with an optical depth greater than 1
    foo = xr.open_mfdataset('/data/mstlu/CFice_1tau_trans_CAL_LID_L3_Cloud_Occurrence-Standard-V1-00.200606-201612_trop.nc',chunks={'time': chunk_time,'latitude': chunk_lat}).unknown
    # Cloud fraction non ice clouds with an optical depth between 0.3 and 1
    bar = xr.open_mfdataset('/data/mstlu/CFice_03tau1_CAL_LID_L3_Cloud_Occurrence-Standard-V1-00.200606-201612_trop.nc',chunks={'time': chunk_time,'latitude': chunk_lat}).unknown
    # data used by marion to ID high clouds is the some of these two profiles
    cf_high = foo + bar
    return cf_high



def load_lowcf():
    '''
    Returns the CALLID (CALIPSO‐Cloud‐Occurrence) product for cloud fraction of all clouds (ice and water)
    '''
    cf_low = xr.open_mfdataset('/data/mstlu/CF_CAL_LID_L3_Cloud_Occurrence-Standard-V1-00.200606-201612_trop.nc',chunks={'time': chunk_time,'latitude': chunk_lat}).unknown
    return cf_low

def load_lowmidcf():
    '''
    Returns the CALLID (CALIPSO‐Cloud‐Occurrence) product for cloud fraction of all clouds (ice and water)
    '''
    cf_lowmid = xr.open_mfdataset('/data/mstlu/CF_CAL_LID_L3_Cloud_Occurrence-Standard-V1-00.200606-201612_trop.nc',chunks={'time': chunk_time,'latitude': chunk_lat}).unknown
    return cf_lowmid


def load_ts(dataset,ascending=False):
    '''
    Returns the hadcrut4 dataset formatted to the tropics and the same dimensions as CALLID data
    '''
    fh = load_highcf()
    if dataset=='hadcrut5':
        ts = xr.open_mfdataset('/data/mstlu/hadCRUT5_absolute_185001-202208.nc',chunks={'time': chunk_time,'latitude': chunk_lat}).unknown
    if dataset=='hadcrut4':
        ts = xr.open_mfdataset('/data/mstlu/HadCRUT4_absolute.nc',chunks={'time': chunk_time,'latitude': chunk_lat}).temperature_anomaly
    ts = ts.rename({'latitude':'lat','longitude':'lon'})
    ts = ts.interp(lat=fh.lat.values).interp(lon=fh.lon.values).interp(time=fh.time.values)

    # extrapolate the end points (+/- lon=187.75) by taking a weighted average of the closest valid gridpoints
    ts.loc[dict(lon= 178.75)] = 2/3*ts.loc[dict(lon= 176.25)]+1/3*ts.loc[dict(lon=-176.25)]
    ts.loc[dict(lon=-178.75)] = 2/3*ts.loc[dict(lon=-176.25)]+1/3*ts.loc[dict(lon= 176.25)]

    # look at the convective SSTs (defined by where omega <= 40 hPa/day)
    if ascending:
        omega = load_omega()
        ts = ts.where(omega <= 40)
        lon_grid, lat_grid = np.meshgrid(ts.lon,ts.lat)
        globe_ocean_mask = np.broadcast_to(globe.is_ocean(lat_grid, lon_grid), (127,31,144))
        ts = ts.where(globe_ocean_mask)

    return ts

def load_omega():
    '''
    Returns the ERA5 500 hPa vertical velocity formatted to the tropics and the same dimensions as CALLID data
    '''
    fh = load_highcf()
    omega = xr.open_dataset('/data/bmckim/era5_omega.nc').w * 86400 / 100 # convert from Pa/s to hPa/day
    omega = omega.rename({'latitude':'lat','longitude':'lon'})
    omega = omega.assign_coords(lon=(((omega.lon + 180) % 360) - 180)).sortby('lon')
    omega = omega.interp(lat=fh.lat.values).interp(lon=fh.lon.values).interp(time=fh.time.values)
    return omega

def load_rh():
    '''
    Returns the ERA5 1000 hPa relative humidity formatted to the tropics and the same dimensions as CALLID data
    '''
    fh = load_highcf()
    rh = xr.open_dataset('/data/bmckim/era5_rh1000.nc').r
    rh = rh.rename({'latitude':'lat','longitude':'lon'})
    rh = rh.assign_coords(lon=(((rh.lon + 180) % 360) - 180)).sortby('lon')
    rh = rh.interp(lat=fh.lat.values).interp(lon=fh.lon.values).interp(time=fh.time.values)
    return rh

def area_weighted_mean(ds):
    '''
    Returns the tropically averaged dataset, weighted by cosine(latitude)
    '''
    weights = np.cos(np.deg2rad(ds.lat))
    weights.name = 'weights'
    ds_weighted = ds.weighted(weights).mean(('lon','lat'))
    return ds_weighted


def process_highcf(cf_high):
    '''
    Returns the cloud fraction data as an array, after applying a rolling average,
    looking above 8km, and ignoring clear-sky grid points; returns the height, time, lat, and lon coordinates
    '''
    # applies rolling mean, ignores places below 8km, and anyplace with f=0
    foo1 = cf_high.transpose('alt','lat','lon', 'time') # dimensions are reorded to alt, lat, lon, and time
    foo2 = foo1.rolling(alt=8, center=True).mean() # 480m rolling average
    foo3 = foo2.where(foo2.alt>8) # look above 8km
    foo3 = foo3.where(foo3>0) # ignore 0s
    foo4 = foo3.values
    heights = foo3.alt.values
    times = foo3.time.values
    lats = foo3.lat.values
    lons = foo3.lon.values
    return foo4, heights, times, lats, lons

def process_lowcf(cf_low):
    '''
    Returns the cloud fraction data as an array, after applying a rolling average,
    looking below 6km, and ignoring clear-sky grid points; returns the height, time, lat, and lon coordinates
    '''
    # applies rolling mean, ignores places below 8km, and anyplace with f=0
    foo1 = cf_low.transpose('alt','lat','lon', 'time') # dimensions are reorded to alt, lat, lon, and time
    foo2 = foo1.rolling(alt=8, center=True).mean() # 480m rolling average
    foo3 = foo2.where(foo2.alt<=6) # look below 6 km
    foo3 = foo3.where(foo3>0) # ignore 0s
    foo4 = foo3.values
    heights = foo3.alt.values
    times = foo3.time.values
    lats = foo3.lat.values
    lons = foo3.lon.values
    return foo4, heights, times, lats, lons

def process_lowmidcf(cf_lowmid):
    '''
    Returns the cloud fraction data as an array, after applying a rolling average,
    looking below 6km, and ignoring clear-sky grid points; returns the height, time, lat, and lon coordinates
    '''
    # applies rolling mean, ignores places below 8km, and anyplace with f=0
    foo1 = cf_lowmid.transpose('alt','lat','lon', 'time') # dimensions are reorded to alt, lat, lon, and time
    foo2 = foo1.rolling(alt=8, center=True).mean() # 480m rolling average
    foo3 = foo2.where(foo2.alt<=8) # look below 6 km
    foo3 = foo3.where(foo3>0) # ignore 0s
    foo4 = foo3.values
    heights = foo3.alt.values
    times = foo3.time.values
    lats = foo3.lat.values
    lons = foo3.lon.values
    return foo4, heights, times, lats, lons

def load_ceres():
    '''
    Returns ceres data that have been interpolated onto the fh and fl grids
    '''
    ceres = xr.open_mfdataset('/data/bmckim/CERES_EBAF_Ed4.1_Subset_200606-201712.nc',chunks={'time': chunk_time,'latitude': chunk_lat})
    # shift longitudes to be consistent with fh and fl
    ceres = ceres.assign_coords(lon=ceres.lon-180)
    ceres = ceres.roll(lon=180)
    # put on same grid as fh
    cf_high = load_highcf()
    ceres = ceres.interp(lat=cf_high.lat.values).interp(lon=cf_high.lon.values).interp(time=cf_high.time.values)
    ceres = ceres.rename({'toa_lw_all_mon':'r','toa_lw_clr_t_mon':'rcs','toa_sw_all_mon':'s','toa_sw_clr_t_mon':'scs','toa_cre_sw_mon':'cre_sw','toa_cre_lw_mon':'cre_lw','toa_cre_net_mon':'cre'})
    return ceres

def load_era5(save=False):
    # ERA5 data for atmospheric temperature and geopotential
    cf_high = load_highcf()
    era5 = xr.open_mfdataset('/data/bmckim/era5_geopot_temp.nc',chunks={'time': chunk_time,'latitude': chunk_lat})
    era5 = era5.rename({'latitude':'lat','longitude':'lon'})
    era5 = era5.assign_coords(lon=(((era5.lon + 180) % 360) - 180)).sortby('lon')
    # put on same grid as cf
    era5 = era5.interp(lat=cf_high.lat.values).interp(lon=cf_high.lon.values).interp(time=cf_high.time.values)
    # add levels to reduce error in identifying low cloud temperature
    era5 = era5.interp(level=np.linspace(1,1001,101))
    # turn geopotential into actual height (in km)
    z = era5.z/(9.8*1000)
    t = era5.t
    path_to_data = '/data/bmckim/identified_cloud_fraction/'
    if save:
        print('saving data to '+path_to_data+'era5_x_data.nc')
        z.to_netcdf(path_to_data+'era5_z_data.nc')
        t.to_netcdf(path_to_data+'era5_t_data.nc')
    # NOTE CHECK IF THE DATA SHOULD BE ROLLED BEFORE PROCEEDING
    return z, t

def anvil_top(cf_array, heights, times, lats, lons, method, save=False, cf_c=0.03, nan=True):
    """
    Returns the altitude of the anvil cloud top, the cloud fraction
    at this altitude, and the mask
    if method is 'centroid':
                the anvil cloud top is defined as the cloud fraction peak (in the vertical)
                which is the closest to the centroid (centroid of the vertical profile of the CF)
                and where the cloud fraction is greater than cf_c.
    if method is 'max':
                the anvil cloud top is defined as the maximal cloud fraction peak (in the vertical)
                and where the cloud fraction is greater than cf_c.
    if nan = True:
                when no anvil cloud is found, the cloud fraction is set to NaN, otherwise it is set to 0

    """
    if nan:
        fill_value = np.nan
    else:
        fill_value = 0

    f_output = np.empty_like(cf_array[0,:,:,:])
    z_output = np.empty_like(cf_array[0,:,:,:])
    centroid_output = np.empty_like(cf_array[0,:,:,:])
    path_to_data = '/data/bmckim/identified_cloud_fraction/'
    print('save = '+str(save))
    print('method = '+method)
    print('loop is '+str(np.shape(cf_array)[1])+' iterations long')
    print('starting loop')
    for i in range(np.shape(cf_array)[1]):
        if i % 5 == 0:
            print('i = '+str(i))
        for j in range(np.shape(cf_array)[2]):
            for k in range(np.shape(cf_array)[3]):
                if (np.sum(~np.isnan(cf_array[:,i,j,k]))==0):
                    # if the entire column is nan, set outputs to nan
                    f_output[i,j,k] = fill_value
                    z_output[i,j,k] = fill_value


                else:
                    # find the local maxima
                    local_max_indices = argrelextrema(cf_array[:,i,j,k], comparator=np.greater_equal)
                    f_list = []
                    z_list = []
                    for index in local_max_indices[0]:
                        f_list.append(cf_array[index,i,j,k])
                        z_list.append(heights[index])
                    if len(f_list)==0:
                        # if no local maxima found, set outputs to nan
                        f_output[i,j,k] = fill_value
                        z_output[i,j,k] = fill_value
                    # return a nan if none of the maxes are below the threshold (cf_c)
                    # print(f_list)
                    if np.sum(np.array(f_list)>cf_c)==0:
                        f_output[i,j,k] = fill_value
                        z_output[i,j,k] = fill_value

                    else:
                        if method=='centroid':
                            # find the cloud centroid
                            f_profilemasked = cf_array[:,i,j,k][~np.isnan(cf_array[:,i,j,k])]
                            h_profilemasked = heights[~np.isnan(cf_array[:,i,j,k])]
                            z_centroid = np.ma.average(h_profilemasked, weights=f_profilemasked)
                            dist_from_centroid = np.abs(z_centroid - z_list)
                            # choose the maxima closest to the centroid
                            z_id = z_list[np.argmin(dist_from_centroid)]
                            f_id = f_list[np.argmin(dist_from_centroid)]

                        if method=='max':
                            z_id = z_list[np.argmax(f_list)]
                            f_id = f_list[np.argmax(f_list)]
                            z_centroid = fill_value

                        f_output[i,j,k] = f_id
                        z_output[i,j,k] = z_id
                        centroid_output[i,j,k] = z_centroid

    f_mask = ~np.isnan(f_output)

    da_f = xr.DataArray(
    data=f_output,
    dims=['lat', 'lon', 'time'],
    coords=dict(
        lat=(['lat'], lats),
        lon=(['lon'], lons),
        time=times,
    ), attrs=dict(
        description='high cloud fraction, '+str(cf_c)+' threshold, '+method+' method',
        units='unitless',),)
    da_f = da_f.rename('fh')

    da_z = xr.DataArray(
    data=z_output,
    dims=['lat', 'lon', 'time'],
    coords=dict(
        lat=(['lat'], lats),
        lon=(['lon'], lons),
        time=times,
    ), attrs=dict(
        description='high cloud height, '+str(cf_c)+' threshold, '+method+' method',
        units='km',),)
    da_z = da_z.rename('zh')

    da_fmask = xr.DataArray(
    data=f_mask,
    dims=['lat', 'lon', 'time'],
    coords=dict(
        lat=(['lat'], lats),
        lon=(['lon'], lons),
        time=times,
    ), attrs=dict(
        description='high cloud mask, '+str(cf_c)+' threshold, '+method+' method',
        units='Boolean',),)
    da_fmask = da_fmask.rename('fh_mask')

    da_centroid = xr.DataArray(
    data=centroid_output,
    dims=['lat', 'lon', 'time'],
    coords=dict(
        lat=(['lat'], lats),
        lon=(['lon'], lons),
        time=times,
    ), attrs=dict(
        description='high cloud centroid, '+str(cf_c)+' threshold, '+method+' method',
        units='km',),)
    da_centroid = da_centroid.rename('centroid_height')

    if save:
        print('saving to '+path_to_data+'fh_'+method+'_0pt'+str(int(cf_c*100))+'.nc')
        da_f.to_netcdf(path_to_data+'fh_'+method+'_0pt'+str(int(cf_c*100))+'.nc')
        da_z.to_netcdf(path_to_data+'zh_'+method+'_0pt'+str(int(cf_c*100))+'.nc')
        da_fmask.to_netcdf(path_to_data+'fhmask_'+method+'_0pt'+str(int(cf_c*100))+'.nc')


    print('anvil_top program completed')
    print('')
    return da_f, da_z, da_fmask, da_centroid


def shallow_top(cf_array, heights, times, lats, lons, method, save=False, cf_c=0.03):
    """
    Returns the altitude of the shallow cloud top, the cloud fraction
    at this altitude, and the mask
    if method is 'centroid':
                the shallow cloud top is defined as the cloud fraction peak (in the vertical)
                which is the closest to the centroid (centroid of the vertical profile of the CF)
                and where the cloud fraction is greater than cf_c.
    if method is 'max':
                the shallow cloud top is defined as the maximal cloud fraction peak (in the vertical)
                and where the cloud fraction is greater than cf_c.

    """
    nan = False
    if nan:
        fill_value = np.nan
    else:
        fill_value = 0
    f_output = np.empty_like(cf_array[0,:,:,:])
    z_output = np.empty_like(cf_array[0,:,:,:])
    centroid_output = np.empty_like(cf_array[0,:,:,:])
    path_to_data = '/data/bmckim/identified_cloud_fraction/'
    print('save = '+str(save))
    print('method = '+method)
    print('loop is '+str(np.shape(cf_array)[1])+' iterations long')
    print('starting loop')
    for i in range(np.shape(cf_array)[1]):
        if i % 5 == 0:
            print('i = '+str(i))
        for j in range(np.shape(cf_array)[2]):
            for k in range(np.shape(cf_array)[3]):
                if (np.sum(~np.isnan(cf_array[:,i,j,k]))==0):
                    # if the entire column is nan, set outputs to nan
                    f_output[i,j,k] = fill_value
                    z_output[i,j,k] = fill_value

                else:
                    # find the local maxima
                    local_max_indices = argrelextrema(cf_array[:,i,j,k], comparator=np.greater_equal)
                    f_list = []
                    z_list = []
                    for index in local_max_indices[0]:
                        f_list.append(cf_array[index,i,j,k])
                        z_list.append(heights[index])
                    if len(f_list)==0:
                        # if no local maxima found, set outputs to nan
                        f_output[i,j,k] = fill_value
                        z_output[i,j,k] = fill_value
                    # return a nan if none of the maxes are below the threshold (cf_c)
                    # print(f_list)
                    if np.sum(np.array(f_list)>cf_c)==0:
                        f_output[i,j,k] = fill_value
                        z_output[i,j,k] = fill_value

                    else:
                        if method=='centroid':
                            # find the cloud centroid
                            f_profilemasked = cf_array[:,i,j,k][~np.isnan(cf_array[:,i,j,k])]
                            h_profilemasked = heights[~np.isnan(cf_array[:,i,j,k])]
                            z_centroid = np.ma.average(h_profilemasked, weights=f_profilemasked)
                            dist_from_centroid = np.abs(z_centroid - z_list)
                            # choose the maxima closest to the centroid
                            z_id = z_list[np.argmin(dist_from_centroid)]
                            f_id = f_list[np.argmin(dist_from_centroid)]

                        if method=='max':
                            z_id = z_list[np.argmax(f_list)]
                            f_id = f_list[np.argmax(f_list)]
                            z_centroid = np.nan

                        f_output[i,j,k] = f_id
                        z_output[i,j,k] = z_id
                        centroid_output[i,j,k] = z_centroid

    f_mask = ~np.isnan(f_output)

    da_f = xr.DataArray(
    data=f_output,
    dims=['lat', 'lon', 'time'],
    coords=dict(
        lat=(['lat'], lats),
        lon=(['lon'], lons),
        time=times,
    ), attrs=dict(
        description='low cloud fraction, '+str(cf_c)+' threshold, '+method+' method',
        units='unitless',),)
    da_f = da_f.rename('fl')

    da_z = xr.DataArray(
    data=z_output,
    dims=['lat', 'lon', 'time'],
    coords=dict(
        lat=(['lat'], lats),
        lon=(['lon'], lons),
        time=times,
    ), attrs=dict(
        description='low cloud height, '+str(cf_c)+' threshold, '+method+' method',
        units='km',),)
    da_z = da_z.rename('zl')

    da_fmask = xr.DataArray(
    data=f_mask,
    dims=['lat', 'lon', 'time'],
    coords=dict(
        lat=(['lat'], lats),
        lon=(['lon'], lons),
        time=times,
    ), attrs=dict(
        description='low cloud mask, '+str(cf_c)+' threshold, '+method+' method',
        units='Boolean',),)
    da_fmask = da_fmask.rename('fl_mask')

    da_centroid = xr.DataArray(
    data=centroid_output,
    dims=['lat', 'lon', 'time'],
    coords=dict(
        lat=(['lat'], lats),
        lon=(['lon'], lons),
        time=times,
    ), attrs=dict(
        description='low cloud centroid, '+str(cf_c)+' threshold, '+method+' method',
        units='km',),)
    da_centroid = da_centroid.rename('centroid_height')

    if save:
        print('saving to '+path_to_data+'fl_'+method+'_0pt'+str(int(cf_c*100))+'.nc')
        da_f.to_netcdf(path_to_data+'fl_'+method+'_0pt'+str(int(cf_c*100))+'.nc')
        da_z.to_netcdf(path_to_data+'zl_'+method+'_0pt'+str(int(cf_c*100))+'.nc')
        da_fmask.to_netcdf(path_to_data+'flmask_'+method+'_0pt'+str(int(cf_c*100))+'.nc')


    print('shallow_top program completed')
    print('')
    return da_f, da_z, da_fmask, da_centroid

def shallowmid_top(cf_array, heights, times, lats, lons, method, save=False, cf_c=0.03):
    """
    Returns the altitude of the shallow cloud top, the cloud fraction
    at this altitude, and the mask
    if method is 'centroid':
                the shallow cloud top is defined as the cloud fraction peak (in the vertical)
                which is the closest to the centroid (centroid of the vertical profile of the CF)
                and where the cloud fraction is greater than cf_c.
    if method is 'max':
                the shallow cloud top is defined as the maximal cloud fraction peak (in the vertical)
                and where the cloud fraction is greater than cf_c.

    """
    nan = False
    if nan:
        fill_value = np.nan
    else:
        fill_value = 0
    f_output = np.empty_like(cf_array[0,:,:,:])
    z_output = np.empty_like(cf_array[0,:,:,:])
    centroid_output = np.empty_like(cf_array[0,:,:,:])
    path_to_data = '/data/bmckim/identified_cloud_fraction/'
    print('save = '+str(save))
    print('method = '+method)
    print('loop is '+str(np.shape(cf_array)[1])+' iterations long')
    print('starting loop')
    for i in range(np.shape(cf_array)[1]):
        if i % 5 == 0:
            print('i = '+str(i))
        for j in range(np.shape(cf_array)[2]):
            for k in range(np.shape(cf_array)[3]):
                if (np.sum(~np.isnan(cf_array[:,i,j,k]))==0):
                    # if the entire column is nan, set outputs to nan
                    f_output[i,j,k] = fill_value
                    z_output[i,j,k] = fill_value

                else:
                    # find the local maxima
                    local_max_indices = argrelextrema(cf_array[:,i,j,k], comparator=np.greater_equal)
                    f_list = []
                    z_list = []
                    for index in local_max_indices[0]:
                        f_list.append(cf_array[index,i,j,k])
                        z_list.append(heights[index])
                    if len(f_list)==0:
                        # if no local maxima found, set outputs to nan
                        f_output[i,j,k] = fill_value
                        z_output[i,j,k] = fill_value
                    # return a nan if none of the maxes are below the threshold (cf_c)
                    # print(f_list)
                    if np.sum(np.array(f_list)>cf_c)==0:
                        f_output[i,j,k] = fill_value
                        z_output[i,j,k] = fill_value

                    else:
                        if method=='centroid':
                            # find the cloud centroid
                            f_profilemasked = cf_array[:,i,j,k][~np.isnan(cf_array[:,i,j,k])]
                            h_profilemasked = heights[~np.isnan(cf_array[:,i,j,k])]
                            z_centroid = np.ma.average(h_profilemasked, weights=f_profilemasked)
                            dist_from_centroid = np.abs(z_centroid - z_list)
                            # choose the maxima closest to the centroid
                            z_id = z_list[np.argmin(dist_from_centroid)]
                            f_id = f_list[np.argmin(dist_from_centroid)]

                        if method=='max':
                            z_id = z_list[np.argmax(f_list)]
                            f_id = f_list[np.argmax(f_list)]
                            z_centroid = np.nan

                        f_output[i,j,k] = f_id
                        z_output[i,j,k] = z_id
                        centroid_output[i,j,k] = z_centroid

    f_mask = ~np.isnan(f_output)

    da_f = xr.DataArray(
    data=f_output,
    dims=['lat', 'lon', 'time'],
    coords=dict(
        lat=(['lat'], lats),
        lon=(['lon'], lons),
        time=times,
    ), attrs=dict(
        description='low cloud fraction, '+str(cf_c)+' threshold, '+method+' method',
        units='unitless',),)
    da_f = da_f.rename('fl')

    da_z = xr.DataArray(
    data=z_output,
    dims=['lat', 'lon', 'time'],
    coords=dict(
        lat=(['lat'], lats),
        lon=(['lon'], lons),
        time=times,
    ), attrs=dict(
        description='low cloud height, '+str(cf_c)+' threshold, '+method+' method',
        units='km',),)
    da_z = da_z.rename('zl')

    da_fmask = xr.DataArray(
    data=f_mask,
    dims=['lat', 'lon', 'time'],
    coords=dict(
        lat=(['lat'], lats),
        lon=(['lon'], lons),
        time=times,
    ), attrs=dict(
        description='low cloud mask, '+str(cf_c)+' threshold, '+method+' method',
        units='Boolean',),)
    da_fmask = da_fmask.rename('fl_mask')

    da_centroid = xr.DataArray(
    data=centroid_output,
    dims=['lat', 'lon', 'time'],
    coords=dict(
        lat=(['lat'], lats),
        lon=(['lon'], lons),
        time=times,
    ), attrs=dict(
        description='low cloud centroid, '+str(cf_c)+' threshold, '+method+' method',
        units='km',),)
    da_centroid = da_centroid.rename('centroid_height')

    if save:
        print('saving to '+path_to_data+'flm_'+method+'_0pt'+str(int(cf_c*100))+'.nc')
        da_f.to_netcdf(path_to_data+'flm_'+method+'_0pt'+str(int(cf_c*100))+'.nc')
        da_z.to_netcdf(path_to_data+'zlm_'+method+'_0pt'+str(int(cf_c*100))+'.nc')
        da_fmask.to_netcdf(path_to_data+'flmmask_'+method+'_0pt'+str(int(cf_c*100))+'.nc')


    print('shallowmid_top program completed')
    print('')
    return da_f, da_z, da_fmask, da_centroid

def group_data(data):
    '''
    Returns the data grouped into its annual averages (by El Nino, i.e. July to June)
    Note: Requires 2D data--height and time (e.g. already zonally averaged)
    '''
    # group data into years from June-May, then average, then save to a list
    years = [2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]
    months_a = [7, 8, 9, 10, 11, 12]
    months_b = [1, 2, 3, 4, 5, 6]
    data_list = []
    for year in years:
        list_a = []
        list_b = []
        for mon in months_a:
            list_a.append(str(year)+'-'+str(mon))
        for mon in months_b:
            list_b.append(str(year+1)+'-'+str(mon))
        list_total = list_a + list_b
        # print(list_total)
        data_list.append(data.sel(time=list_total,method='nearest').mean('time'))
    data_list = np.array(data_list)
    data_list = data_list[1:-1]               # remove first and last year (2005-2006, 2016-2017)
    return data_list


def find_albedos():
    '''
    Returns the inferred surface, low cloud, and high cloud albedo.
    First, it assumes fh=fl=0 so that the surface albedo (asurf) can be determined.
    Second, it looks for places where fh=0 or nan so that the low cloud albedo (al) can be determined.
    Third, it uses the full formula for cloud radiative effect so that the high cloud albedo (ah) can be determined.
    '''
    ceres = load_ceres()
    a_surf = ceres.scs/ceres.solar_mon # clear-sky upwelling SW / downwelling solar (both at TOA)

    # load in the cloud data
    # fl = xr.open_mfdataset('/data/bmckim/identified_cloud_fraction/fl_centroid_0pt20.nc').fl
    # fh = xr.open_mfdataset('/data/bmckim/identified_cloud_fraction/fh_centroid_0pt10.nc').fh
    # print('fl = '+str(fl.mean(('lon','lat','time')).values))
    # print('fh = '+str(fh.mean(('lon','lat','time')).values))

    # # to find al, look where fh=0 or nan, but where fl is defined
    # fl_masked = fl.where(np.isnan(fh)).where(~np.isnan(ceres.s))
    # ceres_masked = ceres.where(np.isnan(fh)).where(~np.isnan(fl))
    # a_surf_masked = a_surf.where(np.isnan(fh)).where(~np.isnan(fl))

    # numerator = (ceres_masked.s - ceres_masked.solar_mon * a_surf_masked)
    # denominator = (fl_masked * ceres_masked.solar_mon * (1 - a_surf_masked))
    # print('numerator = '+str(numerator.mean().values))
    # print('denominator = '+str(denominator.mean().values))

    # al = (ceres_masked.s - ceres_masked.solar_mon * a_surf_masked) / (fl_masked * ceres_masked.solar_mon * (1 - a_surf_masked))
    # # al = -(ceres_masked.toa_cre_net_mon - 2*10*fl_masked) / (ceres_masked.solar_mon * (1-a_surf_masked) * fl_masked)
    # # al = - ceres_masked.toa_cre_sw_mon / (ceres_masked.solar_mon * (1 - a_surf_masked) * fl_masked)
    # # al = numerator / denominator
    # print('asurf = '+str(a_surf.mean().values))
    # print('al = '+str(al.mean().values))
    # return a_surf, al
    return a_surf

def find_albedos2():
    '''
    Returns the inferred surface, low cloud, and high cloud albedo.

    It breaks the tropics into ascending and descending regions based on the omega500 metric. Then it takes the ascending-averaged observed shortwave cloud radiative effect and matches it to our equation for shortwave cloud radiative effect. It does the same for the descending-averaged shortwave cloud radiative effect. Then we have two constraints and two unknowns. Because the equation is nonlinear, we use a python function to solve the system of equations, resulting in a calculation of the high and low cloud albedos.
    '''
    ceres = load_ceres()
    a_surf = ceres.scs/ceres.solar_mon # clear-sky upwelling SW / downwelling solar (both at TOA)
    return a_surf

def cloud_temps(cf_c):
    '''
    Returns the low and high cloud temperatures given the cutoff of cloud fraction (cf_c).
    '''
    # load in the cloud heights
    zl = xr.open_mfdataset('/data/bmckim/identified_cloud_fraction/zl_centroid_0pt'+cf_c+'.nc').zl
    zh = xr.open_mfdataset('/data/bmckim/identified_cloud_fraction/zh_centroid_0pt'+cf_c+'.nc').zh

    # load in the air data
    z = xr.open_dataset('/data/bmckim/identified_cloud_fraction/era5_z_data.nc').z
    t = xr.open_dataset('/data/bmckim/identified_cloud_fraction/era5_t_data.nc').t

    # find the cloud temperature to within a 200 meter threshold
    threshold=0.2 # km
    tl = t.where(z<zl+threshold).where(z>zl-threshold).mean('level')
    th = t.where(z<zh+threshold).where(z>zh-threshold).mean('level')
    return tl, th


def predicted_allsky(cf_c,albedo,low_clouds=True):
    '''
    Returns the predicted all-sky OLR (r) and all-sky shortwave (s) given the cutoff of cloud fraction (cf_c).
    '''
    # load in the cloud fractions

    fh = xr.open_mfdataset('/data/bmckim/identified_cloud_fraction/fh_centroid_0pt'+cf_c+'.nc').fh
    if low_clouds:
        fl = xr.open_mfdataset('/data/bmckim/identified_cloud_fraction/fl_centroid_0pt'+cf_c+'.nc').fl
    else:
        fl = xr.zeros_like(fh)
    # fl = xr.open_mfdataset('/data/bmckim/identified_cloud_fraction/fl_centroid_0pt7.nc').fl
    # fh = xr.open_mfdataset('/data/bmckim/identified_cloud_fraction/fh_centroid_0pt3.nc').fh

    # load in the cloud temperatures
    tl, th = cloud_temps(cf_c)

    # load in the ceres dataset
    ceres = load_ceres()

    # load in the surface temperature
    ts = load_ts(dataset='hadcrut5')
    # interpolate onto the cloud grid
    # ts = ts.interp(lat=fh.lat.values).interp(lon=fh.lon.values).interp(time=fh.time.values)

    # load in the surface albedo
    a_surf = find_albedos()

    # constants
    lamcs = -2
    al, ah = (albedo, albedo)

    # make predictions
    r = ceres.rcs*(1-fh) + cc.sigma*th**4*fh + lamcs*(ts-tl)*fl*(1-fh)
    s = ceres.solar_mon*(1-a_surf)*(1-ah*fh)*(1-al*fl)

    # predicted CRE
    cre_lw = - (r - ceres.rcs)
    cre_sw = s - (ceres.solar_mon-ceres.scs)
    cre = cre_sw + cre_lw

    lowcloud_lw = lamcs*(ts-tl)*fl
    lowcloud_sw = ceres.solar_mon*(1-a_surf)*al*fl

    # predicted CRE_h
    fl = xr.zeros_like(fh)
    # make predictions
    sh = ceres.solar_mon*(1-a_surf)*(1-ah*fh)*(1-al*fl)
    cre_lwh = - (rh - ceres.rcs)
    cre_swh = sh - (ceres.solar_mon-ceres.scs)
    cre_h = cre_swh + cre_lwh

    fl = xr.open_mfdataset('/data/bmckim/identified_cloud_fraction/fl_centroid_0pt'+cf_c+'.nc').fl
    lowcloud_lwh = lamcs * (ts - tl) * fl * fh
    lowcloud_swh = ceres.solar_mon * (1-a_surf) * al * fl * ah * fh


    # r = rcs*(1-fh) + cc.sigma*th**4*fh + lamcs*(ts-tl)*fl*(1-fh)
    # solar * (1-a_s) * (1-ah*fh) * (1-al*fl)
    return r, s, cre, cre_lw, cre_sw, lowcloud_lw, lowcloud_sw, cre_h, lowcloud_lwh, lowcloud_swh

def predicted_allsky2(cf_c,albedo,method='centroid'):
    '''
    Returns the predicted all-sky OLR (r) and all-sky shortwave (s) given the cutoff of cloud fraction (cf_c).
    '''
    # load in the cloud fractions
    if method=='centroid':
        fh = xr.open_mfdataset('/data/bmckim/identified_cloud_fraction/fh_centroid_0pt'+cf_c+'.nc').fh
        fl = xr.open_mfdataset('/data/bmckim/identified_cloud_fraction/fl_centroid_0pt'+cf_c+'.nc').fl

    elif method=='max':
        fh = xr.open_mfdataset('/data/bmckim/identified_cloud_fraction/fh_max_0pt'+cf_c+'.nc').fh
        fl = xr.open_mfdataset('/data/bmckim/identified_cloud_fraction/fl_max_0pt'+cf_c+'.nc').fl

        # fl = xr.open_mfdataset('/data/bmckim/identified_cloud_fraction/fl_centroid_0pt'+cf_c+'.nc').fl

    # Fill low cloud nans with 0s and only consider low clouds beneath anvil clouds
    fl = fl.fillna(0).where(~np.isnan(fh))
    # fh = fh.fillna(0)
    # fl = fl.fillna(0)
    # fl = fl+0.02

    # fl = xr.open_mfdataset('/data/bmckim/identified_cloud_fraction/fl_centroid_0pt7.nc').fl
    # fh = xr.open_mfdataset('/data/bmckim/identified_cloud_fraction/fh_centroid_0pt3.nc').fh

    # load in the cloud temperatures
    tl, th = cloud_temps(cf_c)

    # load in the ceres dataset
    ceres = load_ceres()
    ceres = ceres.where(~np.isnan(fh))

    # load in the surface temperature
    ts = load_ts(dataset='hadcrut5')
    # interpolate onto the cloud grid
    ts = ts.where(~np.isnan(fh))

    # load in the surface albedo
    a_surf = find_albedos()

    # constants
    lamcs = -2
    al, ah = (albedo, albedo)

    # make predictions
    # r = ceres.rcs*(1-fh) + cc.sigma*th**4*fh + lamcs*(ts-tl)*fl*(1-fh)
    # s = ceres.solar_mon*(1-a_surf)*(1-ah*fh)*(1-al*fl)

    # predicted CRE
    # cre_lw = - (r - ceres.rcs)
    # cre_sw = s - (ceres.solar_mon-ceres.scs)
    cre_lw = ceres.rcs*fh - cc.sigma*th**4*fh - lamcs*(ts-tl)*fl*(1-fh)
    cre_sw = ceres.solar_mon*(1-a_surf)*(-ah*fh - al*fl + ah*al*fh*fl)
    cre = cre_sw + cre_lw

    cre_h = -ceres.solar_mon*(1-a_surf)*ah*fh + ceres.rcs*fh - cc.sigma*th**4*fh
    cre_l = -ceres.solar_mon*(1-a_surf)*al*fl - lamcs*(ts-tl)*fl
    m_lh = fl*fh*(ceres.solar_mon*al*ah*(1-a_surf) + lamcs*(ts-tl))

    a = ceres.solar_mon*fh*fl
    b = -ceres.solar_mon*(1-a_surf)*(fh+fl)
    c = -ceres.cre+cre_lw

    cre_h_sw = -ceres.solar_mon*(1-a_surf)*ah*fh
    cre_h_lw = ceres.rcs*fh - cc.sigma*th**4*fh
    m_lh_sw = fl*fh*al*ah*ceres.solar_mon*(1-a_surf)


    return cre, cre_lw, cre_sw, cre_h, cre_l, m_lh, fh, fl, th, tl, ts, ceres.scs, ceres.rcs, a_surf, a, b, c, ceres.cre, cre_h_sw, cre_h_lw, m_lh_sw

def iris_feedback(cf_c,albedo,low_clouds=True):
    '''
    Returns the predicted anvil cloud area feedback "iris feedback"

    Takes the tropical average SST, but then the cloud-masked predicted radiative effects and the cloud-masked cloud fraction (weighted by it's area fraction)
    '''

    # load in surface temperature data
    ts = load_ts(dataset='hadcrut5')
    # restrict to tropics
    ts = ts.where(ts.lat<=30,drop=True).where(ts.lat>=-30,drop=True)
    # weight by area
    ts_weights = np.cos(np.deg2rad(ts.lat))
    ts_weights.name = "weights"
    ts_weighted = ts.weighted(ts_weights)
    # put into annual averages
    ts_annual = group_data(ts_weighted.mean(('lon','lat')))
    # get the average over the period
    ts_avg = np.mean(ts_annual)


    # load in the high cloud fraction
    fh = xr.open_mfdataset('/data/bmckim/identified_cloud_fraction/fh_centroid_0pt'+cf_c+'.nc').fh
    # weight by area
    cf_weights = np.cos(np.deg2rad(fh.lat))
    cf_weights.name = 'weights'
    fh_weighted = fh.weighted(cf_weights)
    # get the annual values
    fh_annual = group_data(fh_weighted.mean(('lon','lat')))
    fh_avg = np.mean(fh_annual)

    # get the area fraction considered
    area_annual = group_data(xr.ones_like(fh).where(~np.isnan(fh)).fillna(0).weighted(cf_weights).mean(('lon','lat')))
    area_average = np.mean(area_annual)


    # compute the anomalies
    ts_anom = ts_annual-np.mean(ts_annual)
    fh_anom = fh_annual-np.mean(fh_annual)

    # compute the change in f with warming
    df_dts, b, r, p, std_err = sp.stats.linregress(ts_anom, fh_anom)

    # compute the cre masked by the high cloud
    _, _, cre, _, _, lowcloud_lw, lowcloud_sw, cre_h, lowcloud_lwh, lowcloud_swh = predicted_allsky(cf_c,albedo,low_clouds)
    cre = cre.where(~np.isnan(fh)).weighted(cf_weights).mean(('lon','lat','time')).values
    lowcloud_lw = lowcloud_lw.where(~np.isnan(fh)).weighted(cf_weights).mean(('lon','lat','time')).values
    lowcloud_sw = lowcloud_sw.where(~np.isnan(fh)).weighted(cf_weights).mean(('lon','lat','time')).values

    cre_h = cre_h.where(~np.isnan(fh)).weighted(cf_weights).mean(('lon','lat','time')).values
    lowcloud_lwh = lowcloud_lwh.where(~np.isnan(fh)).weighted(cf_weights).mean(('lon','lat','time')).values
    lowcloud_swh = lowcloud_swh.where(~np.isnan(fh)).weighted(cf_weights).mean(('lon','lat','time')).values


    iris = 1/fh_avg * df_dts * area_average * (cre + lowcloud_lw + lowcloud_sw)
    iris_h = 1/fh_avg * df_dts * area_average * (cre_h + lowcloud_lwh + lowcloud_swh)

    iris_dict = {'iris_feedback':iris,
                 'cf_c':cf_c,
                 'albedo':albedo,
                 'fh_avg':fh_avg,
                 'df_dts':df_dts,
                 'area_average':area_average,
                 'cre':cre,
                 'lowcloud_lw':lowcloud_lw,
                 'lowcloud_sw':lowcloud_sw
                }

    iris_h_dict = {'iris_feedback':iris_h,
             'cf_c':cf_c,
             'albedo':albedo,
             'fh_avg':fh_avg,
             'df_dts':df_dts,
             'area_average':area_average,
             'cre_h':cre_h,
             'lowcloud_lw':lowcloud_lwh,
             'lowcloud_sw':lowcloud_swh
            }

    return iris_dict, iris_h_dict


def predicted_allsky3(cf_c,method='centroid',tuned=True, albedo_input=0):
    '''
    Returns the predicted all-sky OLR (r) and all-sky shortwave (s) given the cutoff of cloud fraction (cf_c).
    '''
    # load in the cloud fractions
    if method=='centroid':
        fh = xr.open_mfdataset('/data/bmckim/identified_cloud_fraction/fh_centroid_0pt'+cf_c+'.nc').fh
        fl = xr.open_mfdataset('/data/bmckim/identified_cloud_fraction/fl_centroid_0pt'+cf_c+'.nc').fl

    elif method=='max':
        fh = xr.open_mfdataset('/data/bmckim/identified_cloud_fraction/fh_max_0pt'+cf_c+'.nc').fh * 1.7
        fl = xr.open_mfdataset('/data/bmckim/identified_cloud_fraction/fl_max_0pt'+cf_c+'.nc').fl


    # Fill nans with 0s
    fl = fl.fillna(0)
    fh = fh.fillna(0)

    # load in the cloud temperatures
    tl, th = cloud_temps(cf_c)
    th = th.fillna(0)
    tl = tl.fillna(0)

    # load in the ceres dataset
    ceres = load_ceres()

    # load in the surface temperature
    ts = load_ts(dataset='hadcrut5')

    # load in the surface albedo
    a_surf = find_albedos()

    # constants
    lamcs = -2

    # predicted longwave cloud radiative effect
    cre_lw = ceres.rcs*fh - cc.sigma*th**4*fh - lamcs*(ts-tl)*fl*(1-fh)

    # fit the cloud albedo
    a = (area_weighted_mean(ceres.solar_mon*fh*fl)).mean().values
    b = (area_weighted_mean(-ceres.solar_mon*(1-a_surf)*(fh+fl))).mean().values
    c = (area_weighted_mean(-ceres.cre+cre_lw)).mean().values

    if tuned:
        albedo =  -b - np.sqrt(b**2 - 4*a*c/(2*a))
    else:
        albedo = albedo_input

    al, ah = (albedo, albedo)

    cre_sw = ceres.solar_mon*(1-a_surf)*(-ah*fh - al*fl + ah*al*fh*fl)
    cre = cre_sw + cre_lw

    cre_h = -ceres.solar_mon*(1-a_surf)*ah*fh + ceres.rcs*fh - cc.sigma*th**4*fh
    cre_l = -ceres.solar_mon*(1-a_surf)*al*fl - lamcs*(ts-tl)*fl
    m_lh = fl*fh*(ceres.solar_mon*al*ah*(1-a_surf) + lamcs*(ts-tl))

    cre_h_sw = -ceres.solar_mon*(1-a_surf)*ah*fh
    cre_h_lw = ceres.rcs*fh - cc.sigma*th**4*fh
    m_lh_sw = fl*fh*al*ah*ceres.solar_mon*(1-a_surf)

    tl, th = cloud_temps(cf_c)
    scs = ceres.solar_mon*(1-a_surf)
    rcs = ceres.rcs

    return cre, cre_lw, cre_sw, cre_h, cre_l, m_lh, fh, fl, th, tl, ts, ceres.scs, ceres.rcs, a_surf, a, b, c, ceres.cre, cre_h_sw, cre_h_lw, m_lh_sw, albedo, ceres.solar_mon, scs, rcs

def predicted_allsky4(cf_c,method='centroid',tuned=True, albedo_input=0):
    '''
    Returns the predicted all-sky OLR (r) and all-sky shortwave (s) given the cutoff of cloud fraction (cf_c).
    '''
    # load in the cloud fractions
    if method=='centroid':
        fh = xr.open_mfdataset('/data/bmckim/identified_cloud_fraction/fh_centroid_0pt'+cf_c+'.nc').fh
        fl = xr.open_mfdataset('/data/bmckim/identified_cloud_fraction/fl_centroid_0pt'+cf_c+'.nc').fl

    elif method=='max':
        fh = xr.open_mfdataset('/data/bmckim/identified_cloud_fraction/fh_max_0pt'+cf_c+'.nc').fh
        fl = xr.open_mfdataset('/data/bmckim/identified_cloud_fraction/fl_max_0pt'+cf_c+'.nc').fl


    # Fill nans with 0s
    fl = fl.fillna(0)
    fh = fh.fillna(0)

    # load in the cloud temperatures
    tl, th = cloud_temps(cf_c)
    th = th.fillna(0)
    tl = tl.fillna(0)

    # load in the ceres dataset
    ceres = load_ceres()

    # load in the surface temperature
    ts = load_ts(dataset='hadcrut5')

    # constants
    lamcs = -2

    # find the high cloud scaling factor
    n = ((area_weighted_mean(ceres.cre_lw)).mean('time') + (lamcs*area_weighted_mean((ts-tl)*fl))).mean('time') / (area_weighted_mean((ceres.rcs-cc.sigma*th**4+lamcs*(ts-tl)*fl)*fh)).mean('time').values

    # n = area_weighted_mean((ceres.cre_lw + lamcs*(ts-tl)*fl)/(ceres.rcs*fh-cc.sigma*th**4*fh+lamcs*(ts-tl)*fl*fh))
    # n = n.mean('time')

    # load in the surface albedo
    a_surf = find_albedos()



    # predicted longwave cloud radiative effect
    fh = n*fh
    cre_lw = ceres.rcs*fh - cc.sigma*th**4*fh - lamcs*(ts-tl)*fl*(1-fh)

    # fit the cloud albedo
    a = (area_weighted_mean(ceres.solar_mon*fh*fl)).mean().values
    b = (area_weighted_mean(-ceres.solar_mon*(1-a_surf)*(fh+fl))).mean().values
    # c = (area_weighted_mean(-ceres.cre+cre_lw)).mean().values
    c = (area_weighted_mean(-ceres.cre_sw)).mean().values

    if tuned:
        albedo =  -b - np.sqrt(b**2 - 4*a*c/(2*a))
    else:
        albedo = albedo_input

    al, ah = (albedo, albedo)

    cre_sw = ceres.solar_mon*(1-a_surf)*(-ah*fh - al*fl + ah*al*fh*fl)
    cre = cre_sw + cre_lw

    cre_h = -ceres.solar_mon*(1-a_surf)*ah*fh + ceres.rcs*fh - cc.sigma*th**4*fh
    cre_l = -ceres.solar_mon*(1-a_surf)*al*fl - lamcs*(ts-tl)*fl
    m_lh = fl*fh*(ceres.solar_mon*al*ah*(1-a_surf) + lamcs*(ts-tl))

    cre_h_sw = -ceres.solar_mon*(1-a_surf)*ah*fh
    cre_h_lw = ceres.rcs*fh - cc.sigma*th**4*fh
    m_lh_sw = fl*fh*al*ah*ceres.solar_mon*(1-a_surf)

    tl, th = cloud_temps(cf_c)
    scs = ceres.solar_mon*(1-a_surf)
    rcs = ceres.rcs

    return cre, cre_lw, cre_sw, cre_h, cre_l, m_lh, fh, fl, th, tl, ts, ceres.scs, ceres.rcs, a_surf, a, b, c, ceres.cre, cre_h_sw, cre_h_lw, m_lh_sw, albedo, ceres.solar_mon, scs, rcs, n


def predicted_allsky5(cf_c,method='max',tuned=True, albedo_input=0):
    '''
    Returns the predicted all-sky OLR (r) and all-sky shortwave (s) given the cutoff of cloud fraction (cf_c).
    '''
    # load in the cloud fractions
    if method=='centroid':
        fh = xr.open_mfdataset('/data/bmckim/identified_cloud_fraction/fh_centroid_0pt'+cf_c+'.nc').fh
        fl = xr.open_mfdataset('/data/bmckim/identified_cloud_fraction/flm_centroid_0pt'+cf_c+'.nc').fl

    elif method=='max':
        fh = xr.open_mfdataset('/data/bmckim/identified_cloud_fraction/fh_max_0pt'+cf_c+'.nc').fh
        fl = xr.open_mfdataset('/data/bmckim/identified_cloud_fraction/flm_max_0pt'+cf_c+'.nc').fl


    # Fill nans with 0s
    fl = fl.fillna(0)
    fh = fh.fillna(0)

    # load in the cloud temperatures
    tl, th = cloud_temps(cf_c)
    th = th.fillna(0)
    tl = tl.fillna(0)

    # load in the ceres dataset
    ceres = load_ceres()

    # load in the surface temperature
    ts = load_ts(dataset='hadcrut5')

    # load in the vertical velocities
    omega = load_omega()

    # constants
    lamcs = -2

    # find the high cloud scaling factor
    n = ((area_weighted_mean(ceres.cre_lw)).mean('time') + (lamcs*area_weighted_mean((ts-tl)*fl))).mean('time') / (area_weighted_mean((ceres.rcs-cc.sigma*th**4+lamcs*(ts-tl)*fl)*fh)).mean('time').values

    # predicted longwave cloud radiative effect
    fh = n*fh
    cre_lw = ceres.rcs*fh - cc.sigma*th**4*fh - lamcs*(ts-tl)*fl*(1-fh)

    ds = xr.merge([ceres.solar_mon.rename('solar_mon'), ceres.cre_sw.rename('cre_sw'), ceres.cre_lw.rename('cre_lw'), ceres.rcs.rename('rcs'), fh.rename('fh'), fl.rename('fl'), ts.rename('ts'),tl.rename('tl'),th.rename('th')])
    ds = ds.transpose('lat', 'lon', 'time')
    # print(ds.fh.shape)
    # print(ds.cre_lw.shape)
    # lon_grid, lat_grid = np.meshgrid(fh.lon,fh.lat)
    # globe_ocean_mask = globe.is_ocean(lat_grid, lon_grid)[:,:,np.newaxis]*np.ones((31, 144, 127))
    # print(np.shape(globe_ocean_mask))
    # ds = ds.where(globe_ocean_mask)

    # lon_grid, lat_grid, time_grid = np.meshgrid(ds.lon,ds.lat,ds.time)
    # globe_ocean_mask = np.broadcast_to(globe.is_ocean(lat_grid, lon_grid), (127,31,144))
    # globe_ocean_mask = globe.isocean(lat_grid,lon_grid,time_grid)
    # ds = ds.where(globe_ocean_mask)

#     ds_asc = ds.where(omega <= 20) # find the ascending regions
#     ds_dec = ds.where(omega >  20) # find the descending regions

#     # fit the cloud scaling factors
#     def cre_lw_equations(vars):
#         nh, nl = vars

#         a_asc = area_weighted_mean(ds_asc.rcs*ds_asc.fh).mean().values
#         b_asc = area_weighted_mean(-cc.sigma*ds_asc.th**4 * ds_asc.fh).mean().values
#         c_asc = area_weighted_mean(-lamcs*(ds_asc.ts-ds_asc.tl)*ds_asc.fl).mean().values
#         d_asc = area_weighted_mean(lamcs*(ds_asc.ts-ds_asc.tl)*ds_asc.fh*ds_asc.fl).mean().values
#         e_asc = area_weighted_mean(-ds_asc.cre_lw).mean().values

#         a_dec = area_weighted_mean(ds_dec.rcs*ds_dec.fh).mean().values
#         b_dec = area_weighted_mean(-cc.sigma*ds_dec.th**4 * ds_dec.fh).mean().values
#         c_dec = area_weighted_mean(-lamcs*(ds_dec.ts-ds_dec.tl)*ds_dec.fl).mean().values
#         d_dec = area_weighted_mean(lamcs*(ds_dec.ts-ds_dec.tl)*ds_dec.fh*ds_dec.fl).mean().values
#         e_dec = area_weighted_mean(-ds_dec.cre_lw).mean().values

#         eq_asc = a_asc*nh + b_asc*nh + c_asc*nl + d_asc*nh*nl + e_asc
#         eq_dec = a_dec*nh + b_dec*nh + c_dec*nl + d_dec*nh*nl + e_dec

#         return [eq_asc, eq_dec]

#     initial_guess = [1.0, 1.0]
#     solution = fsolve(cre_lw_equations, initial_guess)

#     nh, nl = solution

#     # return nh, nl

#     fh = nh*fh
#     fl = nl*fl

#     cre_lw = ceres.rcs*fh - cc.sigma*th**4*fh - lamcs*(ts-tl)*fl*(1-fh)


    # load in the surface albedo
    a_surf = find_albedos2()

    ds = xr.merge([ceres.solar_mon.rename('solar_mon'), ceres.cre_sw.rename('cre_sw'), ceres.cre_lw.rename('cre_lw'), ceres.rcs.rename('rcs'), fh.rename('fh'), fl.rename('fl'), ts.rename('ts'),tl.rename('tl'),th.rename('th')])
    ds = ds.transpose('lat', 'lon', 'time')

    ds_asc = ds.where(omega <= 25) # find the ascending regions
    ds_dec = ds.where(omega >  25) # find the descending regions


    # fit the cloud albedo
    def cre_sw_equations(vars):
        ah, al = vars

        a_asc = area_weighted_mean(-ds_asc.solar_mon*(1-a_surf)*ds_asc.fh).mean().values
        b_asc = area_weighted_mean(-ds_asc.solar_mon*(1-a_surf)*ds_asc.fl).mean().values
        c_asc = area_weighted_mean(ds_asc.solar_mon*(1-a_surf)*ds_asc.fh*ds_asc.fl).mean().values
        d_asc = area_weighted_mean(-ds_asc.cre_sw).mean().values

        a_dec = area_weighted_mean(-ds_dec.solar_mon*(1-a_surf)*ds_dec.fh).mean().values
        b_dec = area_weighted_mean(-ds_dec.solar_mon*(1-a_surf)*ds_dec.fl).mean().values
        c_dec = area_weighted_mean(ds_dec.solar_mon*(1-a_surf)*ds_dec.fh*ds_dec.fl).mean().values
        d_dec = area_weighted_mean(-ds_dec.cre_sw).mean().values


        eq_asc = a_asc*ah + b_asc*al + c_asc*ah*al + d_asc
        eq_dec = a_dec*ah + b_dec*al + c_dec*ah*al + d_dec

        return [eq_asc, eq_dec]

    initial_guess = [0.5, 0.5]
    solution = fsolve(cre_sw_equations, initial_guess)

    ah, al = solution

#     # return n, a_surf, ah, al

    cre_sw = ceres.solar_mon*(1-a_surf)*(-ah*fh - al*fl + ah*al*fh*fl)
    cre = cre_sw + cre_lw

    cre_h = -ceres.solar_mon*(1-a_surf)*ah*fh + ceres.rcs*fh - cc.sigma*th**4*fh
    cre_l = -ceres.solar_mon*(1-a_surf)*al*fl - lamcs*(ts-tl)*fl
    m_lh = fl*fh*(ceres.solar_mon*al*ah*(1-a_surf) + lamcs*(ts-tl))

    cre_l_sw = -ceres.solar_mon*(1-a_surf)*al*fl
    cre_l_lw = - lamcs*(ts-tl)*fl
    cre_h_sw = -ceres.solar_mon*(1-a_surf)*ah*fh
    cre_h_lw = ceres.rcs*fh - cc.sigma*th**4*fh
    m_lh_sw = fl*fh*al*ah*ceres.solar_mon*(1-a_surf)

    tl, th = cloud_temps(cf_c)
    scs = ceres.solar_mon*(1-a_surf)
    rcs = ceres.rcs

    return cre, cre_lw, cre_sw, cre_h, cre_l, m_lh, fh, fl, th, tl, ts, ceres.scs, ceres.rcs, a_surf, ceres.cre, cre_h_sw, cre_h_lw, m_lh_sw, ah, al, ceres.solar_mon, scs, rcs, n, cre_l_sw, cre_l_lw

    # return cre, cre_lw, cre_sw, cre_h, cre_l, m_lh, fh, fl, th, tl, ts, ceres.scs, ceres.rcs, a_surf, ceres.cre, cre_h_sw, cre_h_lw, m_lh_sw, ah, al, ceres.solar_mon, scs, rcs, nh, nl


def albedo_variabilities(n,cf_c,method='centroid'):
    '''
    Returns the predicted all-sky OLR (r) and all-sky shortwave (s) given the cutoff of cloud fraction (cf_c).
    '''
    # load in the cloud fractions
    if method=='centroid':
        fh = xr.open_mfdataset('/data/bmckim/identified_cloud_fraction/fh_centroid_0pt'+cf_c+'.nc').fh
        fl = xr.open_mfdataset('/data/bmckim/identified_cloud_fraction/flm_centroid_0pt'+cf_c+'.nc').fl

    elif method=='max':
        fh = xr.open_mfdataset('/data/bmckim/identified_cloud_fraction/fh_max_0pt'+cf_c+'.nc').fh
        fl = xr.open_mfdataset('/data/bmckim/identified_cloud_fraction/flm_max_0pt'+cf_c+'.nc').fl


    # Fill nans with 0s
    fl = fl.fillna(0)
    fh = fh.fillna(0)

    fh = n*fh

    # load in the cloud temperatures
    tl, th = cloud_temps(cf_c)
    th = th.fillna(0)
    tl = tl.fillna(0)

    # load in the ceres dataset
    ceres = load_ceres()

    # load in the surface temperature
    ts = load_ts(dataset='hadcrut5')

    # load in the surface albedo
    a_surf = find_albedos()

    # load in the vertical velocities
    omega = load_omega()

    # constants
    lamcs = -2

    # predicted longwave cloud radiative effect
    cre_lw = ceres.rcs*fh - cc.sigma*th**4*fh - lamcs*(ts-tl)*fl*(1-fh)

    ds = xr.merge([ceres.solar_mon.rename('solar_mon'), ceres.cre_sw.rename('cre_sw'), ceres.cre_lw.rename('cre_lw'), ceres.rcs.rename('rcs'), fh.rename('fh'), fl.rename('fl'), ts.rename('ts'),tl.rename('tl'),th.rename('th')])
    ds = ds.transpose('lat', 'lon', 'time')

    print('dataset merged...')

    ds_asc = ds.where(omega <= 25) # find the ascending regions
    ds_dec = ds.where(omega >  25) # find the descending regions

    print('grouping ascending data...')

    a_asc = group_data(area_weighted_mean(-ds_asc.solar_mon*(1-a_surf)*ds_asc.fh))
    b_asc = group_data(area_weighted_mean(-ds_asc.solar_mon*(1-a_surf)*ds_asc.fl))
    c_asc = group_data(area_weighted_mean(ds_asc.solar_mon*(1-a_surf)*ds_asc.fh*ds_asc.fl))
    d_asc = group_data(area_weighted_mean(-ds_asc.cre_sw))

    print('grouping descending data...')

    a_dec = group_data(area_weighted_mean(-ds_dec.solar_mon*(1-a_surf)*ds_dec.fh))
    b_dec = group_data(area_weighted_mean(-ds_dec.solar_mon*(1-a_surf)*ds_dec.fl))
    c_dec = group_data(area_weighted_mean(ds_dec.solar_mon*(1-a_surf)*ds_dec.fh*ds_dec.fl))
    d_dec = group_data(area_weighted_mean(-ds_dec.cre_sw))

    # fit the cloud albedo
    print('solving for cloudy albedos')
    ah_list = []
    al_list = []

    for i in range(len(a_asc)):
        def cre_sw_equations(vars):
            ah, al = vars

            eq_asc = a_asc[i]*ah + b_asc[i]*al + c_asc[i]*ah*al + d_asc[i]
            eq_dec = a_dec[i]*ah + b_dec[i]*al + c_dec[i]*ah*al + d_dec[i]

            return [eq_asc, eq_dec]

        initial_guess = [0.5, 0.5]
        solution = fsolve(cre_sw_equations, initial_guess)

        ah_i, al_i = solution
        ah_list.append(ah_i)
        al_list.append(al_i)
        print('i = '+str(i))
        print('ah_i = '+str(ah_i))
        print('al_i = '+str(al_i))
        print()

    ah_array = np.array(ah_list)
    al_array = np.array(al_list)

#     # fit the cloud albedo for every year
#     a = group_data(area_weighted_mean(ceres.solar_mon*fh*fl))
#     b = group_data(area_weighted_mean(-ceres.solar_mon*(1-a_surf)*(fh+fl)))
#     c = group_data(area_weighted_mean(-ceres.cre+cre_lw))
#     albedo =  -b - np.sqrt(b**2 - 4*a*c/(2*a))
    # return albedo

    return ah_array, al_array

def albedo_variability(n,cf_c,method='centroid'):
    '''
    Returns the predicted all-sky OLR (r) and all-sky shortwave (s) given the cutoff of cloud fraction (cf_c).
    '''
    # load in the cloud fractions
    if method=='centroid':
        fh = xr.open_mfdataset('/data/bmckim/identified_cloud_fraction/fh_centroid_0pt'+cf_c+'.nc').fh
        fl = xr.open_mfdataset('/data/bmckim/identified_cloud_fraction/flm_centroid_0pt'+cf_c+'.nc').fl

    elif method=='max':
        fh = xr.open_mfdataset('/data/bmckim/identified_cloud_fraction/fh_max_0pt'+cf_c+'.nc').fh
        fl = xr.open_mfdataset('/data/bmckim/identified_cloud_fraction/flm_max_0pt'+cf_c+'.nc').fl


    # Fill nans with 0s
    fl = fl.fillna(0)
    fh = fh.fillna(0)

    fh = n*fh

    # load in the cloud temperatures
    tl, th = cloud_temps(cf_c)
    th = th.fillna(0)
    tl = tl.fillna(0)

    # load in the ceres dataset
    ceres = load_ceres()

    # load in the surface temperature
    ts = load_ts(dataset='hadcrut5')

    # load in the surface albedo
    a_surf = find_albedos()

    # constants
    lamcs = -2

    # predicted longwave cloud radiative effect
    cre_lw = ceres.rcs*fh - cc.sigma*th**4*fh - lamcs*(ts-tl)*fl*(1-fh)

    # fit the cloud albedo for every year
    a = group_data(area_weighted_mean(ceres.solar_mon*fh*fl))
    b = group_data(area_weighted_mean(-ceres.solar_mon*(1-a_surf)*(fh+fl)))
    c = group_data(area_weighted_mean(-ceres.cre+cre_lw))

    albedo =  -b - np.sqrt(b**2 - 4*a*c/(2*a))


    return albedo
