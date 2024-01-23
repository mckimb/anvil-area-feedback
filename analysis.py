import numpy as np
from faceted import faceted as fc
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib as mpl
import pretty_plotting_funcs as ppf
from matplotlib import cm
import cartopy.crs as ccrs
import warnings
from scipy.signal import argrelextrema, argrelmax
import scipy as sp
import cloud_program as cp
from cartopy.util import add_cyclic_point
import cartopy.util as cutil
import statistics as st
from matplotlib.pyplot import cm
import matplotlib.patches as mpatches
warnings.filterwarnings('ignore')

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['text.usetex'] = True
mpl.rcParams['figure.dpi']= 600
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
lightblue = '#bde0e7'
lightblue = '#8dcad6'
blue = '#2b7db5'
gold = '#d19711'


# take the raw cloud fraction data and process into two layers suitable for predicting all sky radiative Effects.
# load in the high cloud data with 0.3 < tau < 5
cf_high = cp.load_highcf()
# apply rolling average, look above 8km, etc
cfh_array, heights, times, lats, lons = cp.process_highcf(cf_high)
fh_0pt03_max, zh_0pt03_max, _, _ = cp.anvil_top(cfh_array, heights, times, lats, lons, method='max', save=True, cf_c=0.03)

# load in the low cloud data
cf_low = cp.load_lowcf()
# apply rolling average, look below 8 km, etc
cfl_array, heights, times, lats, lons = cp.process_lowcf(cf_low)
fl_0pt03_max, zl_0pt03_max, _, _ = cp.shallow_top(cfl_array, heights, times, lats, lons, method='centroid', save=True, cf_c=0.03)


# caculate inferred cloud radiative effects
cre, cre_lw, cre_sw, cre_h, cre_l, m_lh, fh, fl, th, tl, ts, scs, rcs, a_surf, ceres_cre, cre_h_sw, cre_h_lw, m_lh_sw, ah, al, insol, scs, rcs, n, cre_l_lw, cre_l_sw = cp.predicted_allsky5(cf_c='3', method='max')

# store the results in a dictionary
table1_dict = {'anvil area fraction':(cp.area_weighted_mean(fh)).mean().values,
         'low cloud fraction':(cp.area_weighted_mean(fl)).mean().values,
         'anvil temperature':(cp.area_weighted_mean(th)).mean().values,
         'low cloud temperature':(cp.area_weighted_mean(tl)).mean().values,
         'surface temperature':(cp.area_weighted_mean(ts)).mean().values,
         'ah':ah,
         'al':al,
         'surface albedo':(cp.area_weighted_mean(a_surf)).mean().values,
         'incoming shortwave radiation':(cp.area_weighted_mean(insol)).mean().values,
         'shortwave in clear-skies':(cp.area_weighted_mean(scs)).mean().values,
         'olr in clear-skies':(cp.area_weighted_mean(rcs)).mean().values,
         'cloud radiative effect':(cp.area_weighted_mean(cre)).mean().values,
         'shortwave cre':(cp.area_weighted_mean(cre_sw)).mean().values,
         'longwave cre': (cp.area_weighted_mean(cre_lw)).mean().values,
         'anvil cre': (cp.area_weighted_mean(cre_h)).mean().values,
         'low cre': (cp.area_weighted_mean(cre_l)).mean().values,
         'overlap': (cp.area_weighted_mean(m_lh)).mean().values,
         'shortwave overlap': (cp.area_weighted_mean(m_lh_sw)).mean().values,
         'longwave overlap': (cp.area_weighted_mean(m_lh-m_lh_sw)).mean().values,
         'anvil_and_overlap': (cp.area_weighted_mean(cre_h+m_lh)).mean().values
        }

# obtain observed cloud radiative effects
ceres = cp.load_ceres()
table2_dict = {
         'cloud radiative effect':(cp.area_weighted_mean(ceres.cre)).mean().values,
         'shortwave cre':(cp.area_weighted_mean(ceres.cre_sw)).mean().values,
         'longwave cre': (cp.area_weighted_mean(ceres.cre_lw)).mean().values
        }

        #####################################################################
        ######################### PLOT CERES CRE (Figure 2a) ############################
        #####################################################################
        ceres = cp.load_ceres()
        fig, axes, cax = fc(3,1, width=4, aspect=60/180, bottom_pad=-0.5, cbar_mode='single', internal_pad = -0.53, cbar_pad=0., cbar_location = 'bottom', cbar_short_side_pad=0, axes_kwargs={'projection': ccrs.PlateCarree(central_longitude=180)})

        lon = 0
        ax = axes[0]
        cmap='RdBu_r'
        cdata, cyclic_lons = add_cyclic_point(ceres.cre.mean('time').values, coord=ceres.cre.lon.values)
        c1 = ax.contourf(cyclic_lons, cre.lat.values, cdata, add_colorbar=False, vmin=-80, vmax=80, cmap=cmap, transform=ccrs.PlateCarree(central_longitude=lon), levels=np.linspace(-80,80,17), extend='both')
        c2 = ax.contour( cyclic_lons, cre.lat.values, cdata, add_colorbar=False, vmin=-80, vmax=80, transform=ccrs.PlateCarree(central_longitude=lon), levels=np.linspace(-80,80,17), colors='k', linewidths=0.2, alpha=0.2, extend='both')
        ax.set_title('')
        ax.coastlines(linewidth=0.6)
        t1 = ax.text(-178,20,'$C$ = '+str(np.round(table2_dict['cloud radiative effect'],1)), color='k',fontsize='small')
        t1.set_bbox(dict(facecolor='white', alpha=0.95, linewidth=0,pad=1.))
        ax.tick_params(axis='x', colors='dimgray', width=0)
        ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
        ax.set_xticklabels(['','','','','','',''],color='k', fontsize='x-small')


        ax.add_patch(mpatches.Rectangle(xy=[80, -15], width=70, height=30, facecolor=None, edgecolor='k',
                                            alpha=1, fill=False,linewidth=1,transform=ccrs.PlateCarree()))
        ax.add_patch(mpatches.Rectangle(xy=[-115, -29], width=40, height=20, facecolor=None, edgecolor='k',
                                            alpha=1, fill=False,linewidth=1,transform=ccrs.PlateCarree()))

        ax = axes[1]
        cdata, cyclic_lons = add_cyclic_point(ceres.cre_sw.mean('time').values, coord=ceres.cre.lon.values)
        c1 = ax.contourf(cyclic_lons, cre.lat.values, cdata, add_colorbar=False, vmin=-80, vmax=80, cmap=cmap, transform=ccrs.PlateCarree(central_longitude=lon), levels=np.linspace(-80,80,17), extend='both')
        c2 = ax.contour( cyclic_lons, cre.lat.values, cdata, add_colorbar=False, vmin=-80, vmax=80, transform=ccrs.PlateCarree(central_longitude=lon), levels=np.linspace(-80,80,17), colors='k', linestyles='solid', linewidths=0.2, alpha=0.2, extend='both')
        ax.set_title('')
        ax.coastlines(linewidth=0.6)
        t1 = ax.text(-178,20,'$C^{sw}$ = '+str(np.round(table2_dict['shortwave cre'],1)), color='k',fontsize='small')
        t1.set_bbox(dict(facecolor='white', alpha=0.95, linewidth=0,pad=1.))
        ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
        ax.tick_params(axis='x', colors='silver', width=0)
        ax.set_xticklabels(['','','','','','',''],color='k', fontsize='x-small')


        ax = axes[2]
        cdata, cyclic_lons = add_cyclic_point(ceres.cre_lw.mean('time').values, coord=ceres.cre.lon.values)
        c1 = ax.contourf(cyclic_lons, cre.lat.values, cdata, add_colorbar=False, vmin=-80, vmax=80, cmap=cmap, transform=ccrs.PlateCarree(central_longitude=lon), levels=np.linspace(-80,80,17), extend='both')
        c2 = ax.contour( cyclic_lons, cre.lat.values, cdata, add_colorbar=False, vmin=-80, vmax=80, transform=ccrs.PlateCarree(central_longitude=lon), levels=np.linspace(-80,80,17), colors='k', linewidths=0.2, linestyles='solid', alpha=0.2, extend='both')
        ax.set_title('')
        ax.coastlines(linewidth=0.6)
        t1 = ax.text(-178,20,'$C^{lw}$ = '+str(np.round(table2_dict['longwave cre'],1)), color='k',fontsize='small')
        t1.set_bbox(dict(facecolor='white', alpha=0.95, linewidth=0,pad=1.))
        ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
        ax.tick_params(axis='x', colors='silver', width=0)
        ax.set_xticklabels([r'$0^\circ$',r'$60^\circ$ E', r'$120^\circ$ E', r'$180^\circ$', r'$120^\circ$ W', r'$60^\circ$ W', r'$0^\circ$'],color='k', fontsize='x-small')


        for ax in axes:
            ax.tick_params(axis='both', direction='in', top=True, right=True, left=True, bottom=True)
            ax.spines['geo'].set_edgecolor('silver')
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(1.5)

            ax.set_yticks([-15, 0, 15])
            ax.tick_params(axis='y', colors='silver', width=0)
            ax.set_yticklabels([r'$15^\circ$ S', r'$0^\circ$', r'$15^\circ$ N'],color='k', fontsize='x-small')

        plt.colorbar(c1, cax=cax, orientation='horizontal', label='Cloud Radiative Effects / Wm$^{-2}$', ticks=np.linspace(-80,80,9), extend='both')

        #####################################################################
        ###################### PLOT PREDICTED CRE (Figure 2b) ###########################
        #####################################################################
        fig, axes, cax = fc(3,1, width=4, aspect=60/180, bottom_pad=-0.5, cbar_mode='single', internal_pad = -0.53, cbar_pad=0, cbar_location = 'bottom', cbar_short_side_pad=0, axes_kwargs={'projection': ccrs.PlateCarree(central_longitude=180)})
        size='small'
        lon = 0
        ax = axes[0]
        cdata, cyclic_lons = add_cyclic_point(cre.mean('time').values, coord=cre.lon.values)
        c1 = ax.contourf(cyclic_lons, cre.lat.values, cdata, add_colorbar=False, vmin=-80, vmax=80, cmap='RdBu_r', transform=ccrs.PlateCarree(central_longitude=lon), levels=np.linspace(-80,80,17), extend='both')
        c2 = ax.contour( cyclic_lons, cre.lat.values, cdata, add_colorbar=False, vmin=-80, vmax=80, transform=ccrs.PlateCarree(central_longitude=lon), levels=np.linspace(-80,80,17), colors='k', linewidths=0.2, alpha=0.2, linestyles='solid', extend='both')
        ax.set_title('')
        ax.coastlines(linewidth=0.6)
        t1=ax.text(-178,20,'$C$ = '+str(np.round(table1_dict['cloud radiative effect'],1)), color='k',fontsize=size)
        t1.set_bbox(dict(facecolor='white', alpha=0.95, linewidth=0,pad=1.))
        ax.tick_params(axis='x', colors='dimgray')
        ax.set_xticklabels(['','','','','','',''],color='k', fontsize='x-small')


        ax = axes[1]
        cdata, cyclic_lons = add_cyclic_point(cre_sw.mean('time').values, coord=cre.lon.values)
        c1 = ax.contourf(cyclic_lons, cre_sw.lat.values, cdata, add_colorbar=False, vmin=-80, vmax=80, cmap='RdBu_r', transform=ccrs.PlateCarree(central_longitude=lon), levels=np.linspace(-80,80,17), extend='both')
        c2 = ax.contour( cyclic_lons, cre_sw.lat.values, cdata, add_colorbar=False, vmin=-80, vmax=80, transform=ccrs.PlateCarree(central_longitude=lon), levels=np.linspace(-80,80,17), colors='k', linewidths=0.2, linestyles='solid',alpha=0.2, extend='both')
        ax.set_title('')
        ax.coastlines(linewidth=0.6)
        t1 = ax.text(-178,20,'$C^{sw}$ = '+str(np.round(table1_dict['shortwave cre'],1)), color='k',fontsize=size)
        t1.set_bbox(dict(facecolor='white', alpha=0.95, linewidth=0,pad=1.))
        ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
        ax.tick_params(axis='x', colors='silver', width=0)
        ax.set_xticklabels(['','','','','','',''],color='k', fontsize='x-small')

        ax = axes[2]
        cdata, cyclic_lons = add_cyclic_point(cre_lw.mean('time').values, coord=cre.lon.values)
        c1 = ax.contourf(cyclic_lons, cre_lw.lat.values, cdata, add_colorbar=False, vmin=-80, vmax=80, cmap='RdBu_r', transform=ccrs.PlateCarree(central_longitude=lon), levels=np.linspace(-80,80,17), extend='both')
        ax.set_title('')
        ax.coastlines(linewidth=0.6)
        t1 = ax.text(-178,20,'$C^{lw}$ = '+str(np.round(table1_dict['longwave cre'],1)), color='k',fontsize=size)
        t1.set_bbox(dict(facecolor='white', alpha=0.95, linewidth=0,pad=1.))
        ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
        ax.tick_params(axis='x', colors='silver', width=0)
        ax.set_xticklabels([r'$0^\circ$',r'$60^\circ$ E', r'$120^\circ$ E', r'$180^\circ$', r'$120^\circ$ W', r'$60^\circ$ W', r'$0^\circ$'],color='k', fontsize='x-small')

        for ax in axes:
            ax.tick_params(axis='both', direction='in', top=True, right=True, left=True, bottom=True)
            # ax.outline_patch.set_edgecolor('silver')
            ax.spines['geo'].set_edgecolor('silver')
            ax.set_yticks([-15, 0, 15])
            ax.tick_params(axis='y', colors='silver', width=0)
            ax.set_yticklabels([r'$15^\circ$ S', r'$0^\circ$', r'$15^\circ$ N'],color='k', fontsize='x-small')
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(1.5)


        plt.colorbar(c1, cax=cax, orientation='horizontal', label='Cloud Radiative Effects / Wm$^{-2}$', ticks=np.linspace(-80,80,9), extend='neither')

        %%time
        ceres = cp.load_ceres()
        #####################################################################
        ##################### PLOT DIFFERENCE CRE (Figure 2c) ###########################
        #####################################################################
        fig, axes, cax = fc(3,1, width=4, aspect=60/180, bottom_pad=-0.5, cbar_mode='single', internal_pad = -0.53, cbar_pad=0, cbar_location = 'bottom', cbar_short_side_pad=0, axes_kwargs={'projection': ccrs.PlateCarree(central_longitude=180)})

        lon = 0
        ax = axes[0]
        cdata, cyclic_lons = add_cyclic_point((ceres.cre-cre).mean('time').values, coord=cre.lon.values)
        c1 = ax.contourf(cyclic_lons, cre.lat.values, cdata, add_colorbar=False, vmin=-80, vmax=80, cmap='RdBu_r', transform=ccrs.PlateCarree(central_longitude=lon), levels=np.linspace(-80,80,17), extend='both')
        c2 = ax.contour( cyclic_lons, cre.lat.values, cdata, add_colorbar=False, vmin=-80, vmax=80, transform=ccrs.PlateCarree(central_longitude=lon), levels=np.linspace(-80,80,17), colors='k', linewidths=0.2, alpha=0.2, linestyles='solid', extend='both')
        ax.set_title('')
        ax.coastlines(linewidth=0.6)
        ax.text(-178,20,'$C$', color='k', fontsize='small')
        ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
        ax.tick_params(axis='x', colors='silver', width=0)
        ax.set_xticklabels(['','','','','','',''],color='k', fontsize='x-small')


        ax = axes[1]
        cdata, cyclic_lons = add_cyclic_point((ceres.cre_sw-cre_sw).mean('time').values, coord=cre.lon.values)
        c1 = ax.contourf(cyclic_lons, cre_sw.lat.values, cdata, add_colorbar=False, vmin=-80, vmax=80, cmap='RdBu_r', transform=ccrs.PlateCarree(central_longitude=lon), levels=np.linspace(-80,80,17), extend='both')
        c2 = ax.contour( cyclic_lons, cre_sw.lat.values, cdata, add_colorbar=False, vmin=-80, vmax=80, transform=ccrs.PlateCarree(central_longitude=lon), levels=np.linspace(-80,80,17), colors='k', linewidths=0.2, linestyles='solid',alpha=0.2, extend='both')
        ax.set_title('')
        ax.coastlines(linewidth=0.6)
        ax.text(-178,20,'$C^{sw}$', color='k', fontsize='small')
        ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
        ax.tick_params(axis='x', colors='silver' ,width=0)
        ax.set_xticklabels(['','','','','','',''],color='k', fontsize='x-small')

        ax = axes[2]
        cdata, cyclic_lons = add_cyclic_point((ceres.cre_lw-cre_lw).mean('time').values, coord=cre.lon.values)
        c1 = ax.contourf(cyclic_lons, cre_lw.lat.values, cdata, add_colorbar=False, vmin=-80, vmax=80, cmap='RdBu_r', transform=ccrs.PlateCarree(central_longitude=lon), levels=np.linspace(-80,80,17), extend='both')
        c2 = ax.contour( cyclic_lons, cre_lw.lat.values, cdata, add_colorbar=False, vmin=-80, vmax=80, transform=ccrs.PlateCarree(central_longitude=lon), levels=np.linspace(-80,80,17), colors='k', linewidths=0.2, linestyles='solid', alpha=0.3, extend='both')
        ax.set_title('')
        ax.coastlines(linewidth=0.6)
        ax.text(-178,20,'$C^{lw}$', color='k', fontsize='small')
        ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
        ax.tick_params(axis='x', colors='silver', width=0)
        ax.set_xticklabels([r'$0^\circ$',r'$60^\circ$ E', r'$120^\circ$ E', r'$180^\circ$', r'$120^\circ$ W', r'$60^\circ$ W', r'$0^\circ$'],color='k', fontsize='x-small')


        for ax in axes:
            ax.tick_params(axis='both', direction='in', top=True, right=True, left=True, bottom=True)
            ax.spines['geo'].set_edgecolor('silver')
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(1.5)
                ax.spines[axis].set_color('silver')

        plt.colorbar(c1, cax=cax, orientation='horizontal', label='Cloud Radiative Effects / Wm$^{-2}$', ticks=np.linspace(-80,80,9), extend='neither')

#####################################################################
##################### PLOT CLOUD FRACTION (Figure 3a,b) ###########################
#####################################################################

from cartopy.util import add_cyclic_point
import cartopy.util as cutil


cdata, cyclic_lons = add_cyclic_point(fh.mean('time').values, coord=fh.lon.values)
fig, axes, cax = fc(2,1, width=4, aspect=60/180, bottom_pad=-0.5, cbar_mode='single', internal_pad = -0.53, cbar_pad=-0.2, cbar_location = 'bottom', cbar_short_side_pad=0, axes_kwargs={'projection': ccrs.PlateCarree(central_longitude=180)})
ax = axes[0]
c1 = ax.contourf(cyclic_lons, fh.lat.values, cdata, add_colorbar=False, vmin=0, vmax=0.5, cmap='Blues', transform=ccrs.PlateCarree(central_longitude=0), levels=np.linspace(0,0.5,11), extend='both')
c2 = ax.contour( cyclic_lons, fh.lat.values, cdata, add_colorbar=False, vmin=0, vmax=0.5, transform=ccrs.PlateCarree(central_longitude=0), levels=np.linspace(0,0.5,11), colors='navy', linewidths=0.2, alpha=0.3, extend='both')
ax.set_title('')
ax.coastlines(linewidth=0.6)
t1 = ax.text(-178,20,'$f_h$ = '+str(np.round(table1_dict['anvil area fraction'],2)), color='k', fontsize='small')
t1.set_bbox(dict(facecolor='white', alpha=0.95, linewidth=0,pad=1.))
ax.add_patch(mpatches.Rectangle(xy=[80, -15], width=70, height=30, facecolor=None, edgecolor='k',
                                    alpha=1, fill=False,linewidth=1,transform=ccrs.PlateCarree()))

cdata, cyclic_lons = add_cyclic_point(fl.mean('time').values, coord=fl.lon.values)
ax = axes[1]
c1 = ax.contourf(cyclic_lons, fl.lat.values, cdata, add_colorbar=False, vmin=0, vmax=0.5, cmap='Blues', transform=ccrs.PlateCarree(central_longitude=0), levels=np.linspace(0,0.5,11), extend='both')
c2 = ax.contour( cyclic_lons, fl.lat.values, cdata, add_colorbar=False, vmin=0, vmax=0.5, transform=ccrs.PlateCarree(central_longitude=0), levels=np.linspace(0,0.5,11), colors='navy', linewidths=0.2, alpha=0.3,extend='both')
ax.set_title('')
ax.coastlines(linewidth=0.6)
t1 = ax.text(-178,20,'$f_\ell$ = '+str(np.round(table1_dict['low cloud fraction'],2)), color='k', fontsize='small')
t1.set_bbox(dict(facecolor='white', alpha=0.95, linewidth=0,pad=1.))
ax.add_patch(mpatches.Rectangle(xy=[-115, -29], width=40, height=20, facecolor=None, edgecolor='k',
                                    alpha=1, fill=False,linewidth=1,transform=ccrs.PlateCarree()))
ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
ax.tick_params(axis='x', colors='silver' ,width=0)
ax.set_xticklabels(['','','','','','',''],color='k', fontsize='x-small')

for ax in axes:
    ax.tick_params(axis='both', direction='in', top=True, right=True, left=True, bottom=True)
    # ax.outline_patch.set_edgecolor('silver')
    ax.spines['geo'].set_edgecolor('silver')
    # ax.outline_patch.set_edg
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
    ax.set_yticks([-15, 0, 15])
    ax.tick_params(axis='y', colors='silver', width=0)
    ax.set_yticklabels([r'$15^\circ$ S', r'$0^\circ$', r'$15^\circ$ N'],color='k', fontsize='x-small')

plt.colorbar(c1, cax=cax, orientation='horizontal', label='Cloud Top Fraction', ticks=np.linspace(0,0.5,6), extend='both')

# save results for later
cre_mean = cp.area_weighted_mean(cre).mean().values
cre_l_mean = cp.area_weighted_mean(cre_l).mean().values
m_lh_mean = cp.area_weighted_mean(m_lh).mean().values
cre_h_sw_mean = cp.area_weighted_mean(cre_h_sw).mean().values
cre_h_lw_mean = cp.area_weighted_mean(cre_h_lw).mean().values
cre_h_mean = cp.area_weighted_mean(cre_h).mean().values

sig_cre = cre.std().values
sig_cre_l = cre_l.std().values
sig_cre_lh = m_lh.std().values
sig_cre_h_sw = cre_h_sw.std().values
sig_cre_h_lw = cre_h_lw.std().values
sig_cre_h = cre_h.std().values

%%time
#####################################################################
######################## PLOT CLIMATOLOGY (Figure 3d-h) ###########################
#####################################################################
vmin=-80
vmax=80
num_levels = 17
fig, axes, cax = fc(6,1, width=4, aspect=60/180, bottom_pad=-0.5, cbar_mode='single', internal_pad = -0.53, cbar_pad=0, cbar_location = 'bottom', cbar_short_side_pad=0, axes_kwargs={'projection': ccrs.PlateCarree(central_longitude=180)})

lon = 0
ax = axes[0]
cdata, cyclic_lons = add_cyclic_point(cre.mean('time').values, coord=cre.lon.values)
c1 = ax.contourf(cyclic_lons, cre.lat.values, cdata, add_colorbar=False, vmin=vmin, vmax=vmax, cmap='RdBu_r', transform=ccrs.PlateCarree(central_longitude=lon), levels=np.linspace(vmin,vmax,num_levels),extend='both')
c2 = ax.contour(cyclic_lons, cre.lat.values, cdata, add_colorbar=False, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(central_longitude=lon), levels=np.linspace(vmin,vmax,num_levels), colors='k', linewidths=0.2, alpha=0.3, extend='both')
ax.set_title('')
ax.coastlines(linewidth=0.6)
t1 = ax.text(-178,20,'$C$ = '+str(int(np.round(cre_mean)))+', $\sigma$ = '+str(int(np.round(sig_cre))), color='k', fontsize='small')
t1.set_bbox(dict(facecolor='white', alpha=.95, linewidth=0,pad=1.))

ax = axes[1]
cdata, cyclic_lons = add_cyclic_point(cre_l.mean('time').values, coord=cre.lon.values)
c1 = ax.contourf(cyclic_lons, cre.lat.values, cdata, add_colorbar=False, vmin=vmin, vmax=vmax, cmap='RdBu_r', transform=ccrs.PlateCarree(central_longitude=lon), levels=np.linspace(vmin,vmax,num_levels), extend='both')
c2 = ax.contour(cyclic_lons, cre.lat.values, cdata, add_colorbar=False, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(central_longitude=lon), levels=np.linspace(vmin,vmax,num_levels), colors='k', linewidths=0.2, alpha=0.3, extend='both')
ax.set_title('')
ax.coastlines(linewidth=0.6)
t1 = ax.text(-178,20,'$C_\ell$ = '+str(int(np.round(cre_l_mean)))+', $\sigma$ = '+str(int(np.round(sig_cre_l))), color='k', fontsize='small')
t1.set_bbox(dict(facecolor='white', alpha=.95, linewidth=0,pad=1.))

ax = axes[2]
cdata, cyclic_lons = add_cyclic_point(m_lh.mean('time').values, coord=cre.lon.values)
c1 = ax.contourf(cyclic_lons, cre.lat.values, cdata, add_colorbar=False, vmin=vmin, vmax=vmax, cmap='RdBu_r', transform=ccrs.PlateCarree(central_longitude=lon), levels=np.linspace(vmin,vmax,num_levels), extend='both')
c2 = ax.contour(cyclic_lons, cre.lat.values, cdata, add_colorbar=False, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(central_longitude=lon), levels=np.linspace(vmin,vmax,num_levels), colors='k', linewidths=0.2, alpha=0.3, extend='both')
ax.set_title('')
ax.coastlines(linewidth=0.6)
t1 = ax.text(-178,20,'$m_{\ell h}$ = '+str(int(np.round(m_lh_mean)))+', $\sigma$ = '+str(int(np.round(sig_cre_lh))), color='k', fontsize='small')
t1.set_bbox(dict(facecolor='white', alpha=.95, linewidth=0,pad=1.))

ax = axes[3]
cdata, cyclic_lons = add_cyclic_point(cre_h_sw.mean('time').values, coord=cre.lon.values)
c1 = ax.contourf(cyclic_lons, cre.lat.values, cdata, add_colorbar=False, vmin=vmin, vmax=vmax, cmap='RdBu_r', transform=ccrs.PlateCarree(central_longitude=lon), levels=np.linspace(vmin,vmax,num_levels), extend='both')
c2 = ax.contour(cyclic_lons, cre.lat.values, cdata, add_colorbar=False, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(central_longitude=lon), levels=np.linspace(vmin,vmax,num_levels), colors='k', linewidths=0.2, alpha=0.3, extend='both')
ax.set_title('')
ax.coastlines(linewidth=0.6)
t1 = ax.text(-178,20,'$C_h^{sw}$ = '+str(int(np.round(cre_h_sw_mean)))+', $\sigma$ = '+str(int(np.round(sig_cre_h_sw))), color='k', fontsize='small')
t1.set_bbox(dict(facecolor='white', alpha=.95, linewidth=0,pad=1.))

ax = axes[4]
cdata, cyclic_lons = add_cyclic_point(cre_h_lw.mean('time').values, coord=cre.lon.values)
c1 = ax.contourf(cyclic_lons, cre.lat.values, cdata, add_colorbar=False, vmin=vmin, vmax=vmax, cmap='RdBu_r', transform=ccrs.PlateCarree(central_longitude=lon), levels=np.linspace(vmin,vmax,num_levels), extend='both')
c2 = ax.contour(cyclic_lons, cre.lat.values, cdata, add_colorbar=False, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(central_longitude=lon), levels=np.linspace(vmin,vmax,num_levels), colors='k', linewidths=0.2, alpha=0.3, extend='both')
ax.set_title('')
ax.coastlines(linewidth=0.6)
t1 = ax.text(-178,20,'$C_h^{lw}$ = '+str(int(np.round(cre_h_lw_mean)))+', $\sigma$ = '+str(int(np.round(sig_cre_h_lw))), color='k', fontsize='small')
t1.set_bbox(dict(facecolor='white', alpha=.95, linewidth=0,pad=1.))

ax = axes[5]
cdata, cyclic_lons = add_cyclic_point(cre_h.mean('time').values, coord=cre.lon.values)
c1 = ax.contourf(cyclic_lons, cre.lat.values, cdata, add_colorbar=False, vmin=vmin, vmax=vmax, cmap='RdBu_r', transform=ccrs.PlateCarree(central_longitude=lon), levels=np.linspace(vmin,vmax,num_levels), extend='both')
c2 = ax.contour(cyclic_lons, cre.lat.values, cdata, add_colorbar=False, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(central_longitude=lon), levels=np.linspace(vmin,vmax,num_levels), colors='k', linewidths=0.2, alpha=0.3, extend='both')
ax.set_title('')
ax.coastlines(linewidth=0.6)
t1 = ax.text(-178,20,'$C_h$ = '+str(int(np.round(cre_h_mean)))+', $\sigma$ = '+str(int(np.round(sig_cre_h))), color='k', fontsize='small')
t1.set_bbox(dict(facecolor='white', alpha=.95, linewidth=0,pad=1.))
ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
ax.tick_params(axis='x', colors='silver', width=0)
ax.set_xticklabels([r'$0^\circ$',r'$60^\circ$ E', r'$120^\circ$ E', r'$180^\circ$', r'$120^\circ$ W', r'$60^\circ$ W', r'$0^\circ$'],color='k', fontsize='x-small')


for ax in axes:
    ax.tick_params(axis='both', direction='in', top=True, right=True, left=True, bottom=True)
    ax.spines['geo'].set_edgecolor('silver')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
    ax.set_yticks([-15, 0, 15])
    ax.tick_params(axis='y', colors='silver', width=0)
    ax.set_yticklabels([r'$15^\circ$ S', r'$0^\circ$', r'$15^\circ$ N'],color='k', fontsize='x-small')


plt.colorbar(c1, cax=cax, orientation='horizontal', label='Cloud Radiative Effects / Wm$^{-2}$', ticks=np.linspace(vmin,vmax,9), extend='neither')


# load in cloud fraction again
fh = xr.open_mfdataset('/data/bmckim/identified_cloud_fraction/fh_max_0pt3.nc').fh*n
fh_log = np.log(fh)
fh_log_weighted = cp.area_weighted_mean(fh_log)
fh_log_annual = cp.group_data(fh_log_weighted)

%%time
#####################################################################
################# HIGH CLOUD FRACTION vs Ts (Figure 4a) #########################
#####################################################################
width=2.
fig, axes = fc(1, 1, width=width, aspect=1, internal_pad=0)
ax = axes[0]
ppf.make_pretty_plot(ax, xmin=297.4, xmax=298.6, ymin=-2.28+0.57, ymax=-2.14+0.57, xlabel=r'$T_s$ / K',ylabel=r'$\ln f_h$',
                 control_ticks=True,delxmaj=0.5,delxmin=0.1,delymaj=0.05,delymin=0.01)

foo = sp.stats.linregress(ts_annual, fh_log_annual)
m, b, r, p, std_err = sp.stats.linregress(ts_annual, fh_log_annual)
intercept_stderr = foo.intercept_stderr

print('m = '+str(m))
print('std_err = '+str(std_err))

xn = np.linspace(np.min(ts_annual)-np.min(ts_annual)/1000,np.max(ts_annual)+np.max(ts_annual)/1000,100)
yn = np.polyval([m, b], xn)
ynlow = np.polyval([m+std_err, b-intercept_stderr], xn)
ynhigh = np.polyval([m-std_err, b+intercept_stderr], xn)

ax.text(298.6, -2.145+0.565, '$d \ln f_h/dT_s = -$'+str(int(np.abs(np.round(m,2)*100)))+r'$\pm$'+str(int(np.round(std_err,2)*100))+r'\% K$^{-1}$', fontsize='small', horizontalalignment='right', color='darkred')
ax.text(298.6, -2.157+0.565, '$r = -$'+str("{:.2}".format(np.abs(np.round(r,4)))), fontsize='small', horizontalalignment='right', color='darkred')

ax.scatter(ts_annual, fh_log_annual, color='darkred', s=10, zorder=2,alpha=1, facecolor='lightcoral', linewidth=0.5)
ax.plot(xn, yn, color='darkred', linestyle='solid', linewidth=0.95, dash_capstyle='round', zorder=2)
ax.fill_between(xn, ynlow, ynhigh, color='lightcoral', alpha=0.3, edgecolor='white')

plt.savefig('figure.svg', format='svg')
display(SVG(filename='figure.svg'))

# calculate albedos again
n = 1.648
high_albedos, low_albedos = cp.albedo_variabilities(n,cf_c='3',method='max')

#####################################################################
################### PLOT Albedo Variability (Figure 4b) #########################
#####################################################################
width=2.
fig, axes = fc(1, 1, width=width, aspect=1, internal_pad=0)
ax = axes[0]
ppf.make_pretty_plot(ax, xmin=297.4, xmax=298.6, ymin=np.log(0.4), ymax=np.log(0.65), xlabel=r'$T_s$ / K',ylabel=r'$\ln \alpha$',
                 control_ticks=True,delxmaj=0.5,delxmin=0.1,delymaj=0.1,delymin=0.02)

# get rid of 2015-2016 El Nino
ts_annual_reduced = ts_annual[:-1]
high_albedos_reduced = high_albedos[:-1]
low_albedos_reduced = low_albedos[:-1]

foo = sp.stats.linregress(ts_annual_reduced, np.log(high_albedos_reduced))
m, b, r, p, std_err = sp.stats.linregress(ts_annual_reduced, np.log(high_albedos_reduced))
intercept_stderr = foo.intercept_stderr
xn = np.linspace(np.min(ts_annual)-np.min(ts_annual)/1000,np.max(ts_annual)+np.max(ts_annual)/1000,100)
yn = np.polyval([m, b], xn)
ynlow = np.polyval([m+std_err, b-intercept_stderr], xn)
ynhigh = np.polyval([m-std_err, b+intercept_stderr], xn)
ax.scatter(ts_annual, np.log(high_albedos), color='darkred', s=10, zorder=2,alpha=1, facecolor='lightcoral', linewidth=0.5)
ax.plot(xn, yn, color='darkred', linestyle='solid', linewidth=0.95, dash_capstyle='round', zorder=2)
ax.fill_between(xn, ynlow, ynhigh, color='lightcoral', alpha=0.3, edgecolor='white')
ax.text(297.47, -0.46, r'$d \ln \alpha_h / dT_s = $ '+str(int(np.round(m,2)*100))+r'$\pm$'+str(int(np.round(std_err,2)*100))+r'\% K$^{-1}$', fontsize='small', horizontalalignment='left', color='darkred')
ax.text(297.47, -0.52, '$r = $ '+str("{:.2}".format(np.round(r,4))), fontsize='small', horizontalalignment='left', color='darkred')
print('m = '+str(m))
print('std_err = '+str(std_err))
print('r = '+str(r))
