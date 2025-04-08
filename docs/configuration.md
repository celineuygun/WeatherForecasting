### WRF and WPS Configuration Files

**`namelist.wps`**: This file contains the WPS configuration settings, including the domains, time window, and other simulation parameters. Make sure to update the `start_date`, `end_date`, and domain configurations according to your desired simulation setup. `geog_data_path` should point to the location where your `WPS_GEOG` data is stored.

```fortran
&share
 wrf_core = 'ARW',
 max_dom = 2,
 start_date = 'YYYY-MM-DD_HH:MM:SS', 'YYYY-MM-DD_HH:MM:SS',
 end_date   = 'YYYY-MM-DD_HH:MM:SS', 'YYYY-MM-DD_HH:MM:SS',
 interval_seconds = 10800
/

&geogrid
 parent_id         =   1,  1,
 parent_grid_ratio =   1,  3,
 i_parent_start    =   1,  30,
 j_parent_start    =   1,  30,
 e_we              =  100, 106,
 e_sn              =  100, 106,
 dx = 9000,
 dy = 9000,
 map_proj = 'lambert',
 ref_lat   =  42.0,
 ref_lon   =   9.0,
 truelat1  =  30.0,
 truelat2  =  60.0,
 stand_lon =   9.0,
 geog_data_res = 'default','default',
 geog_data_path = '/path/to/WPS_GEOG'
/

&ungrib
 out_format = 'WPS',
 prefix = 'FILE',
/

&metgrid
 fg_name = 'FILE'
/
```

**`namelist.input`**: This file contains the input configuration for the WRF model. It includes settings for the simulation period, number of domains, and other parameters. The `max_dom`, `start_date`, and `end_date` parameters should match the ones in `namelist.wps`.

```fortran
&time_control
 run_days                            = 1,
 run_hours                           = 0,
 start_year                          = YYYY, YYYY, 
 start_month                         = MM, MM, 
 start_day                           = DD, DD, 
 start_hour                          = HH, HH, 
 end_year                            = YYYY, YYYY, 
 end_month                           = MM, MM, 
 end_day                             = DD, DD, 
 end_hour                            = HH, HH, 
 interval_seconds                    = 10800,
 input_from_file                     = .true., .true.,
 history_interval                    = 60, 60,
 frames_per_outfile                  = 1, 1,
 restart                             = .false.,
 restart_interval                    = 7200,
 io_form_history                     = 2,
 io_form_restart                     = 2,
 io_form_input                       = 2,
 io_form_boundary                    = 2
/

&domains
 time_step                           = 54,
 max_dom                             = 2,
 e_we                                = 100,106,
 e_sn                                = 100,106,
 e_vert                              = 45, 45,
 p_top_requested                     = 5000,
 num_metgrid_levels                  = 34,
 num_metgrid_soil_levels             = 4,
 dx                                  = 9000,9000,
 dy                                  = 9000,9000,
 grid_id                             = 1,2,
 parent_id                           = 1,1,
 i_parent_start                      = 1,30,
 j_parent_start                      = 1,30,
 parent_grid_ratio                   = 1,3,
 parent_time_step_ratio              = 1,3,
 feedback                            = 1,
 smooth_option                       = 0
/

&physics
 physics_suite                       = 'CONUS',
 mp_physics                          = -1, -1,
 ra_lw_physics                       = -1, -1,
 ra_sw_physics                       = -1, -1,
 bl_pbl_physics                      = -1, -1,
 cu_physics                          = -1, -1,
 sf_sfclay_physics                   = -1, -1,
 sf_surface_physics                  = -1, -1,
 radt                                = 15, 15,
 bldt                                = 0, 0,
 cudt                                = 0, 0,
 num_land_cat                        = 21,
 fractional_seaice                   = 1,
 sf_urban_physics                    = 0, 0
/

&fdda
/

&dynamics
 w_damping                           = 0,
 diff_opt                            = 2, 2,
 km_opt                              = 4, 4,
 diff_6th_opt                        = 0, 0,
 diff_6th_factor                     = 0.12, 0.12,
 base_temp                           = 290.,
 damp_opt                            = 3,
 zdamp                               = 5000., 5000.,
 dampcoef                            = 0.2, 0.2,
 khdif                               = 0, 0,
 kvdif                               = 0, 0,
 non_hydrostatic                     = .true., .true.,
 moist_adv_opt                       = 1, 1,
 scalar_adv_opt                      = 1, 1,
 gwd_opt                             = 1, 0
/

&bdy_control
 spec_bdy_width                      = 5,
 specified                           = .true.,
/

&grib2
/

&namelist_quilt
 nio_tasks_per_group                 = 1,
 nio_groups                          = 2,
/
```

### Setting Up Zsh for WRF and WPS

If you're using **Zsh** as your default shell, make sure to update your **`.zshrc`** file to include the following paths for proper execution of WRF and WPS. Replace the placeholder values (like the ones for `WRF_DIR`, `WPS_DIR`, or the Conda path) with the actual locations where WRF, WPS, and Conda are installed on your system.

```bash
export PATH="/opt/homebrew/bin:$PATH"
export MKL_SERVICE_FORCE_INTEL=1

# Homebrew and system paths
export CC=/opt/homebrew/bin/gcc-14
export CXX=/opt/homebrew/bin/g++-14
export FC=/opt/homebrew/bin/gfortran
export F77=/opt/homebrew/bin/gfortran
export MPIF90=/opt/homebrew/bin/mpif90
export PATH=/opt/homebrew/bin:$PATH

# NetCDF
export NETCDF=/opt/homebrew
export NETCDFINC=$NETCDF/include
export NETCDFLIB=$NETCDF/lib
export PATH=$NETCDF/bin:$PATH
export LD_LIBRARY_PATH=$NETCDF/lib:$LD_LIBRARY_PATH

# Jasper and libpng
export JASPERLIB=/opt/homebrew/opt/jasper/lib
export JASPERINC=/opt/homebrew/opt/jasper/include
export PNG_LIB=/opt/homebrew/opt/libpng/lib
export PNG_INC=/opt/homebrew/opt/libpng/include

export LDFLAGS="-L$NETCDF/lib -L$PNG_LIB -L$JASPERLIB $LDFLAGS"
export CPPFLAGS="-I$NETCDF/include -I$PNG_INC -I$JASPERINC $CPPFLAGS"

export WRF_DIR=/path/to/WRF
export WPS_DIR=/path/to/WPS

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/path/to/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/path/to/etc/profile.d/conda.sh" ]; then
        . "/path/to/etc/profile.d/conda.sh"
    else
        export PATH="/path/to/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

export PATH="$HOME/miniconda3/bin:$PATH"
```

Remember to source your .zshrc file to apply the changes:

```bash
source ~/.zshrc
```