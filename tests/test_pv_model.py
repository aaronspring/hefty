import pytest
# from pandas.testing import assert_frame_equal
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pvlib

from hefty.pv_model import model_pv_power

def test_model_pv_power():
    # based on https://github.com/williamhobbs/pv-system-model/blob/main/quick_demo.ipynb
    latitude = 33.5
    longitude = -86.8
    tz = 'UTC'
    freq = '5min'
    times = pd.date_range('2020-12-21 18:00', '2020-12-21 18:20', freq=freq,
                          tz=tz)
    loc = pvlib.location.Location(latitude, longitude, tz)
    # model clear sky
    cs = loc.get_clearsky(times)
    # add temperature, wind, albedo
    cs['temp_air'] = 10
    cs['wind_speed'] = 5
    cs['albedo'] = 0.15

    # drop dhi to test dhi calc
    cs = cs.drop(columns=['dhi'])

    # mock horizon profile from
    # https://pvlib-python.readthedocs.io/en/stable/gallery/shading/plot_simple_irradiance_adjustment_for_horizon_shading.html
    hp = pd.Series([
        10.7, 11.8, 11.5, 10.3, 8.0, 6.5, 3.8, 2.3, 2.3, 2.3, 4.6, 8.0, 10.3,
        11.1, 10.7, 10.3, 9.2, 6.1, 5.3, 2.3, 3.1, 1.9, 1.9, 2.7, 3.8, 5.3,
        6.5, 8.4, 8.8, 8.4, 8.4, 8.4, 6.5, 6.1, 6.5, 6.1, 7.3, 9.2, 8.4, 8.0,
        5.7, 5.3, 5.3, 4.2, 4.2, 4.2, 7.3, 9.5], index=np.arange(0, 360, 7.5))

    # define plants
    plant_dict = {
        # 'latitude': [32.8, 31.6, 32.1],
        # 'longitude': [-83.6, -84.2, -81.1],
        'mount_type': ['single-axis', 'single-axis', 'fixed'],
        'max_tracker_angle': [60, 45, None],
        'axis_azimuth': [180, 180, None],
        'axis_tilt': [0, 0, None],
        'fixed_tilt': [None, None, 20],
        'fixed_azimuth': [None, None, 180],
        'gcr': [0.45, 0.35, 0.6],
        'nameplate_dc': [120, 120, 120],
        'nameplate_ac': [100, 100, 100],
        'cell_type': ['thin-film_cdte', 'crystalline_half-cut', 'crystalline'],
        'gamma_pdc': [-0.0025, -0.0035, -0.0035],
        'backtrack': [False, True, None],
        'bifacial': [False, True, True],
        'dc_loss_fraction': [0.15, 0.15, 0.15],
        'n_cells_up': [1, 24, 12],
        'row_side_num_mods': [4, 1, 2],
        'shade_loss_model': ['linear', 'non-linear_simple_twin_module', 'non-linear_simple'],
        'cross_axis_slope': [0, 1, 0],
        'slope_aware_backtracking': [False, True, None],
        'programmed_cross_axis_slope': [None, 1, None],
        'transient_cell_temp': [False, False, True],
        'k': [None, None, 0.01],
        'cap_adjustment': [None, None, True],
        'horizon_profile': [None, None, hp],
        'default_site_transposition_model': ['haydavies', 'haydavies', 'perez']
        }
    # make a dataframe
    plants_df = pd.DataFrame(data=plant_dict)
    num_plants = len(plants_df)

    # model
    power_clear_sky = {}
    for plant in range(num_plants):
        print(plant+1)
        # convert row of dataframe to a dictionary
        plant_data = plants_df.loc[plant].dropna().to_dict() 
        power_clear_sky[plant], _ = model_pv_power(
            cs, latitude=latitude, longitude=longitude, **plant_data)
    expected_0 = [52.57427667, 52.67423954, 52.80234901, 52.9574905,
                  53.13831252]
    expected_1 = [57.7304743, 57.80085983, 57.8910615, 58.00032732,
                  58.12774744]
    expected_2 = [81.97457884, 78.63431208, 78.38870274, 78.08727182,
                  77.73002612]
    assert_allclose(power_clear_sky[0].values, expected_0)
    assert_allclose(power_clear_sky[1].values, expected_1)
    assert_allclose(power_clear_sky[2].values, expected_2)
