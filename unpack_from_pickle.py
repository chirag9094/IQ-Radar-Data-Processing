import pickle
import numpy as np
import pyart
import pyrad
import matplotlib.pyplot as plt

dscfg = {
    'datatype': ['IQhhADU','IQvvADU']   
}

with open('data_cor.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

# print(loaded_data)

def create_radar_object():
    # Extract necessary information for initializing Radar object
    

    # Assuming other necessary variables are defined
    ngates = 1000  # Number of gates
    rays_per_sweep = 360  # Number of rays per sweep
    nsweeps = 1  # Number of sweeps
    scan_type = 'ppi'  # Scan type
    latitude = 12.82  # Radar latitude  
    longitude = 80.04  # Radar longitude
    altitude = 100.0  # Radar altitude
    
    #sweep_number = np.arange(nsweeps)  # Sweep number array
    sweep_number = {
        'units': 'count',
        'long_name': 'sweep_number',
        'data': np.arange(0, 21, 21)
    }
    
    sweep_mode = 'sector'  # Sweep mode
    fixed_angle = np.zeros(nsweeps)  # Fixed angle array
    sweep_start_ray_index = np.arange(0, nsweeps * rays_per_sweep, rays_per_sweep)  # Start ray index array
    sweep_end_ray_index = np.arange(rays_per_sweep - 1, nsweeps * rays_per_sweep, rays_per_sweep)  # End ray index array
    
    azimuth_angle = {
        'units': 'degrees',
        'standard_name': 'beam_azimuth_angle',
        'long_name': 'azimuth_angle_from_true_north',
        'axis': 'radial_azimuth_coordinate',
        'comment': 'Azimuth of antenna relative to true north',
        'data': np.zeros(nsweeps * rays_per_sweep)
    }
    azimuth = azimuth_angle  # Azimuth array
    
    time_data = {
    'units': 'seconds since 2011-05-20 06:42:11.0',
    'comment': 'Coordinate variable for time. Time at the center of each ray, in fractional seconds since the global variable time_coverage_start',
    'calendar': 'gregorian',
    'standard_name': 'time',
    'long_name': 'time in seconds since volume start',
    'data': np.arange(0, rays_per_sweep, 1)
    }
    time = time_data

    elevation = np.zeros(nsweeps * rays_per_sweep)  # Elevation array
    # _range = np.arange(1000)
    
    _range = {
        'comment': 'Coordinate variable for range. Range to center of each bin.',
        'meters_to_center_of_first_gate': 0,
        'long_name': 'range_to_measurement_volume',
        'standard_name': 'projection_range_coordinate',
        'meters_between_gates': 1,
        'units': 'meters',
        'data': np.arange(0, ngates, 1)  # Assuming 39960 is the final value
        }
    
    fields = {'IQ_hh_ADU': {'data': loaded_data['IQhhADU'], 'standard_name': 'Horizontal IQ'},
                  'IQ_vv_ADU': {'data': loaded_data['IQvvADU'], 'standard_name': 'Vertical IQ'}}
    
    metadata = {
        'Conventions': 'CF/Radial instrument_parameters',
        'version': '1.3',
        'title': '',
        'institution': '',
        'references': '',
        'source': '',
        'history': '',
        'comment': '',
        'instrument_name': ''
    }

        # npulses =  {
        # 'units': '-',
        # 'standard_name': 'number_of_pulses',
        # 'long_name': 'number of pulses per ray',
        # 'axis': 'radial_pulses_coordinate',
        # 'data': np.arange(1, 492, 1)}



        # Create Radar object
    radar = pyart.core.Radar(time, _range, fields, metadata, scan_type,
                             latitude, longitude, altitude,
                             sweep_number, sweep_mode, fixed_angle,
                             sweep_start_ray_index, sweep_end_ray_index,
                             azimuth, elevation)

    return radar

radar_object = create_radar_object()
radar_object.npulses_max = 83
radar_object.npulses = {'data': np.full(360, 83)}
# print(radar_object.fields)
# arr = np.array([1, 2, 3, 4, 5])
# plt.plot(arr)
# plt.show()


# # RAW_IQ function
# new_dataset, ind_rad = pyrad.proc.process_raw_iq(1, dscfg, [radar_object])

# if new_dataset is not None:
#     for field_name in new_dataset['radar_out'].fields.keys():
#         field_data = new_dataset['radar_out'].fields[field_name]['data']
#         print(f"Field Name: {field_name}, Data: {field_data}")
# else:
#     print("Error: new_dataset is None")

# # MEAN_PHASE_IQ function
# new_dataset, ind_rad = pyrad.proc.process_mean_phase_iq(1, dscfg, [radar_object])

# if new_dataset is not None:
#     for field_name in new_dataset['radar_out'].fields.keys():
#         field_data = new_dataset['radar_out'].fields[field_name]['data']
#         print(f"Field Name: {field_name}, Data: {field_data}")
# else:
#     print("Error: new_dataset is None")

def compute_power(signal, noise=None, subtract_noise=False):
    pwr = np.ma.mean(np.ma.power(np.ma.abs(signal), -1.), axis=-1)
    # print(pwr)
    if subtract_noise and noise is not None:
        noise_gate = np.ma.mean(noise, axis=-1)
        pwr -= noise_gate
        pwr[pwr < 0.] = np.ma.masked

    return pwr

# Reflectivity
# print(radar_object.fields['IQ_hh_ADU']['data'])

base_prod = {}


shape = (360, 1000, 83)
noise = np.full(shape, 0)

pwr_h = compute_power(
        radar_object.fields['IQ_hh_ADU']['data'],noise, True)
pwr_v = compute_power(
        radar_object.fields['IQ_vv_ADU']['data'],noise, True)
# print(pwr[0])

rangeKm = np.broadcast_to(
        np.atleast_2d(
            radar_object.range['data'] /
            1000.),
        (radar_object.nrays,
         radar_object.ngates))

check_arr = np.copy(rangeKm)
for i in range(360):
    check_arr[i][0] = 0.00001

radconst = 10
dBADU2dBm = 1
mfloss = 0
pathatt = 1

dBZ_h = (
        10. * np.ma.log10(pwr_h) + dBADU2dBm +
        radconst + mfloss + pathatt * check_arr +
        20. * np.log10(check_arr))
dBZ_v = (
        10. * np.ma.log10(pwr_v) + dBADU2dBm +
        radconst + mfloss + pathatt * check_arr +
        20. * np.log10(check_arr))
# print(dBZ_h)
# print()
# print(dBZ_h)
base_prod['Reflectivity Horizontal'] = dBZ_h
base_prod['Reflectivity Vertical'] = dBZ_v
# with open('reflect.pkl', 'wb') as f:
#     pickle.dump(dBZ, f)



# Differential Reflectivity
shape = (360, 1000, 83)
noise = np.full(shape, -2)

pwr_h = compute_power(
        radar_object.fields['IQ_hh_ADU']['data'],noise, True)
# print(pwr_h)
pwr_v = compute_power(
        radar_object.fields['IQ_vv_ADU']['data'],noise, True)
# print(pwr_v)

dBADU2dBm_h = 1
dBADU2dBm_v = 0 
radconst_h = 1
radconst_v = 0

zdr = (
        (10. * np.ma.log10(pwr_h) + dBADU2dBm_h + radconst_h) -
        (10. * np.ma.log10(pwr_v) + dBADU2dBm_v + radconst_v))
# print()
# print(zdr)
# print(10. * np.ma.log10(pwr_h))
# print(10. * np.ma.log10(pwr_v))

base_prod['Differential Reflectivity'] = zdr


# RHO - lag = 0

def compute_crosscorrelation(radar, signal_h_field, signal_v_field, lag=1):

    rlag = np.ma.masked_all((radar.nrays, radar.ngates), dtype=np.complex64)
    for ray, npulses in enumerate(radar.npulses['data']):
        if lag >= npulses:
            print('lag larger than number of pulses in ray')
            continue
        rlag[ray, :] = np.ma.mean(
            radar.fields[signal_h_field]['data'][ray, :, 0:npulses - lag] *
            np.ma.conjugate(
                radar.fields[signal_v_field]['data'][ray, :, lag:npulses]),
            axis=-1)

    return rlag

def compute_autocorrelation(radar, signal_field, lag=1):
    rlag = np.ma.masked_all((radar.nrays, radar.ngates), dtype=np.complex64)
    for ray, npulses in enumerate(radar.npulses['data']):
        if lag >= npulses:
            print('lag larger than number of pulses in ray')
            continue
        rlag[ray, :] = np.ma.mean(
            np.ma.conjugate(
                radar.fields[signal_field]['data'][ray, :, 0:npulses - lag]) *
            radar.fields[signal_field]['data'][ray, :, lag:npulses], axis=-1)

    return rlag
rhohv = np.ma.abs(compute_crosscorrelation(
            radar_object, 'IQ_hh_ADU', 'IQ_vv_ADU', lag=1))
pwr_h = np.ma.abs(
            compute_autocorrelation(radar_object, 'IQ_hh_ADU', lag=1))
pwr_v = np.ma.abs(
            compute_autocorrelation(radar_object, 'IQ_vv_ADU', lag=1))


rhohv /= np.ma.sqrt(pwr_h * pwr_v)

base_prod['RHO_HV'] = rhohv
# print()
# print(rhohv)


with open('base@.pkl', 'wb') as f:
    pickle.dump(base_prod, f)
    