import cdtools
from cdtools.tools import plotting as p
from matplotlib import pyplot as plt
import torch as t
import numpy as np
from helper_functions import compare_rpi_with_ptycho, savefig_plus_im

if __name__ == '__main__':

    shot_idx=70
    obj_size=400

    # For repeat 0
    calibration_cxi_file = 'e17965_1_00678.cxi'
    
    # For repeat 1
    rpi_cxi_file = 'e17965_1_00677.cxi'
    
    
    ptycho_results_filename = \
        calibration_cxi_file[:-4] + '_ptycho_reconstruction.h5'
    results_filename = \
        rpi_cxi_file[:-4] + f'_ssp_reconstruction.h5'

    ssp_results = cdtools.tools.data.h5_to_nested_dict(
        'reconstructions/' + results_filename)
    ptycho_results = cdtools.tools.data.h5_to_nested_dict(
        'reconstructions/' + ptycho_results_filename)

    ptycho_center = [815 ,665] 
    
    ptycho_radius = 300
    window = np.s_[ptycho_center[0]-ptycho_radius:
                   ptycho_center[0]+ptycho_radius,
                   ptycho_center[1]-ptycho_radius:
                   ptycho_center[1]+ptycho_radius]

    # for exponentiated objs
    ssp_results['obj'] = np.exp(1j*ssp_results['obj'])
    
    ssp_center = [254,247]
    
    ssp_radius = 13
    ssp_window = np.s_[ssp_center[0]-ssp_radius:
                       ssp_center[0]+ssp_radius,
                       ssp_center[1]-ssp_radius:
                       ssp_center[1]+ssp_radius]


    ssp_results['obj'] = ssp_results['obj'][ssp_window]
    ptycho_results['obj'] = np.exp(1j*ptycho_results['obj'])

    clim = [-1.035961, 0.26065496]

    ssp_obj_phase = np.angle(ssp_results['obj'])
    clipped_ssp_obj = np.minimum(np.maximum(ssp_obj_phase, clim[0]), clim[1])

    print('Pixel size:', np.abs(ssp_results['obj_basis'][0,1]))
    p.plot_real(clipped_ssp_obj, cmap='cividis')
    savefig_plus_im('figures/ssp_obj_phase')
    plt.show()
    
