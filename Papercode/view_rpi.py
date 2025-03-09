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
        rpi_cxi_file[:-4] + f'_shot_idx_{shot_idx}_obj_size_{obj_size}.h5'

    results = cdtools.tools.data.h5_to_nested_dict(
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
    results['obj'] = np.exp(1j*results['obj'])
    
    # For oversampling=1
    rpi_window = np.s_[:,:] 

    results['obj'] = results['obj'][rpi_window]
    ptycho_results['obj'] = np.exp(1j*ptycho_results['obj'])

    nbins=25
    comparison = compare_rpi_with_ptycho(ptycho_results, results, window,
                                         nbins=nbins)

    clim = [np.min(np.angle(comparison['ptycho_obj'])),
            np.max(np.angle(comparison['ptycho_obj']))]            

    # this is done to save the image with the right colorbar scale
    ptycho_obj = np.minimum(np.angle(comparison['shifted_ptycho_obj']), clim[1])
    ptycho_obj = np.exp(1j*np.maximum(ptycho_obj, clim[0]))

    p.plot_phase(ptycho_obj, basis=comparison['basis'], cmap='cividis')
    savefig_plus_im('figures/ptycho_downsampled_phase')

    row = 53
    upsampled = cdtools.tools.image_processing.fourier_upsample(
        cdtools.tools.image_processing.fourier_upsample(
            cdtools.tools.image_processing.fourier_upsample(
                t.as_tensor(comparison['rpi_obj'])))).numpy()
    upsampled_ptycho = cdtools.tools.image_processing.fourier_upsample(
        cdtools.tools.image_processing.fourier_upsample(
            cdtools.tools.image_processing.fourier_upsample(
                t.as_tensor(comparison['shifted_ptycho_obj'])))).numpy()
    original = comparison['rpi_obj'][row,90:110]
    original_ptycho = comparison['shifted_ptycho_obj'][row,90:110]
    xaxis = np.arange(20) * np.abs(comparison['basis'][0,1])
    smoothed = upsampled[row*8, 90*8:110*8]
    smoothed_ptycho = upsampled_ptycho[row*8, 90*8:110*8]
    smoothed_xaxis = np.arange(20*8) * np.abs(comparison['basis'][0,1]) / 8

    plt.figure()#figsize=(3,4), dpi=600)
    plt.plot(1e9 * smoothed_xaxis, np.angle(smoothed_ptycho), 'k-', label='ptycho')
    plt.plot(1e9 * xaxis, np.angle(original_ptycho), 'k.')
    plt.plot(1e9 * smoothed_xaxis, np.angle(smoothed), label='phase')
    plt.plot(1e9 * xaxis, np.angle(original), 'k.')
    plt.xlabel('x (nm)')
    plt.ylabel('Phase (rad)')
    plt.legend()
    plt.savefig('figures/Lineout.pdf')

    # this is done to save the image with the right colorbar scale
    rpi_obj = np.minimum(np.angle(comparison['rpi_obj']), clim[1])
    rpi_obj = np.exp(1j*np.maximum(rpi_obj, clim[0]))
      
    p.plot_phase(rpi_obj, basis=comparison['basis'], cmap='cividis')
    plt.clim(clim)
    savefig_plus_im('figures/rpi_phase')

    plt.figure()
    plt.semilogy(1e-6*comparison['frc_freqs'], comparison['ssnr'])
    plt.semilogy(1e-6*comparison['frc_freqs'],
                 np.ones(len(comparison['ssnr']))*0.4142, 'k--')
    plt.grid(which='both')
    plt.xlabel('Spatial Frequency (cycles/um)')
    plt.ylabel('SSNR')
    plt.savefig('figures/RPI_SSNR.pdf')

    plt.show()
