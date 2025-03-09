import cdtools
from cdtools.tools import plotting as p
from matplotlib import pyplot as plt
import torch as t
import numpy as np
from helper_functions import compare_rpi_with_ptycho, savefig_plus_im

if __name__ == '__main__':

    # For repeat 0
    cxi_file_0 = 'e17965_1_00677.cxi'

    # For repeat 1
    cxi_file_1 = 'e17965_1_00678.cxi'

    
    results_filename_0 = cxi_file_0[:-4] + f'_ptycho_reconstruction.h5'
    results_filename_1 = cxi_file_1[:-4] + f'_ptycho_reconstruction.h5'

    # And we save out the results as a .mat file
    results_0 = cdtools.tools.data.h5_to_nested_dict(
        'reconstructions/' + results_filename_0)
    results_1 = cdtools.tools.data.h5_to_nested_dict(
        'reconstructions/' + results_filename_1)

    results_0['obj'] = np.exp(1j*results_0['obj'])
    results_1['obj'] = np.exp(1j*results_1['obj'])
    
    sl = np.s_[400:1050,450:1100]

    compared_results = cdtools.tools.analysis.standardize_reconstruction_pair(
        results_0, results_1, window=sl)

    print('Probe nrms error:', compared_results['probe_nrms_error'])
    print('Probe nmse:', compared_results['probe_nrms_error']**2)
    print('Probe amplitude-minimized nmse:', compared_results['probe_nmse'])
    
    dataset_0 = cdtools.datasets.Ptycho2DDataset.from_cxi('cxi_files/' + cxi_file_0)
    dataset_1 = cdtools.datasets.Ptycho2DDataset.from_cxi('cxi_files/' + cxi_file_1)
    print('Wavelength:', 1e10*dataset_0.wavelength, 'Angstrom')
    print('Distance:', dataset_0.detector_geometry['distance'], 'm')
    print('Pix Size', np.abs(1e6*dataset_0.detector_geometry['basis'][0,1]), 'um')
    
    dataset_1.inspect()
    plt.savefig('figures/ptycho_dataset_1.pdf')
    p.plot_real(t.log10(dataset_0.patterns[70] + 1), cmap='inferno')
    savefig_plus_im('figures/RPI_diffraction_pattern')


    mode_powers = np.sum(np.abs(results_1['probe'])**2, axis=(1,2))
    mode_powers = 100 * mode_powers / np.sum(mode_powers)
    print('Mode powers:', mode_powers)
    plt.bar(np.arange(len(mode_powers)), mode_powers)
    plt.savefig('figures/ptycho_dataset_1_probe_mode_power.pdf')

    spacing = [np.abs(results_1['probe_basis'][0,1])]*2
    z1 = -0.955e-3
    prop1 = cdtools.tools.propagators.generate_angular_spectrum_propagator(
        results_1['probe'].shape[-2:],
        spacing,
        results_1['wavelength'],
        z1
    )
    p.plot_colorized(
        cdtools.tools.propagators.near_field(
            t.as_tensor(results_1['probe']),
            prop1
        ),
        basis=results_1['probe_basis']
    )
    savefig_plus_im(f'figures/ptycho_at_grating_colorized_mode_0')

    z2 = -5.5e-3
    prop2 = cdtools.tools.propagators.generate_angular_spectrum_propagator(
        results_1['probe'].shape[-2:],
        spacing,
        results_1['wavelength'],
        z2
    )
    p.plot_colorized(
        cdtools.tools.propagators.near_field(
            t.as_tensor(results_1['probe']),
            prop2
        ),
        basis=results_1['probe_basis']
    )
    savefig_plus_im(f'figures/ptycho_at_focus_colorized_mode_0')

    fft = cdtools.tools.propagators.far_field(t.as_tensor(results_0['probe']))
    logfft = np.log10(t.abs(fft)**2 + 1) * t.exp(1j * t.angle(fft))
    p.plot_colorized(logfft)
    savefig_plus_im(f'figures/ptycho_fft_log_colorized_mode_0')

    
    for idx in range(2):
        p.plot_colorized(results_1['probe'][idx],
                         basis=results_1['probe_basis'])
        savefig_plus_im(f'figures/ptycho_probe_mode_{idx}')
        p.plot_real(np.log10(t.abs(cdtools.tools.propagators.far_field(t.as_tensor(results_0['probe'][idx])))**2 + 1),
                    basis=results_1['probe_basis'])
        savefig_plus_im(f'figures/ptycho_fourier_probe_log_plus_1_mode_{idx}')
                

    p.plot_phase(results_1['obj'][sl], basis=results_1['obj_basis'],
                 cmap='cividis')
    savefig_plus_im('figures/ptycho_1_obj_phase')

    plt.figure()
    plt.semilogy(1e-6*compared_results['frc_freqs'], compared_results['ssnr'])
    plt.xlabel('Spatial Frequency (cycles/um)')
    plt.ylabel('SSNR')
    plt.title('SSNR of ptycho estimated via FRC.pdf')
    plt.grid(which='both')
    plt.savefig('figures/Ptycho_SSNR.pdf')
    
    plt.figure()
    plt.plot(1e-6*compared_results['probe_freqs'],
             compared_results['probe_frc'])
    plt.title('Generalized FRC of probe')
    plt.xlabel('Spatial Frequency (cycles/um)')
    plt.savefig('figures/Ptycho_Probe_Generalized_FRC.pdf')
    plt.show()
    
