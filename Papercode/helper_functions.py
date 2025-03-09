"""This file defines some helper functions that are broadly useful for this
project, but don't make sense to include in the main cdtools library

"""

import torch as t
import numpy as np
from scipy import interpolate
from scipy import optimize as opt
from cdtools.tools import image_processing as ip
from cdtools.tools import propagators
from cdtools.tools import plotting as p
import cdtools
from matplotlib import pyplot as plt

def savefig_plus_im(fname, vmin=None, vmax=None):
    """Saves the current figure, and the displayed im in full resolution.

    The figure gets saved as a pdf, but with imshow images, that pdf will
    have a sampled/rescaled version of the image. This function saves out
    a full resolution png at the same time.
    """
    plt.savefig(fname + '.pdf')#'figs/ptycho_obj_amplitude.pdf')
    im = plt.gca().get_images()[0]
    
    plt.imsave(fname + '.png', im.get_array(), cmap=im.cmap)


def fourier_pad(probe, padding):
    probe = propagators.far_field(t.clone(probe))
    probe = t.nn.functional.pad(probe, (padding, padding, padding, padding))
    return propagators.inverse_far_field(probe)


def center_probe(probe):
    # Make sure we dont screw with the input probe
    probe = t.clone(probe)
    for i in range(4):
        # Empirically, 4 iterations is repeatable to subpixel accuracy
        probe_abs_sq = t.sum(t.abs(probe)**2,axis=0)
        
        centroid = ip.centroid(probe_abs_sq)
        
        for i in range(probe.shape[0]):
            probe[i] = ip.sinc_subpixel_shift(probe[i],
                                              (-centroid[0]+probe.shape[-2]/2,
                                               -centroid[1]+probe.shape[-1]/2))

    return probe

def center_probe_fourier(probe):
    # Make sure we dont screw with the input probe
    probe = propagators.far_field(t.clone(probe))
    for i in range(4):
        # Empirically, 4 iterations is repeatable to subpixel accuracy
        probe_abs_sq = t.sum(t.abs(probe)**2,axis=0)
        
        
        centroid = ip.centroid(probe_abs_sq)
        
        for i in range(probe.shape[0]):
            probe[i] = ip.sinc_subpixel_shift(
                probe[i],
                (-centroid[0]+probe.shape[-2]/2,
                 -centroid[1]+probe.shape[-1]/2))

    return propagators.inverse_far_field(probe)


# The following helper functions help out with analysis of the results,
# for example by packaging two half/half reconstructions into a form
# where they can be compared

# Removes a phase ramp from a reconstructed image
def remove_phase_ramp(im, window, probe=None):
    #im = t.as_tensor(im)
    window = im[window]

    Is, Js = np.mgrid[:window.shape[0],:window.shape[1]]
    def zero_freq_component(freq):
        phase_ramp = np.exp(2j * np.pi * (freq[0] * Is + freq[1] * Js))
        return -np.abs(np.sum(phase_ramp * window))**2
    
    x0 = np.array([0,0])
    result = opt.minimize(zero_freq_component, x0)
    center_freq = result['x']

    Is, Js = np.mgrid[:im.shape[0],:im.shape[1]]
    phase_ramp = np.exp(2j * np.pi * (center_freq[0] * Is + center_freq[1] * Js))
    im = im * phase_ramp
    
    if probe is not None:
        Is, Js = np.mgrid[:probe.shape[-2],:probe.shape[-1]]
        phase_ramp = np.exp(-2j * np.pi * (center_freq[0] * Is + center_freq[1] * Js))
        probe = probe * phase_ramp
        return im, probe
    else:
        return im

# removes an exponential dependence of the amplitude on position, which can
# occur if shot-to-shot intensities are reconstructed
def remove_amplitude_exponent(im, window, probe=None, weights=None, translations=None, basis=None):
    window = np.abs(im[window])
    Is, Js = np.mgrid[:window.shape[0],:window.shape[1]]
    def rms_error(x):
        constant = x[0]
        growth_rate = x[1:]
        exponential_decay = constant * np.exp((growth_rate[0] * Is + growth_rate[1] * Js))
        return np.sum((window - exponential_decay)**2)
    
    x0 = np.array([1,0,0])
    result = opt.minimize(rms_error, x0, method='Nelder-Mead')
    growth_rate = result['x'][1:]
    #print(growth_rate)

    Is, Js = np.mgrid[:im.shape[0],:im.shape[1]]
    exponential_decay = np.exp(-(growth_rate[0] * Is + growth_rate[1] * Js))
    im = im * exponential_decay
    to_return = (im,)
    
    if probe is not None:
        Is, Js = np.mgrid[:probe.shape[-2],:probe.shape[-1]]
        exponential_decay = np.exp((growth_rate[0] * Is + growth_rate[1] * Js))
        probe = probe * exponential_decay
        to_return = to_return + (probe,)
        
    if weights is not None:
        pix_translations = cdtools.tools.interactions.translations_to_pixel(t.as_tensor(basis), t.as_tensor(translations)).numpy()
        pix_translations -= np.min(pix_translations,axis=0)
        weights = weights * np.exp(growth_rate[0] * pix_translations[:,0] + growth_rate[1] * pix_translations[:,1])
        to_return = to_return + (weights,)
    
    if len(to_return) == 1:
        return to_return[0]
    else:
        return to_return

# This is a good apodization function to use, to avoid the contribution of
# the sharp edge of the window.
def hann_window(im):
    Xs, Ys = np.mgrid[:im.shape[0],:im.shape[1]]
    Xhann = np.sin(np.pi*Xs/(im.shape[1]-1))**2
    Yhann = np.sin(np.pi*Ys/(im.shape[0]-1))**2
    Hann = Xhann * Yhann
    return im * Hann


def compare_rpi_with_ptycho(ptycho_results, rpi_results, window, nbins=30,
                            phase_frc=False):
    
    pixel_ratio = (np.abs(rpi_results['obj_basis'][0,1]) /
                   np.abs(ptycho_results['obj_basis'][0,1]))


    base_ptycho_obj = ptycho_results['obj'][window]
    initial_shape = np.array(base_ptycho_obj.shape)
    final_shape = initial_shape // pixel_ratio
    left_pad = ((initial_shape - final_shape + 1) // 2).astype(int)
    right_pad = (initial_shape - final_shape - left_pad).astype(int)

    rescaled_ptycho_obj = propagators.far_field(t.as_tensor(base_ptycho_obj))
    rescaled_ptycho_obj = t.nn.functional.pad(rescaled_ptycho_obj,
                        (-left_pad[-1], -right_pad[-1],
                         -left_pad[-2], -right_pad[-2]))
    rescaled_ptycho_obj =  propagators.inverse_far_field(rescaled_ptycho_obj)

    rpi_obj = rpi_results['obj']
    
    rpi_radius = int(rpi_obj.shape[0]/4)
    rpi_center = rpi_obj.shape[0]//2
    original_rpi_obj = rpi_obj.copy()
    rpi_obj = rpi_obj[rpi_center-rpi_radius:rpi_center+rpi_radius,
                      rpi_center-rpi_radius:rpi_center+rpi_radius]
    phase_offset = np.angle(np.sum(rpi_obj))

    #rpi_obj = rpi_obj * np.exp(-1j*phase_offset)
    #original_rpi_obj = original_rpi_obj * np.exp(-1j*phase_offset)

    ptycho_center = np.array(rescaled_ptycho_obj.shape) // 2
    clipped_ptycho_obj = rescaled_ptycho_obj[
        ptycho_center[0] - rpi_radius:
        ptycho_center[0] + rpi_radius,
        ptycho_center[1] - rpi_radius:
        ptycho_center[1] + rpi_radius].numpy()

    #print(rpi_obj.shape)
    #print(clipped_ptycho_obj.shape)
    #exit()
    shift = cdtools.tools.image_processing.find_shift(
        t.as_tensor(hann_window(rpi_obj)),
        t.as_tensor(hann_window(clipped_ptycho_obj)))

    shifted_ptycho_obj = cdtools.tools.image_processing.sinc_subpixel_shift(
        t.as_tensor(rescaled_ptycho_obj), shift).numpy()
    shifted_clipped_ptycho_obj = shifted_ptycho_obj[
        ptycho_center[0] - rpi_radius:
        ptycho_center[0] + rpi_radius,
        ptycho_center[1] - rpi_radius:
        ptycho_center[1] + rpi_radius]
    
    #shifted_rpi_obj = cdtools.tools.image_processing.sinc_subpixel_shift(
    #    t.as_tensor(original_rpi_obj), -shift).numpy()
    #shifted_clipped_rpi_obj = shifted_rpi_obj[
    #    rpi_center-rpi_radius:rpi_center+rpi_radius,
    #    rpi_center-rpi_radius:rpi_center+rpi_radius]
    

    if phase_frc:
        freqs, frc, threshold = cdtools.tools.analysis.calc_frc(
            hann_window(np.angle(shifted_clipped_ptycho_obj)),
            hann_window(np.angle(rpi_obj)),
            rpi_results['obj_basis'],
            nbins=nbins, limit='side',
            im_slice=np.s_[:,:], snr=1.6568) #2.5)
    else:
        freqs, frc, threshold = cdtools.tools.analysis.calc_frc(
            hann_window(shifted_clipped_ptycho_obj),
            hann_window(rpi_obj),
            rpi_results['obj_basis'],
            nbins=nbins, limit='side',
            im_slice=np.s_[:,:], snr=1.6568) #2.5)
        
        
    frc = np.real(frc)

    # We actually use the FRC**2 here, because we're comparing one image
    # with noise to an image without noise. The asymptotic result in this
    # case is FRC = 1 / sqrt(1 + 1 / SNR), not FRC=1/(1 + 1/SNR) as it
    # is in the more typical comparison of two images, both with noise.
    ssnr = frc**2 / (1 - frc**2)
    # This is what I used to think it was, but this was wrong
    #ssnr = 0.5 * frc / (1 - frc)

    # The normalized MSE minimized over an arbitrary global amplitude and phase
    # factor
    eps = 1 - (np.abs(np.sum(hann_window(shifted_clipped_ptycho_obj).conj()
                             * hann_window(rpi_obj)))**2
               /  ( np.sum(np.abs(hann_window(shifted_clipped_ptycho_obj))**2)
                    * np.sum(np.abs(hann_window(rpi_obj))**2)))

    results = {'rpi_obj': rpi_obj,
               'ptycho_obj': clipped_ptycho_obj,
               'shifted_ptycho_obj': shifted_clipped_ptycho_obj,
               'basis': rpi_results['obj_basis'],
               'frc_freqs': freqs,
               'frc': frc,
               'frc_threshold': threshold,
               'ssnr': ssnr,
               'eps': eps}
    
    return results


def calc_sqrt_fidelity(fields_1, fields_2, dims=2):
    """Calculates the square-root-fidelity between two multi-mode wavefields

    The fidelity is a comparison metric between two density matrices
    (i.e. mutual coherence functions) that extends the idea of the
    overlap to incoherent light. As a reminder, the overlap between two
    fields is:

    overlap = abs(sum(field_1 * field_2))
    
    Whereas the square-root-fidelity is defined as:
    
    sqrt_fidelity = trace(sqrt(sqrt(dm_1) <dot> dm_2 <dot> sqrt(dm_1)))

    where dm_n refers to the density matrix encoded by fields_n such
    that dm_n = fields_n <dot> fields_n.conjtranspose(), sqrt
    refers to the matrix square root, and <dot> is the matrix product.
    
    The definition above is not practical, however, as it is not feasible
    to explicitly construct the matrices dm_1 and dm_2 in memory. Therefore,
    we take advantage of the alternate definition based directly on the
    fields_n parameter:

    sqrt_fidelity = sum(svdvals(fields_1 <dot> fields_2.conjtranspose()))
    
    In the definitions above, the fields_n are regarded as collections of
    wavefields, where each wavefield is by default 2-dimensional. The
    dimensionality of the wavefields can be altered via the dims argument,
    but the fields_n arguments must always have at least one more dimension
    than the dims argument. Any additional dimensions are treated as batch
    dimensions.
    
    Parameters
    ----------
    fields_1 : t.Tensor
        The first set of complex-valued field modes
    fields_2 : t.Tensor
        The second M2xN set of complex-valued field modes 
    dims : int
        Default is 2, the number of final dimensions to reduce over.

    Returns
    -------
    fidelity : float or t.Tensor
        The fidelity, or tensor of fidelities, depending on the dim argument

    """

    # These lines generate the matrix of inner products between all the modes
    mult = fields_1.unsqueeze(-dims-2) * fields_2.unsqueeze(-dims-1).conj()
    sumdims = tuple(d - dims for d in range(dims))
    mat = t.sum(mult,dim=sumdims)

    # The nuclear norm is the sum of the singular values
    return t.linalg.matrix_norm(mat, ord='nuc')


def calc_pcmse(fields_1, fields_2, reduction='mean', dims=2):
    """Calculates the PCMSE between two complex partially coherent wavefields

    This function calculates a generalization of the RMS error which uses the
    concept of fidelity to capture the error between incoherent wavefields.
    The extension has several nice properties, in particular:

    1) For coherent wavefields, it precisely matches the magnitude of the
       RMS error.
    2) All mode decompositions of either field that correspond to the same
       density matrix / mutual coherence function will produce the same 
       output
    3) The error will only be zero when comparing mode decompositions that
       correspond to the same density matrix.
    4) Due to (2), one need not worry about the ordering of the modes,
       properly orthogonalizing the modes, and it is even possible to
       compare mode decompositions with different numbers of modes.    
    
    The formal definition of this function, with default options, is:

    output = ( sum(abs(fields_1)**2) + sum(abs(fields_2)**2)
              - 2 * sqrt_fidelity(fields_1,fields_2) ) / npix
    
    Where npix is the number of pixels in the wavefields. If the reduction is
    specified as 'sum', then the result is not divided by this constant.

    In the definitions above, the fields_n are regarded as collections of
    wavefields, where each wavefield is by default 2-dimensional. The
    dimensionality of the wavefields can be altered via the dims argument,
    but the fields_n arguments must always have at least one more dimension
    than the dims argument. Any additional dimensions are treated as batch
    dimensions.
    
    Parameters
    ----------
    fields_1 : t.Tensor
        The first set of complex-valued field modes
    fields_2 : t.Tensor
        The second set of complex-valued field modes
    normalize : bool
        Default is False, whether to normalize to the intensity of fields_1
    dims : (int or tuple of python:ints)
        Default is 2, the number of final dimensions to reduce over.

    Returns
    -------
    rms_error : float or t.Tensor
        The generalized RMS error, or tensor of generalized RMS errors,
        depending on the dim argument
    """

    sumdims = tuple(d - dims - 1 for d in range(dims+1))
    fields_1_intensity = t.sum(t.abs(fields_1)**2, dim=sumdims) 
    fields_2_intensity = t.sum(t.abs(fields_2)**2, dim=sumdims) 
    sqrt_fidelity = calc_sqrt_fidelity(fields_1, fields_2, dims=dims) 

    result = fields_1_intensity + fields_2_intensity - 2 * sqrt_fidelity

    if reduction.strip().lower() == 'mean':
        # The number of pixels in the wavefield
        npix = t.prod(t.as_tensor(fields_1.shape[-dims:],dtype=t.int32))
        return result / npix
    elif reduction.strip().lower() == 'sum':
        return result
    else:
        raise ValueError("The only valid reductions are 'mean' and 'sum'")
    

def calc_pcfrc(fields_1, fields_2, bins):
    """Calculates the PCFRC between two complex partially coherent wavefields

    This function assumes that the fields are input in the form of stacked
    2D images with dimensions MxN1xN2. M is the number of coherent modes,
    and N1 and N2 are the dimensions of the images in I and J. While the
    image sizes must match, the number of modes need not be equivalent.
    
    The returned correlation is a function of spatial frequency, which is
    measured in units of the inverse pixel size.

    Parameters
    ----------
    im1 : t.Tensor
        The first image, a set of complex or real valued arrays
    im2 : t.Tensor
        The first image, a stack of complex or real valued arrays
    bins : int
        Number of bins to break the FRC up into

    Returns
    -------
    freqs : array
        The frequencies associated with each FRC value
    FRC : array
        The FRC values

    """
    # We Fourier transform the two wavefields
    f1 = t.fft.fftshift(t.fft.fft2(fields_1),dim=(-1,-2))
    f2 = t.fft.fftshift(t.fft.fft2(fields_2),dim=(-1,-2))

    # And we generate the associated 2d map of spatial frequencies
    i_freqs = t.fft.fftshift(t.fft.fftfreq(f1.shape[-2]))
    j_freqs = t.fft.fftshift(t.fft.fftfreq(f1.shape[-1]))
    Js,Is = t.meshgrid(j_freqs,i_freqs)
    Rs = t.sqrt(Is**2+Js**2)

    # These lines get a set of spatial frequency bins that match the logic
    # used by np.histogram.
    n_pix, bins = np.histogram(Rs, bins=bins)
    bins = t.as_tensor(bins)
    
    frc = []
    for i in range(len(bins)-1):
        # This implements the projection operator to the appropriate ring
        mask = t.logical_and(Rs<bins[i+1], Rs>=bins[i])
        masked_f1 = f1 * mask[...,:,:]
        masked_f2 = f2 * mask[...,:,:]

        # And we calculate the sqrt_fidelity of the projected wavefields
        numerator = calc_sqrt_fidelity(masked_f1, masked_f2)
        
        denominator_f1 = t.sum(t.abs(masked_f1)**2)
        denominator_f2 = t.sum(t.abs(masked_f2)**2)
        frc.append(numerator / t.sqrt((denominator_f1 * denominator_f2)))

    frc = t.as_tensor(np.array(frc))

    return bins[:-1], frc


