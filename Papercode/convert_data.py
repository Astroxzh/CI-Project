import numpy as np
import h5py
import os
import torch as t
from matplotlib import pyplot as plt
from cdtools.tools import plotting as p
from cdtools.datasets import Ptycho2DDataset
from scipy import io

def import_scan(filename,
                prefix='/das/work/p17/p17965/online_data/data',
                out_prefix='cxi_files',
                mask_file='../../online_data/cxs_software/ptycho/+detector/+pilatus/binary_mask_6.2keV.mat',
                view=False):

    with h5py.File(prefix + '/' + filename, 'r') as f:

        patterns = np.array(f['entry/data/data'])
        patterns = patterns.astype(np.int32)
        
        x = np.array(f['entry/collection/specES1/px']) * 1e-6
        y = np.array(f['entry/collection/specES1/py']) * 1e-6
        wavelength = np.array(f['entry/instrument/monochromator/wavelength'])
        wavelength = wavelength.ravel()[0] * 1e-10

        det_x_pix = np.array(f['entry/data/x_pixel_size']).ravel()[0] * 1e-6
        det_y_pix = np.array(f['entry/data/y_pixel_size']).ravel()[0] * 1e-6

        distance = 7.336 # Manually defined
                            
        translations = np.stack([x, y, 0*x], axis=1)


    patterns = np.flip(patterns, (-1,-2))

    mask_2 = io.loadmat(mask_file)['mask']

    mask_3 = np.ones(mask_2.shape, dtype=np.uint8)
    mask_3[242,328] = 0
    mask_3[234,321] = 0
    mask_3[216,250] = 0

    mask = mask_2 * mask_3

    center = (943, 738)
    radius = (512, 512)

    patterns = patterns[:,
                        center[0]-radius[0]:center[0]+radius[0],
                        center[1]-radius[1]:center[1]+radius[1]]
    mask = mask[center[0]-radius[0]:center[0]+radius[0],
                center[1]-radius[1]:center[1]+radius[1]]

    patterns = patterns * mask
    
    detector_geometry = {
        'basis' : t.as_tensor(np.array([[0,-det_x_pix],
                                        [-det_y_pix,0],
                                        [0,0]], dtype=np.float32)),
        'distance' : distance
    }
    
    pix_size = wavelength * detector_geometry['distance'] / (500 * 75e-6)    
    
    dataset = Ptycho2DDataset(translations, patterns,
                              wavelength=wavelength,
                              detector_geometry=detector_geometry,
                              mask=mask)
    
    # Crop the outer 200 pixels with very low signal
    dataset.pad(-200)

    # This saves out the .cxi file
    filename_end = filename.split('/')[-1]
    print('processing', filename_end)
    dataset.to_cxi(out_prefix + '/' + filename_end[:-3] + '.cxi')
    if view:
        dataset.inspect()        


if __name__ == '__main__':

    def make_filename(index):
        if index < 1000:
            base = 'S00000-00999'
        else:
            base = 'S01000-01999'

        return base + f'/S{index:05}/e17965_1_{index:05}.h5'

    import_scan(make_filename(677), view=True)
    import_scan(make_filename(678), view=True)
    plt.show()
