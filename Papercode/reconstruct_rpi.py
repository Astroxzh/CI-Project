import cdtools
from cdtools.tools import plotting as p
from matplotlib import pyplot as plt
import torch as t
import numpy as np


def reconstruct_rpi(
        calibration_cxi_file,
        rpi_cxi_file,
        shot_idx,
        obj_size,
        view=False,
        save_filename_end='rpi_reconstruction.h5',
        device='cpu'):

    
    results_filename = calibration_cxi_file[:-4] + '_ptycho_reconstruction.h5'

    results = cdtools.tools.data.h5_to_nested_dict(
        'reconstructions/' + results_filename)
    
    # We load the dataset containing the RPI exposure
    dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(
        'cxi_files/' + rpi_cxi_file)

    results['probe'] = results['probe'] * np.mean(results['weights'])
    
    model = cdtools.models.RPI.from_dataset(
        dataset,
        results['probe'],
        n_modes=1,
        obj_size=[obj_size, obj_size],
        background=results['background'],
        exponentiate_obj=True,
        phase_only=True,
        initialization='uniform',
    )

    # Move the data and model to the appropriate device
    model.to(device=device)
    dataset.to(device=device)
    dataset.get_as(device=device)

    for loss in model.LBFGS_optimize(
            50,
            dataset,
            lr=0.4,
            regularization_factor=[0.05],
            subset=[shot_idx]
    ):
        if view and model.epoch % 25 == 0:
            model.inspect(dataset)
        print(model.report())
        
    save_filename = rpi_cxi_file[:-4] + save_filename_end
    model.save_to_h5('reconstructions/' + save_filename, dataset)
    
    if view:
        model.inspect(dataset)

if __name__ == '__main__':

    device='cuda:0'

    shot_idx = 70

    obj_size = 400

    save_filename_end = f'_shot_idx_{shot_idx}_obj_size_{obj_size}.h5'
    
    # For repeat 0, grating in
    calibration_cxi_file = 'e17965_1_00678.cxi'
        
    # For repeat 1, grating in
    rpi_cxi_file = 'e17965_1_00677.cxi'
        
    reconstruct_rpi(calibration_cxi_file,
                    rpi_cxi_file,
                    shot_idx,
                    obj_size=obj_size,
                    save_filename_end=save_filename_end,
                    view=True,
                    device=device)
    
    plt.show()
