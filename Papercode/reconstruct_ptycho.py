import cdtools
from matplotlib import pyplot as plt
import torch as t
import numpy as np
import os



def reconstruct_ptycho(
        cxi_file,
        view=False,
        device='cpu'):

    # First, we load the dataset from a .cxi file
    dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(r'D:\github\CI-Project\Papercode\cxi_files' + '\\' + cxi_file)

    model = cdtools.models.FancyPtycho.from_dataset(
        dataset,
        # Empirically, the top two modes were the most important
        n_modes=2,
        propagation_distance=6e-3,
        # We restrict the probe to a small region at the start
        probe_support_radius=280,
        obj_view_crop=-500,
        # Starting with a phase-only object helps the edges of the probe
        # reconstruct properly with the small scan region we're working with
        exponentiate_obj=True,
        phase_only=True,
    )

    # Move the data and model to the appropriate device
    model.to(device=device)
    dataset.to(device=device)
    dataset.get_as(device=device)

    # We start with some aggressive probe refinement
    for i, loss in enumerate(model.Adam_optimize(50, dataset, lr=0.02, batch_size=5, schedule=False)):
        # And we liveplot the updates to the model as they happen
        print(model.report())
        if view and i % 5 == 0:
            model.inspect(dataset)

    # Once we have the central part nicely converged, we remove the restriction
    # on the probe
    model.probe_support[:] = 1

    # We restrict the object's real part to a small range, because some pixels
    # at the edge jump past pi and get a phase wrap. This just keeps the
    # final result unwrapped so it's nice to look at without further processing
    model.obj.data.real = t.clamp(model.obj.data.real, min=-1, max=1)

    # Continue refining the probe, but slowly expand the probe support
    for i, loss in enumerate(model.Adam_optimize(50, dataset, lr=0.02, batch_size=5, schedule=False)):
        # And we liveplot the updates to the model as they happen
        print(model.report())
        if view and i % 5 == 0:
            model.inspect(dataset)

    # Continue refining the probe, but slowly expand the probe support
    for i, loss in enumerate(model.Adam_optimize(100, dataset, lr=0.008, batch_size=5, schedule=False)):
        # And we liveplot the updates to the model as they happen
        print(model.report())
        if view and i % 5 == 0:
            model.inspect(dataset)

    # We finish by letting the amplitude vary, once the probe is well converged
    model.phase_only = t.as_tensor(False)

    # And the next two steps polish the reconstruction
    for i, loss in enumerate(model.Adam_optimize(50, dataset, lr=0.003, batch_size=25, schedule=False)):
        # And we liveplot the updates to the model as they happen
        print(model.report())
        if view and i % 5 == 0:
            model.inspect(dataset)

    for i, loss in enumerate(model.Adam_optimize(100, dataset, lr=0.001, batch_size=500, schedule=True)):
        # And we liveplot the updates to the model as they happen
        print(model.report())
        if view and i % 10 == 0:
            model.inspect(dataset)

    # This orthogonalizes the incoherent probe modes
    model.tidy_probes()

    save_filename = cxi_file[:-4] + '_ptycho_reconstruction.h5'

    # And we save out the results as a .h5 file
    model.save_to_h5('reconstructions/' + save_filename, dataset)

    # Finally, we plot the results
    if view:
        model.inspect(dataset)
        model.compare(dataset)


if __name__ == '__main__':

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # Set this to the device you would like to run the calculations on
    device = 'cuda:0'

    # For repeat 0, grating in
    cxi_file = 'e17965_1_00677.cxi'

    reconstruct_ptycho(cxi_file, view=True, device=device)

    # For repeat 1, grating in
    cxi_file = 'e17965_1_00678.cxi'

    reconstruct_ptycho(cxi_file, view=True, device=device)

    plt.show()

#%%
