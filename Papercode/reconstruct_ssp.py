import cdtools
from cdtools.tools import plotting as p
from cdtools.tools import image_processing as ip
from matplotlib import pyplot as plt
import torch as t
import numpy as np

# All beams within the fourth order which are
indices = [
    (0,0),
    (1,0),
    (-1,0),
    (0,1),
    (0,-1),
    (1,-1),
    (-1,1),
    (0,2),
    (0,-2),
    (2,-2),
    (-2,2),
    (1,-3),
    (-1,3),
    (2,-3),
    (-2,3),
    (-1,2),
    (1,-2),
    (1,1),
    (-1,-1),
    (2,-1),
    (-2,1),
    (0,3),
    (0,-3),
    (3,-3),
    (-3,3),
    (3,-4),
    (-3,4),
    (2,-4),
    (-2,4),
    (1,-4),
    (-1,4),
    (3,0),
    (-3,0),
    (4,0),
    (-4,0),
    (2,2),
    (-2,-2),
    (4,-2),
    (-4,2),
    (4,-1),
    (-4,1),
    (3,1),
    (-3,-1),    
]

def reconstruct_ssp(
        ssp_cxi_file,
        shot_idx,
        center,
        angle,
        radius,
        view=False,
        save_filename_end='_ssp_reconstruction.h5',
        device='cpu'):

        
    # We load the dataset containing the RPI exposure
    dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(
        'cxi_files/' + ssp_cxi_file)

    dataset.mask[312,338:346] = 0
    dataset.patterns *= dataset.mask

    center = np.array(center)
    vec_0 = radius * np.array([np.cos(angle), np.sin(angle)])
    vec_1 = radius * np.array([np.cos(angle + np.pi / 3), np.sin(angle + np.pi / 3)])
    
    full_pattern = dataset.patterns[shot_idx]
    sum_pattern = t.sum(dataset.patterns,dim=0)
    
    p.plot_real(t.log10(full_pattern + 1))
    points = []
    translations = []
    for index in indices:
        point = center + index[0] * vec_0 + index[1] * vec_1
        translation = [-point[1] / radius * 1.2e-6,
                       -point[0] / radius * 1.2e-6,
                       0]
                       
        points.append(point)
        translations.append(translation)

    points = np.stack(points, axis=0)
    translations = t.as_tensor(np.stack(translations, axis=0))
    
    plt.plot(points[:,1], points[:,0], 'k.')
    
    Is, Js = np.mgrid[:radius,:radius]
    Is = Is - np.mean(Is)
    Js = Js - np.mean(Js)

    mask = np.ones(Is.shape, dtype=int)
    for index in [(1,0),(0,1),(1,-1)]:
        vec = index[0] * vec_0 + index[1] * vec_1
        vec = vec / np.linalg.norm(vec)
        dist = Is * vec[0] + Js * vec[1]
        mask[np.abs(dist) >= radius / 2] = 0
        
    p.plot_real(mask)

    patterns = np.zeros([len(indices), mask.shape[0],mask.shape[1]])
    for idx, point in enumerate(points):
        topleft = [int(np.round(point[0] - Is.shape[0]/2)),
                   int(np.round(point[1] - Is.shape[1]/2))]

        patterns[idx] = full_pattern[topleft[0]:topleft[0] + Is.shape[0],
                                    topleft[1]:topleft[1] + Is.shape[1]]
        patterns[idx] *= mask

    patterns = t.as_tensor(patterns)
        
    p.plot_real(t.log10(patterns + 1))

    dataset.mask = t.as_tensor(mask)
    dataset.translations = translations
    dataset.patterns = patterns
    dataset.intensities = t.sqrt(t.sum(patterns, dim=(1,2)))
    dataset.intensities /= t.max(dataset.intensities)
    
    dataset.inspect()

    model = cdtools.models.FancyPtycho.from_dataset(
        dataset,
        n_modes=2,
        propagation_distance=6e-3,
        # We restrict the probe to a small region at the start
        obj_view_crop=-20,
        probe_fourier_crop=7,#11, #7 for edge crop, 11 for corner crop
        probe_support_radius=20,
        # Starting with a phase-only object helps the edges of the probe
        # reconstruct properly with the small scan region we're working with
        exponentiate_obj=True,
        phase_only=True,
        allow_probe_fourier_shifts=True,
    )
    
    # Move the data and model to the appropriate device
    model.to(device=device)
    dataset.to(device=device)
    dataset.get_as(device=device)
    
    for i, loss in enumerate(model.Adam_optimize(200, dataset, lr=0.002, batch_size=5, schedule=False)):
        # And we liveplot the updates to the model as they happen
        print(model.report())
        if view and i % 100 == 0:
            model.inspect(dataset)

    model.probe_support[:] = 1

    for i, loss in enumerate(model.Adam_optimize(500, dataset, lr=0.002, batch_size=5, schedule=False)):
        # And we liveplot the updates to the model as they happen
        print(model.report())
        if view and i % 100 == 0:
            model.inspect(dataset)


    for i, loss in enumerate(model.Adam_optimize(1000, dataset, lr=0.0002, batch_size=50, schedule=False)):
        # And we liveplot the updates to the model as they happen
        print(model.report())
        if view and i % 100 == 0:
            model.inspect(dataset)


    save_filename = ssp_cxi_file[:-4] + save_filename_end
    model.save_to_h5('reconstructions/' + save_filename, dataset)
    
    model.inspect(dataset)
    model.compare(dataset)

    

if __name__ == '__main__':

    device='cuda:0'
    
    shot_idx = 70

    ssp_cxi_file = 'e17965_1_00677.cxi'
        
    reconstruct_ssp(
        ssp_cxi_file,
        shot_idx,
        (310,313),
        0.01,
        49,
        view=True,
        device=device,
        save_filename_end='_ssp_reconstruction.h5',
    )
    
    plt.show()
