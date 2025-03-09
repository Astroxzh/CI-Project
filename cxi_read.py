import cdtools
from matplotlib import pyplot as plt
import torch as t
import numpy as np

def cxi_read(cxi_file):
    dataset = cdtools.datasets.Ptycho2DDataset.from_cxi('cxi_files/' + cxi_file)
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
