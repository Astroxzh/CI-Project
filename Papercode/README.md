Replication data for "Single-shot X-ray ptychography as a structured illumination method"
-----------------------------------------------------------------------------------------

This folder contains the data and code required to replicate the results contained in
the paper, "Single-shot X-ray ptychography as a structured illumination method",
available as a preprint on the arXiv at https://doi.org/10.48550/arXiv.2410.19197.

Description of included files and folders
=========================================

### README.md

This file

### cdtools.zip

The version of the CDTools library used for this analysis is included as a zipped directory.

### environment.yml

A .yml export of the anaconda environment which CDTools was installed into.

### cxi_files

Contains two files in the [.cxi](https://www.cxidb.org) format. Each file contains one
of the two repeat ptychography scans. The data contained in these files is as produced by
the detectors and beamline readouts. The exceptions are:
* The detector images have been cropped
* A mask for hot and dead pixels has been applied
* Some values which were recorded manually, such as the sample-to-detector distance,
    are included

### reconstructions

Contains the saved ptychography, RPI, and SSP reconstructions reported in the paper

### convert_data.py

The script that was used to prepare the included .cxi files from the raw data and file
structure used at the cSAXS beamline of the Swiss Light Source.

### reconstruct_ptycho.py

Runs the ptychography pre-calibration on both repeat ptychography scans. \
WARNING: This will overwrite the stored ptychography reconstructions. You can always
re-download the exact ptychography reconstructions shown in the paper if you need to.

### reconstruct_rpi.py

Runs the RPI reconstruction on a diffraction pattern from the first scan, using a probe
calibration drawn from the ptychography reconstruction on the second scan.\
WARNING: This will overwrite the stored RPI reconstructions. You can always re-download
the exact RPI reconstructions shown in the paper if you need to.

### reconstruct_ssp.py

Prepares a diffraction pattern from the first scan and runs a single-shot ptychography
reconstruction on it
WARNING: This will overwrite the stored SSP reconstructions. You can always re-download
the exact RPI reconstructions shown in the paper if you need to.


### view_ptycho.py

Plots and saves a collection of figures showing the results of the two ptychography
reconstructions.

### view_rpi.py

Plots the results of the RPI reconstruction

### view_ssp.py

Plots the results of the SSP reconstruction

### helper_functions.py

Contains several helper functions called by the remaining scripts