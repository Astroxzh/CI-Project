#%%
import numpy as np
from math import *
import matplotlib.pyplot as plt


def sphericalWaveG(k, radius, rs):
    # r in meter
    R = np.sqrt(rs + radius**2)
    sphericalwave = 1 * np.exp(1j * k * R) / R
    return sphericalwave 


def angularPro(wavefront, aperture, kz, z):
    # hologram generation
    H = np.exp(1j * kz * z) #transfer function

    objWave = wavefront * aperture
    objWaveF = np.fft.fftshift(np.fft.fft2(objWave))
    imWaveF = objWaveF * H
    hologram = np.abs(imWaveF)**2

    # reconstruction
    recObj = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(hologram)))
    reconstruction = np.abs(recObj)
    return hologram, reconstruction

# setup parameters
dx = 11e-6
N = 2048
z = 4.5e-3
wavelength = 0.2e-9
kwn = 2 * pi / wavelength # wavenumber

# spatial and frequency domain sampling
# image plane
Lx = dx * N
x = np.linspace(-Lx / 2, Lx / 2, N )
y = np.copy(x)
X, Y = np.meshgrid(x, y)
r2 = X**2 + Y**2

kx = np.fft.fftfreq(N, dx)
ky = np.copy(kx)
kX, kY = np.meshgrid(np.fft.fftshift(kx), np.fft.fftshift(ky))
kf = kX**2 + kY**2

kz = np.sqrt(kwn**2 - kf)

# obj plane
dxO = wavelength * z / Lx # obj plane side length
LxO = wavelength * z / dx # obj plane sampling  
# if do so, the pixel size will be in diffraction limit
xO = np.arange(-LxO / 2, LxO / 2, dxO)
yO = np.copy(xO)
XO, YO = np.meshgrid(xO, yO)
rO2 = XO**2 + YO**2

sw70cm = sphericalWaveG(kwn, 4.5e-3, rO2)

plt.imshow((np.abs(sw70cm))**2)
plt.show()