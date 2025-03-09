import numpy as np
import matplotlib.pyplot as plt

# --------------------
# 参数设置
# --------------------
wavelength = 531e-9       # 波长 [m]
k = 2 * np.pi / wavelength # 波数

d_source = 1e-1       # 点光源到光栅平面的距离 1 cm
d_prop = 1e1             # 光栅到探测器的传播距离 5 mm

# 定义光栅平面上二维网格（假设正方形区域）
Lx = 2e-3   # x方向总长度 2 mm
Ly = 2e-3   # y方向总长度 2 mm
Nx = 1024   # x方向采样点数
Ny = 1024   # y方向采样点数

dx = Lx / Nx
dy = Ly / Ny

x = np.linspace(-Lx/2, Lx/2, Nx)
y = np.linspace(-Ly/2, Ly/2, Ny)
X, Y = np.meshgrid(x, y)

# --------------------
# 1. 计算光栅平面上的球面波场分布
# --------------------
# 点光源位于 z = -d_source 处
r = np.sqrt(X**2 + Y**2 + d_source**2)
E_in = (1.0 / r) * np.exp(1j * k * r)  # 忽略常数因子

# --------------------
# 2. 定义一维正弦光栅（仅沿 x 方向调制）
# --------------------
grating_period = 50e-6     # 光栅周期 50 um
modulation_depth = 0.5     # 调制深度
T = 1 + modulation_depth * np.cos(2 * np.pi * X / grating_period)
E_after = E_in * T         # 光栅后的场分布

# --------------------
# 3. Fresnel 衍射（二维）传播到探测器平面
# --------------------
def fresnel_propagation_2d(E0, dx, dy, wavelength, z):
    """
    利用 FFT 计算从 z=0 平面传播 z 距离后的二维场分布（Fresnel 衍射）。
    
    参数:
      E0         : 光栅平面上的场分布 (二维数组)
      dx, dy     : 光栅平面上 x 和 y 方向的采样间隔 [m]
      wavelength : 波长 [m]
      z          : 传播距离 [m]
    返回:
      E_out      : 探测器平面上的场分布 (二维数组)
      x_out, y_out : 探测器平面对应的坐标 (二维数组)
    """
    k = 2 * np.pi / wavelength
    (Ny, Nx) = E0.shape
    
    # 构造光栅平面中心化的坐标
    x_in = np.linspace(-Nx/2*dx, (Nx/2-1)*dx, Nx)
    y_in = np.linspace(-Ny/2*dy, (Ny/2-1)*dy, Ny)
    X_in, Y_in = np.meshgrid(x_in, y_in)
    
    # 预乘二次相位因子
    Q1 = np.exp(1j * k / (2 * z) * (X_in**2 + Y_in**2))
    U0 = E0 * Q1
    
    # 进行二维 FFT
    U0_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(U0))) * (dx * dy)
    
    # 计算频域坐标
    fx = np.fft.fftshift(np.fft.fftfreq(Nx, dx))
    fy = np.fft.fftshift(np.fft.fftfreq(Ny, dy))
    FX, FY = np.meshgrid(fx, fy)
    # 探测器平面坐标
    x_out = wavelength * z * FX
    y_out = wavelength * z * FY
    
    # 后乘二次相位因子
    Q2 = np.exp(1j * k / (2 * z) * (x_out**2 + y_out**2))
    
    # Fresnel 衍射公式中的比例因子
    E_out = np.exp(1j * k * z) / (1j * wavelength * z) * Q2 * U0_fft
    return E_out, x_out, y_out

# 进行传播计算
E_det, X_det, Y_det = fresnel_propagation_2d(E_after, dx, dy, wavelength, d_prop)

# --------------------
# 4. 绘制探测器平面上的二维衍射强度图
# --------------------
I_det = np.abs(E_det)**2

plt.figure(figsize=(6,5))
plt.imshow(I_det, extent=[X_det.min()*1e3, X_det.max()*1e3, Y_det.min()*1e3, Y_det.max()*1e3],
           cmap='inferno', origin='lower')
plt.xlabel('x [mm]')
plt.ylabel('y [mm]')
plt.title('探测器平面衍射强度分布')
plt.colorbar(label='强度 (归一化)')
plt.tight_layout()
plt.show()
