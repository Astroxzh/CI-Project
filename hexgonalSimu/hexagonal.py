import numpy as np
import matplotlib.pyplot as plt

def fresnel_propagation(U0, dx, wavelength, z):
    """
    使用 Fresnel 衍射公式传播场 U0
    U0: 初始场 (二维复数数组)
    dx: 坐标网格间距 (假设 x,y 均匀)
    wavelength: 波长 (m)
    z: 传播距离 (m)
    """
    N = U0.shape[0]
    k = 2 * np.pi / wavelength
    # 构造源平面坐标（以数组中心为原点）
    x = np.arange(-N/2, N/2) * dx
    y = x
    X, Y = np.meshgrid(x, y)
    
    # 源平面二次相位因子
    Q1 = np.exp(1j * k/(2*z) * (X**2 + Y**2))
    U0_mod = U0 * Q1
    
    # FFT 计算
    U1 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(U0_mod))) * dx**2
    
    # 频率坐标
    fx = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    fy = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    FX, FY = np.meshgrid(fx, fy)
    
    # 傅里叶域二次相位因子
    Q2 = np.exp(1j * np.pi * wavelength * z * (FX**2 + FY**2))
    
    U = np.exp(1j * k * z) / (1j * wavelength * z) * Q2 * U1
    return U

# -------------------------------
# 参数设置
# -------------------------------
pitch = 200e-9          # 六角形晶格中心间距 200 nm
# 最密堆积六角形晶格基矢：
#   v1 = [pitch, 0]
#   v2 = [pitch/2, pitch*sqrt(3)/2]
phi_max = np.pi         # 最大相位调制
R_hemi = 60e-9          # 半球形调制区域半径 75 nm

# 入射波能量 6.2 keV 对应的波长约 0.2 nm
wavelength = 0.2e-9     # 0.2 nm = 0.2e-9 m
k = 2 * np.pi / wavelength

# 传播距离设置
d_source = 4.5e-3       # 点源到光栅平面距离 4.5 mm
z_obj = 0.955e-3        # 物面距光栅平面 0.955 mm
z_img = 7.336           # 像面距物面 7.336 m

# -------------------------------
# 建立模拟区域（单位：m）
# -------------------------------
N = 1024
L_extent = 10e-6        # 模拟区域边长 10 µm
x = np.linspace(-L_extent/2, L_extent/2, N)
y = np.linspace(-L_extent/2, L_extent/2, N)
X, Y = np.meshgrid(x, y)

# -------------------------------
# 计算最密堆积六角形晶格中心
# -------------------------------
# 对于最密堆积六角形晶格，晶格中心可写为：
#   X_center = m * pitch + n * (pitch/2)
#   Y_center = n * (pitch*sqrt(3)/2)
# 其中 m, n 可由下式近似得到：
m_float = X / pitch - Y / (np.sqrt(3) * pitch)
n_float = 2 * Y / (np.sqrt(3) * pitch)
m_int = np.rint(m_float)
n_int = np.rint(n_float)
Xc = m_int * pitch + n_int * (pitch/2)
Yc = n_int * (pitch * np.sqrt(3)/2)

# -------------------------------
# 局部坐标与半球形相位调制
# -------------------------------
X_local = X - Xc
Y_local = Y - Yc
R2 = X_local**2 + Y_local**2
# 在 r<=R_hemi 内采用半球形调制，其相位为：
#   φ(r) = φ_max * sqrt(1 - (r/R_hemi)^2)
phi = np.where(R2 <= R_hemi**2, phi_max * np.sqrt(1 - R2 / R_hemi**2), 0)
# 构造透射函数（纯相位调制）
T = np.exp(1j * phi)

# -------------------------------
# 模拟球面波入射
# -------------------------------
# 假设点源位于 (0, 0, -d_source)，光栅平面位于 z=0
R_inc = np.sqrt(X**2 + Y**2 + d_source**2)
E_inc = np.exp(1j * k * R_inc) / R_inc

# 光栅平面的场
U_aperture = T * E_inc

# -------------------------------
# Fresnel 传播：光栅平面 -> 物面 -> 像面
# -------------------------------
dx = x[1] - x[0]

# 传播到物面（0.955 mm）
U_obj = fresnel_propagation(U_aperture, dx, wavelength, z_obj)
I_obj = np.abs(U_obj)**2

# 传播到像面（从物面再传播 7.336 m）
U_img = fresnel_propagation(U_obj, dx, wavelength, z_img)
I_img = np.abs(U_img)**2

# -------------------------------
# 计算物面和像面坐标（基于 FFT 频率）
# -------------------------------
fx = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
x_obj = wavelength * z_obj * fx
y_obj = x_obj  # 对称
x_img = wavelength * z_img * fx
y_img = x_img

# -------------------------------
# 绘图显示
# -------------------------------
plt.figure(figsize=(12, 5))

# 显示物面图样（对数强度）
plt.subplot(1, 2, 1)
plt.imshow(np.log(I_obj + 1), extent=[x_obj[0]*1e6, x_obj[-1]*1e6,
                                       y_obj[0]*1e6, y_obj[-1]*1e6],
           cmap='inferno', origin='lower')
plt.title("物面衍射图样 (log intensity)")
plt.xlabel("x (µm)")
plt.ylabel("y (µm)")
plt.colorbar(label="log(Intensity)")

# 显示像面图样（对数强度）
plt.subplot(1, 2, 2)
plt.imshow(np.log(I_img + 1), extent=[x_img[0]*1e3, x_img[-1]*1e3,
                                       y_img[0]*1e3, y_img[-1]*1e3],
           cmap='inferno', origin='lower')
plt.title("像面衍射图样 (log intensity)")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.colorbar(label="log(Intensity)")

plt.tight_layout()
plt.show()
