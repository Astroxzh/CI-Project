import numpy as np
import matplotlib.pyplot as plt

def fresnel_propagation(U0, dx, wavelength, z):
    """
    使用 Fresnel 衍射公式进行传播
    U0: 初始场
    dx: 网格间距
    wavelength: 波长（单位：m）
    z: 传播距离（单位：m）
    """
    N = U0.shape[0]
    k = 2 * np.pi / wavelength
    # 构造坐标系（以数组中心为原点）
    x = np.arange(-N/2, N/2)*dx
    y = x
    X, Y = np.meshgrid(x, y)
    # 源平面上的二次相位因子
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
pitch = 200e-9        # 六角形晶格中心间距 200 nm
# 最密堆积六角形晶格基矢： v1 = [pitch, 0], v2 = [pitch/2, pitch*sqrt(3)/2]
phi_max = np.pi       # 最大相位调制
R_hemi = 50e-9        # 半球形调制区域半径改为 50 nm

# 球面波参数
d_source = 4.5e-3     # 点源到光栅平面距离 4.5 mm
wavelength = 13.5e-9  # 波长 13.5 nm
k = 2 * np.pi / wavelength

# 传播距离设置
z_obj = 0.955e-3      # 物面距光栅平面 0.955 mm
z_det = 7.336         # 探测器距物面 7.336 m

# -------------------------------
# 建立模拟区域（单位：m）
# -------------------------------
N = 1024
L_extent = 10e-6      # 模拟区域边长 10 µm
x = np.linspace(-L_extent/2, L_extent/2, N)
y = np.linspace(-L_extent/2, L_extent/2, N)
X, Y = np.meshgrid(x, y)

# -------------------------------
# 计算最密堆积六角形晶格中心
# -------------------------------
# 对于最密堆积的六角形晶格，晶格中心可由：
#   X_center = m * pitch + n * (pitch/2)
#   Y_center = n * (pitch*sqrt(3)/2)
# 得到，其中 m, n 由：
m_float = X / pitch - Y / (np.sqrt(3)*pitch)
n_float = 2 * Y / (np.sqrt(3)*pitch)
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
# 半球形调制：在 r<=R_hemi 内
phi = np.where(R2 <= R_hemi**2, phi_max * np.sqrt(1 - R2 / R_hemi**2), 0)
# 构造透射函数（纯相位调制）
T = np.exp(1j * phi)

# -------------------------------
# 模拟球面波入射
# -------------------------------
# 点源位于 (0,0,-d_source)，光栅平面 z=0
R_inc = np.sqrt(X**2 + Y**2 + d_source**2)
E_inc = np.exp(1j * k * R_inc) / R_inc

plt.figure()
plt.imshow(np.angle(E_inc), extent=[x[0]*1e6, x[-1]*1e6, y[0]*1e6, y[-1]*1e6],
           cmap='twilight', origin='lower')


# 光栅平面的场
U_aperture = T * E_inc
plt.figure()
plt.imshow(np.angle(U_aperture), extent=[x[0]*1e6, x[-1]*1e6, y[0]*1e6, y[-1]*1e6],
           cmap='twilight', origin='lower')

# -------------------------------
# 两步 Fresnel 传播
# -------------------------------
# 1. 从光栅平面传播到物面（z_obj = 0.955 mm）
U_obj = fresnel_propagation(U_aperture, x[1]-x[0], wavelength, z_obj)
# 2. 从物面传播到探测器（z_det = 7.336 m）
U_det = fresnel_propagation(U_obj, x[1]-x[0], wavelength, z_det)
I_det = np.abs(U_det)**2

# -------------------------------
# 计算探测器平面坐标（用于显示）
# -------------------------------
dx = x[1]-x[0]
fx = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
x_det = wavelength * z_det * fx
fy = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
y_det = wavelength * z_det * fy

# -------------------------------
# 绘图显示
# -------------------------------
plt.figure(figsize=(12, 5))

# 显示六角形衍射光栅的相位分布
plt.subplot(1, 2, 1)
plt.imshow(np.angle(T), extent=[x[0]*1e6, x[-1]*1e6, y[0]*1e6, y[-1]*1e6],
           cmap='twilight', origin='lower')
plt.title("六角形衍射光栅相位分布\n（半球形调制, R=50 nm）")
plt.xlabel("x (µm)")
plt.ylabel("y (µm)")
plt.colorbar(label="相位 (rad)")

# 显示探测器处远场衍射图样（对数强度）
plt.subplot(1, 2, 2)
plt.imshow(np.log(I_det + 1), extent=[x_det[0]*1e3, x_det[-1]*1e3, y_det[0]*1e3, y_det[-1]*1e3],
           cmap='inferno', origin='lower')
plt.title("探测器处远场衍射图样 (log scale)")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.colorbar(label="log(Intensity)")
plt.tight_layout()
plt.show()



import h5py
import matplotlib.pyplot as plt

def probe_read(filepath=r'Papercode\reconstructions\e17965_1_00678_ptycho_reconstruction.h5'):
    with h5py.File(filepath, 'r') as f:
        # print("Keys in the file:", list(f.keys()))
        dataset = f['probe']
        probe = dataset[:]  # 读取整个数据集
        # there are two probe modes
        return probe
    
    def forward_model(obj_downSampled, probe):
    '''
        the forward model generates the update diffraction field from the down sampled obj
        in this simulation assumes only one probe 
    '''
    obj_fre = fft.fftshift(fft.fft2(obj_downSampled))
    obj_frepad = pad_array(obj_fre, probe)
    obj_pad = fft.ifft2(fft.ifftshift(obj_frepad))
    update_diff_pattern = np.abs(fft.fftshift(fft.fft2((probe[0,:,:] * 0.756 + probe[1,:,:] * 0.244) * obj_pad))) ** 2

    return update_diff_pattern