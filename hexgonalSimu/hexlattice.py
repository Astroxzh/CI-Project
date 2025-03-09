import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 参数设置
# -------------------------------
wavelength = 13.5e-9      # 波长 13.5 nm
k = 2 * np.pi / wavelength

d_source = 20e-3          # 点源到光栅距离 20 mm
L = 1.0                   # 观察屏（远场）距离，取1 m

# -------------------------------
# 模拟区域（光栅平面）的设置（单位：米）
# 为了包含足够多个周期，选取较宽的区域
N = 1024                  # 数组尺寸（建议选用2的幂便于FFT）
x_min = -500e-6           # -500 µm
x_max = 500e-6            # 500 µm
y_min = -500e-6           # -500 µm
y_max = 500e-6            # 500 µm

x = np.linspace(x_min, x_max, N)
y = np.linspace(y_min, y_max, N)
X, Y = np.meshgrid(x, y)
dx = x[1] - x[0]          # 网格间距

# -------------------------------
# 构造一维二元相位光栅
# -------------------------------
# 100线对每毫米意味着周期 period = 10 µm，50%占空比
period = 10e-6            # 10 µm
duty_cycle = 0.5

# 根据 x 坐标决定相位：当 x 模 period 小于 period*duty_cycle 时相位为 0，否则为 π
phase_grating = np.where(np.mod(X, period) < (duty_cycle * period), 0, np.pi)
# 对应的透射率：T = exp(i * phase)
T = np.exp(1j * phase_grating)

# -------------------------------
# 模拟球面波入射
# -------------------------------
# 假设点源位于 (0,0,-d_source)，光栅位于 z = 0
# 每个点处的距离 R = sqrt(x^2 + y^2 + d_source^2)
R = np.sqrt(X**2 + Y**2 + d_source**2)
E_inc = np.exp(1j * k * R) / R  # 球面波

# 透过光栅后的场：二元相位调制
U_aperture = T * E_inc

# -------------------------------
# 远场衍射（Fraunhofer衍射）
# -------------------------------
# 远场衍射图样与光栅平面场的傅里叶变换成正比
U_far = np.fft.fftshift(np.fft.fft2(U_aperture))
I_far = np.abs(U_far)**2

# 计算傅里叶域坐标（单位：1/m），再换算到观察屏物理坐标：
# x_obs = λ·L·f_x, y_obs = λ·L·f_y
fx = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
fy = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
x_obs = wavelength * L * fx
y_obs = wavelength * L * fy

# -------------------------------
# 绘图显示
# -------------------------------
plt.figure(figsize=(12, 5))

# 显示二元相位光栅的相位分布（0 和 π）
plt.subplot(1, 2, 1)
plt.imshow(phase_grating, cmap='gray', extent=[x_min*1e6, x_max*1e6, y_min*1e6, y_max*1e6])
plt.title("二元相位光栅 (0, π)")
plt.xlabel("x (µm)")
plt.ylabel("y (µm)")

# 显示远场衍射图样（采用对数刻度显示）
plt.subplot(1, 2, 2)
plt.imshow(np.log(I_far + 1), cmap='inferno',
           extent=[x_obs.min()*1e3, x_obs.max()*1e3, y_obs.min()*1e3, y_obs.max()*1e3])
plt.title("远场衍射图样 (对数刻度)")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")

plt.tight_layout()
plt.show()