# d = np.sqrt((deta*L0/lambda_)**2-0.25*(L0**2))                          # 传输距离，单位米
# f = d

import numpy as np
import time
import matplotlib.pyplot as plt
from fanc import *


 
# 参数设置
lambda_ = 1.064e-6              # 波长，单位转换为米
k = 2 * np.pi / lambda_         # 波矢大小
N0 = 1024                       # 入射面采样点数
N = 512                         # 出射面采样点数
L0 = 60*1e-3                    # 入射面尺寸，单位转换为米
L = 60*1e-3                    # 出射面尺寸，单位转换为米
w = 1*1e-3
n = 6
ll = 5*1e-3
deta = L0/N0
d = np.sqrt((deta*L0/lambda_)**2-0.25*(L0**2))                          # 传输距离，单位米
f = d
 
# 离散化入射平面和出射面
x0 = np.linspace(-L0/2, L0/2, N0)
y0 = np.linspace(-L0/2, L0/2, N0)
X0, Y0 = np.meshgrid(x0, y0)
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)
dx0, dy0 = np.gradient(x0)[0], np.gradient(y0)[0]
T = {}
 
# 生成光源
centers = [(ll*1/2, ll*np.sqrt(3)/2), (-ll*1/2, ll*np.sqrt(3)/2), (ll*1/2, -ll*np.sqrt(3)/2), (-ll*1/2, -ll*np.sqrt(3)/2), (ll,0), (-ll,0)]
U_total = generate_gaussian_sources(X0, Y0, centers, w)
U_total2 = generate_gaussian_sources(X0, Y0, centers, w)
# 透镜调制
U_after_lens = apply_lens_phase_factor(U_total, X0, Y0, f, lambda_)
 
# 利用矩阵乘法计算出射面光场复振幅
start_time = time.time()
u1 = matrix_method(U_after_lens, k, d, x, y, x0, y0)
checkpoint_time = time.time()
T['matrix_method'] = checkpoint_time-start_time
I1 = np.abs(u1)**2
I1 = I1 / I1.max()
# 绘制
plt.figure()
plt.imshow(I1)
plt.title('matrix_method')
plt.colorbar()
 
 
# 利用积分法计算出射面光场复振幅
#u2 = integrate_method(U_after_lens, d, lambda_, X0, Y0, X, Y)
#I2 = np.abs(u2)**2
#I2 = I2 / I2.max()
# 绘制
#plt.figure()
#plt.imshow(I2)
#plt.title('integrate_method')
#plt.colorbar()
 
 
# 两次傅里叶方法
start_time = time.time()
u5 = d_fft_method(U_after_lens, d, lambda_, dx0, dy0)
checkpoint_time = time.time()
T['d_fft_method'] = checkpoint_time-start_time
I5 = np.abs(u5)**2
I5 = I5 / I5.max()
# 绘制
plt.figure()
plt.imshow(I5)
plt.title('d_fft_method')
plt.colorbar()
 
# 三次傅里叶方法
start_time = time.time()
u3 = t_fft_method(U_after_lens, d, lambda_, dx0, dy0, L0)
checkpoint_time = time.time()
T['t_fft_method'] = checkpoint_time-start_time
I3 = np.abs(u3)**2
I3 = I3 / I3.max()
# 绘制
plt.figure()
plt.imshow(I3)
plt.title('t_fft_method')
plt.colorbar()
 
# 单次傅里叶
start_time = time.time()
u4 = s_fft_method(U_after_lens, d, lambda_, X0, Y0)
checkpoint_time = time.time()
T['s_fft_method'] = checkpoint_time-start_time
I4 = np.abs(u4)**2
I4 = I4 / I4.max()
# 绘制
plt.figure()
plt.imshow(I4)
plt.title('s_fft_method')
plt.colorbar()
 
 
# 光源绘制
plt.figure()
plt.title('origin_img')
plt.imshow(np.abs(U_total))
plt.colorbar()
 
 
plt.show()
 
print(T)