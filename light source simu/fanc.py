import numpy as np
 
from tqdm import tqdm
 
from scipy.fftpack import *
 
 
def generate_gaussian_sources(xx, yy, centers, sigma_x=1, sigma_y=None):
    if sigma_y is None:
        sigma_y = sigma_x
 
    # 初始化叠加后的高斯光源矩阵
    gaussian_sources = np.zeros(xx.shape)
 
    # 计算每个高斯分布并叠加
    for center in centers:
        center_x, center_y = center
 
        # 高斯分布
        exponent = - ((xx - center_x) ** 2 / sigma_x ** 2 + (yy - center_y) ** 2 / sigma_y ** 2)
        gaussian_source = np.exp(exponent)
 
        # 叠加到总光源矩阵
        gaussian_sources += gaussian_source
 
    return gaussian_sources
 
def apply_lens_phase_factor(U, X, Y, f, lambda_):
    k = 2 * np.pi / lambda_
    phase_factor = np.exp(-1j * k * (X**2 + Y**2) / (2 * f))
    return U * phase_factor
 
def matrix_method(U, k, d, x, y, x0, y0):
 
    X0, Y0 = np.meshgrid(x0, y0)
    # 构建矩阵Mx, My和M
    Mx = np.exp(-1j * k / d * np.outer(x, x0))
    My = np.exp(-1j * k / d * np.outer(y0, y))
    M = U * np.exp(1j * k * (X0 ** 2 + Y0 ** 2) / (2 * d))
 
    # 利用矩阵乘法计算出射面光场复振幅
    return Mx @ M @ My
 
def d_fft_method(U, d, lambda_, dx, dy):
 
    k = 2 * np.pi / lambda_
 
    # 轴向传播在频率域的传递函数
    fx = np.linspace(-1/(2*dx), 1/(2*dx), U.shape[0])
    fy = np.linspace(-1/(2*dy), 1/(2*dy), U.shape[1])[::-1]
    FX, FY = np.meshgrid(fx, fy)
    t = 1j*k*d
    uxs = np.sqrt(1 - (lambda_*FX)**2 - (lambda_*FY)**2)
    H = np.exp(t * uxs)
 
    # 将光场变换到频率空间
    U1=fftshift(fft2(U))
    U2=H*U1
    u2=ifft2(ifftshift(U2))
 
    return u2
 
def t_fft_method(U, d, lambda_, dx, dy, L):
 
    k = 2 * np.pi / lambda_
 
    # 轴向传播在频率域的传递函数
    x = np.linspace(-L / 2, L / 2, U.shape[0])
    y = np.linspace(-L / 2, L / 2, U.shape[1])
    X, Y = np.meshgrid(x, y)
    H = np.exp(1j * k / (2*d) * (X ** 2 + Y ** 2))
 
    # 将光场变换到频率空间
    H_freq = fft2(H)
    U_freq = fft2(U)
 
    # 应用传递函数
    U_freq_after_TTF = U_freq * H_freq
 
    # 将光场变换回实空间
    U_z = np.exp(1j*k*d)/1j/lambda_/d * fftshift(ifft2(U_freq_after_TTF))
 
    return U_z
 
def s_fft_method(U, d, lambda_, X, Y):
 
    k = 2 * np.pi / lambda_
 
    Lx = X.shape[0] * lambda_ * d / (X[-1,-1]-X[-0, -0])
    Ly = Y.shape[0] * lambda_ * d / (Y[-1,-1]-Y[-0, -0])
    x1 = np.linspace(-Lx / 2, Lx / 2, X.shape[0])
    y1 = np.linspace(-Ly / 2, Ly / 2, Y.shape[0])
    X1, Y1 = np.meshgrid(x1, y1)
 
    F0 = np.exp(1j * k * d) / (1j * lambda_ * d) * np.exp(1j * k / 2 / d * (X1 ** 2 + Y1 ** 2))
    F = np.exp(1j * k / 2 / d * (X ** 2 + Y ** 2))
    F = fftshift(fft2(U*F))
    return F0 * F
 
    return U
 
def integrate_method(U, d, lambda_, Xin, Yin, Xout, Yout):
 
    dx = Xin[1] - Xin[0]
    dy = dx
 
    # 计算波矢
    k = 2 * np.pi / lambda_
    phase_shift = k / (2 * d)
 
    # 初始化传播后的场分布
    u_propagated = np.zeros((Xout.shape[1], Yout.shape[0]), dtype=np.complex128)
 
    # 对整个空间进行积分以计算传播后的场分布
    for i in tqdm(range(Xout.shape[1]), desc="Outer Loop"):
        for j in range(Xout.shape[0]):
            -1j / lambda_ / d * np.exp(1j * k * d) * sum(sum(U * np.exp(1j * k / 2 / d * ((Xout[i, j] - Xin) ** 2 + (Yout[i, j] - Yin) ** 2))))
 
    return u_propagated