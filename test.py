import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate


# 定义网格
N = 1000  # 网格点数
x = np.linspace(-50, 50, N)
y = np.linspace(-50, 50, N)
X, Y = np.meshgrid(x, y)

# 周期设置
period = 5

# 通过对坐标取模，模拟正弦正负区间，构造二值（binary）光栅
# 当 x 在每周期前半段时取正，后半段时取负；y 同理
sign_x = np.where(np.mod(X, period) < period/2, 1, -1)
# sign_y = np.where(np.mod(Y, period) < period/2, 1, -1)
# 二值光栅：当 sign_x 与 sign_y 同号时为1，不同号时为0
binary_grating = (sign_x > 0).astype(float)

# 旋转光栅：旋转角度为 120° 和 240°，保持原图尺寸
# binary_grating_60 = rotate(binary_grating, 60, reshape=False)
binary_grating_120 = rotate(binary_grating, 60, reshape=False)
# binary_grating_180 = rotate(binary_grating, 180, reshape=False)
binary_grating_240 = rotate(binary_grating, 240, reshape=False)
# binary_grating_300 = rotate(binary_grating, 300, reshape=False)

binary_grating_120[binary_grating_120 > 0.9] = 1
binary_grating_120[binary_grating_120 < 0.1] = 0
binary_grating_240[binary_grating_240 > 0.9] = 1
binary_grating_240[binary_grating_240 < 0.1] = 0
# binary_grating_60[binary_grating_120 > 0.9] = 1
# binary_grating_60[binary_grating_120 < 0.1] = 0
# binary_grating_180[binary_grating_240 > 0.9] = 1
# binary_grating_180[binary_grating_240 < 0.1] = 0
# binary_grating_300[binary_grating_120 > 0.9] = 1
# binary_grating_300[binary_grating_120 < 0.1] = 0


total_grating = binary_grating + binary_grating_120 + binary_grating_240 
# total_grating[total_grating < 3.0] = 0
# total_grating[total_grating > 2.0] = 1

total_grating_crop = total_grating[250:750,250:750]

total_grating_crop_fre = np.fft.fftshift(np.fft.fft2(total_grating_crop))


plt.imshow(total_grating_crop)
plt.figure()
plt.imshow((np.abs(total_grating_crop_fre)))
plt.show()

# # 绘制三个光栅
# plt.figure(figsize=(12, 4))

# plt.subplot(1, 3, 1)
# plt.imshow(binary_grating, extent=[x.min(), x.max(), y.min(), y.max()], cmap='gray')
# plt.title('原始二值光栅')
# plt.xlabel('x')
# plt.ylabel('y')

# plt.subplot(1, 3, 2)
# plt.imshow(binary_grating_120, extent=[x.min(), x.max(), y.min(), y.max()], cmap='gray')
# plt.title('旋转 120°')
# plt.xlabel('x')
# plt.ylabel('y')

# plt.subplot(1, 3, 3)
# plt.imshow(binary_grating_240, extent=[x.min(), x.max(), y.min(), y.max()], cmap='gray')
# plt.title('旋转 240°')
# plt.xlabel('x')
# plt.ylabel('y')

# plt.tight_layout()
# plt.show()
