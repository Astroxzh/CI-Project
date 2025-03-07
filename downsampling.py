import numpy as np

def downsample_complex_image(image, n):
    """
    空域下采样算法，将每 n*n 像素的块下采样为一个新像素。
    对于复值图像，通过对每个块取平均值来实现下采样。

    参数：
    image : np.array, shape=(H, W)
        待下采样的复值图像。
    n : int
        下采样因子，每 n*n 像素对应于一个新像素。

    返回：
    np.array
        下采样后的图像，其尺寸为 (H//n, W//n)。
    """
    H, W = image.shape
    # 裁剪图像，使其尺寸能被 n 整除
    H_crop = (H // n) * n
    W_crop = (W // n) * n
    image_cropped = image[:H_crop, :W_crop]

    # 重塑图像为 (H_crop//n, n, W_crop//n, n)
    reshaped = image_cropped.reshape(H_crop//n, n, W_crop//n, n)
    
    # 对每个 n*n 块取平均值，下采样到新图像中
    downsampled = reshaped.mean(axis=(1, 3))
    return downsampled
#%%
