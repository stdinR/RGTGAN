import os
import cv2
import numpy as np

def add_gaussian_noise(image, mean=0, sigma=25):
    """添加高斯噪声"""
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = np.clip(image + gauss, 0, 255)
    return noisy.astype(np.uint8)

def process_images(input_folder, output_folder, noise_type='gaussian', noise_params=None):
    """处理文件夹中的所有图像"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = os.listdir(input_folder)

    for img_file in image_files:
        input_path = os.path.join(input_folder, img_file)
        output_path = os.path.join(output_folder, img_file)

        # 读取图像
        image = cv2.imread(input_path)

        # 根据选择的噪声类型添加噪声
        if noise_type == 'gaussian':
            noisy_image = add_gaussian_noise(image, **noise_params)
        elif noise_type == 'poisson':
            # 添加泊松噪声
            noisy_image = np.random.poisson(image / 255.0 * noise_params['lambd']) * 255.0
            noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

        # 保存带有噪声的图像
        cv2.imwrite(output_path, noisy_image)


# 设置输入和输出文件夹
input_folder = '/home/ziming/RRSGAN/RRSGAN-main/dataset/train/train_Kaggle2_noise/LR_ori'
output_folder = '/home/ziming/RRSGAN/RRSGAN-main/dataset/train/train_Kaggle2_noise/LR'

# 设置噪声参数
noise_params = {'mean': 0, 'sigma': 5}  # 对于高斯噪声
# noise_params = {'lambd': 5}  # 对于泊松噪声

# 处理图像
process_images(input_folder, output_folder, noise_type='gaussian', noise_params=noise_params)