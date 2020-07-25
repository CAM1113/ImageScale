from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


# 输入image：C * H * W,numpy数据
# boxes为框的列表，每个框提供左上、右下
# labels标签列表
# scores得分列表
def draw_image(image, is_save=False, save_path="./temp.jpg"):
    # np.transpose( xxx,  (2, 0, 1))   # 将 C x H x W 转化为 H x W x C
    if type(image) == Image.Image:
        image = np.array(image)
    # 通道调整
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))

    fig, ax = plt.subplots(1)
    ax.imshow(image)
    plt.axis('off')
    if is_save:
        plt.savefig(save_path)
    plt.show()
