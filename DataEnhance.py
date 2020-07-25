import numpy as np
from PIL import ImageEnhance
from PIL import Image


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


# 重新按比例设置图片大小，图片大小input_shape，空白区域用灰条补齐,
# padding代表是否输出填充灰条的大小
def reshape_image(image, input_shape=None):
    iw, ih = image.size
    w, h = input_shape
    if iw > ih:
        scale = w / iw
    else:
        scale = h / ih

    nw = int(iw * scale)
    nh = int(ih * scale)
    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    dx = abs(w - nw) // 2
    dy = abs(h - nh) // 2
    new_image.paste(image, (dx, dy))
    image = new_image

    padding = (dx, dy, dx, dy)
    return image, padding, scale


# 垂直翻转图片
def td_flip_image(image, boxes):
    w, h = image.size
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
    return image, boxes


# 亮度
def bright_image(img):
    brightness = rand() + 0.5
    enh_bri = ImageEnhance.Brightness(img)
    image_brightened = enh_bri.enhance(brightness)
    return image_brightened


# 色度增强
def color_image(img):
    enh_col = ImageEnhance.Color(img)
    color = rand() + 0.5
    image_colored = enh_col.enhance(color)
    return image_colored


# 对比度增强
def contrast_image(img):
    enh_con = ImageEnhance.Contrast(img)
    contrast = rand() + 0.5
    image_contrasted = enh_con.enhance(contrast)
    return image_contrasted


# 锐度增强
def sharpness_image(img):
    enh_con = ImageEnhance.Sharpness(img)
    sharpness = rand() * 2 + 0.5
    image_sharped = enh_con.enhance(sharpness)
    return image_sharped


# boxes:左上 + 右下
def get_enforce_image(image, input_shape):
    '''数据增强的随机预处理'''
    # 50%的几率水平翻转
    if rand() < .5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # 50%几率修改亮度
    if rand() < .5:
        image = bright_image(image)

    # 50%几率修改色度
    if rand() < .5:
        image = color_image(image)

    # 50%几率修改对比度
    if rand() < .5:
        image = contrast_image(image)

    # 50%几率修改锐度
    if rand() < .5:
        image = sharpness_image(image)

    image = reshape_image(image, input_shape)
    return image
