import torch
from PIL import Image
import numpy as np
from DataEnhance import get_enforce_image
from ImageUtils import draw_image
from NetWork import ScaleNet
from ScaleConfig import scale_config


# 加载权重
def load_weights(model, path):
    print('Loading weights into state dict...')
    model_dict = model.state_dict()
    pre_trained_dict = torch.load(path, map_location=scale_config.device)
    pre_trained_dict = {k: v for k, v in pre_trained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pre_trained_dict)
    model.load_state_dict(model_dict)
    print('Finished!')
    return model


def pred(path=""):
    image = Image.open(path)
    image, padding, scale = get_enforce_image(image, input_shape=scale_config.input_shape)
    draw_image(image=image)

    w, h = image.size
    input_image = image.resize((int(w / 2), int(h / 2)), Image.BICUBIC)
    draw_image(image=input_image)

    output_image = input_image.resize((w, h), Image.BICUBIC)
    draw_image(image=output_image)

    input_image = np.array(input_image).transpose((2, 0, 1))
    input_image = input_image / 255.0
    input_image = torch.from_numpy(input_image).float().to(scale_config.device)
    input_image = input_image.unsqueeze(dim=0)

    output_image = np.array(output_image).transpose((2, 0, 1))
    output_image = output_image / 255.0
    output_image = torch.from_numpy(output_image).float().to(scale_config.device)

    net = ScaleNet()
    net = net.to(scale_config.device)

    net = load_weights(net, './params_20.pth')

    pred = net(input_image)
    pred = pred.squeeze(dim=0)
    output_image = pred + output_image
    output_image = output_image * 255.0


    draw_image(image=output_image)


if __name__ == '__main__':
    pred("./test.jpg")
