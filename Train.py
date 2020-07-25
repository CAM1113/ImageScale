import torch
import numpy as np
from ImageDataSet import ImageDataSet
from Loss import ScaleLoss
from NetWork import ScaleNet
from ScaleConfig import scale_config
from ImageDataSet import ImageDataSet, dataset_collate
from torch.utils.data import DataLoader
from torch import optim

batch_size = 12
dataset = ImageDataSet(data_path='./train.txt', input_shape=scale_config.input_shape)
data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True, drop_last=True,
                         shuffle=True, collate_fn=dataset_collate)


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


def train():
    total_epoch = 50
    lr = 1e-3
    criterion = ScaleLoss()
    net = ScaleNet()
    net = net.to(scale_config.device)
    net = load_weights(net, './params.pth')
    optimizer = optim.Adam(net.parameters(), lr=lr)

    for epoch in range(total_epoch):
        print(epoch)
        if epoch % 20 == 0:
            lr = lr / 10
            optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.001)

        for data in data_loader:
            optimizer.zero_grad()
            target_images, scale_images, enlarge_images = data
            target_images.requires_grad = False
            scale_images.requires_grad = False
            enlarge_images.requires_grad = False
            target_images = target_images.float().to(scale_config.device)
            scale_images = scale_images.float().to(scale_config.device)
            enlarge_images = enlarge_images.float().to(scale_config.device)

            pred = net(scale_images)
            loss = criterion(pred, enlarge_images, target_images)
            print("loss.item() = {}".format(loss.cpu().item()))
            loss.backward()

            optimizer.step()
        torch.save(net.state_dict(), './params_{}.pth'.format(epoch))


if __name__ == '__main__':
    train()
