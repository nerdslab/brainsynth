"""dataset.py"""

import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


def is_power_of_2(num):
    return ((num & (num - 1)) == 0) and num != 0


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        path = self.imgs[index][0]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img


class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


def return_data(args):
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.image_size
    #assert image_size == 64, 'currently only image size of 64 is supported'

    if name.lower() == '3dchairs':
        root = os.path.join(dset_dir, '3DChairs')
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(), ])
        train_kwargs = {'root': root, 'transform': transform}
        dset = CustomImageFolder

    elif name.lower() == 'celeba':
        root = os.path.join(dset_dir, 'CelebA')
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(), ])
        train_kwargs = {'root': root, 'transform': transform}
        dset = CustomImageFolder

    elif name.lower() == 'dsprites':
        root = os.path.join(dset_dir, 'dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        if not os.path.exists(root):
            import subprocess
            print('Now download dsprites-dataset')
            subprocess.call(['./download_dsprites.sh'])
            print('Finished')
        data = np.load(root, encoding='bytes')
        data = torch.from_numpy(data['imgs']).unsqueeze(1).float()
        train_kwargs = {'data_tensor': data}
        dset = CustomTensorDataset

    elif name.lower() == 'mnist':
        root = os.path.join(dset_dir, 'mnist-dataset')
        if not os.path.exists(root):
            print("mnist data is not here")
        from mlxtend.data import loadlocal_mnist
        test_X, test_y = loadlocal_mnist(images_path=os.path.join(root, 't10k-images-idx3-ubyte'),
                                         labels_path=os.path.join(root, 't10k-labels-idx1-ubyte'))
        train_X, train_y = loadlocal_mnist(images_path=os.path.join(root, 't10k-images-idx3-ubyte'),
                                           labels_path=os.path.join(root, 't10k-labels-idx1-ubyte'))

        data = torch.stack([torch.from_numpy(test_X),torch.from_numpy(train_X)],dim=0)
        data = np.reshape(data, (-1,1,28,28))
        print("mnist_data_shape:{}".format(data.shape))
        data = data.float()/255
        train_kwargs = {'data_tensor': data}
        dset = CustomTensorDataset

    elif name.lower() == 'brain':
        root = os.path.join(dset_dir, 'brain-dataset')
        if not os.path.exists(root):
            print("brain data is not here")
        file_path = os.path.join(root, "layer6_norm_brain_dataset_np_float32.npy")
        data = np.load(file_path)
        data = data[:,4:452,:]
        data = np.reshape(data, (-1, 1, 448, 320))
        data = torch.from_numpy(data)
        print("brain_data_shape:{}".format(data.shape))
        train_kwargs = {'data_tensor': data}
        dset = CustomTensorDataset

    elif name.lower() == 'brain_test':
        root = os.path.join(dset_dir, 'brain-dataset')
        if not os.path.exists(root):
            print("brain data is not here")
        file_path = os.path.join(root, "test_brain_dataset_np_float32.npy")
        data = np.load(file_path)
        data = data[:, 4:452, :]
        data = np.reshape(data, (-1, 1, 448, 320))
        data = torch.from_numpy(data)
        print("brain_data_shape:{}".format(data.shape))
        train_kwargs = {'data_tensor': data}
        dset = CustomTensorDataset


    else:
        raise NotImplementedError

    train_data = dset(**train_kwargs)
    if name.lower() == 'brain_test':
        train_loader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  drop_last=False)
    else:
        train_loader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  drop_last=True)



    data_loader = train_loader

    return data_loader


def return_data_sp(dataset):
    name = dataset
    dset_dir = "data"
    batch_size = 64
    num_workers = 2

    if name.lower() == 'brain':
        root = os.path.join(dset_dir, 'brain-dataset')
        if not os.path.exists(root):
            print("brain data is not here")
        file_path = os.path.join(root, "layer6_norm_brain_dataset_np_float32.npy")
        data = np.load(file_path)
        data = data[:,4:452,:]
        data = np.reshape(data, (-1, 1, 448, 320))
        data = torch.from_numpy(data)
        print("brain_data_shape:{}".format(data.shape))
        train_kwargs = {'data_tensor': data}
        dset = CustomTensorDataset

    elif name.lower() == 'brain_test':
        root = os.path.join(dset_dir, 'brain-dataset')
        if not os.path.exists(root):
            print("brain data is not here")
        file_path = os.path.join(root, "test_brain_dataset_np_float32.npy")
        data = np.load(file_path)
        data = data[:, 4:452, :]
        data = np.reshape(data, (-1, 1, 448, 320))
        data = torch.from_numpy(data)
        print("brain_data_shape:{}".format(data.shape))
        train_kwargs = {'data_tensor': data}
        dset = CustomTensorDataset

    else:
        raise NotImplementedError

    train_data = dset(**train_kwargs)
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=False)

    data_loader = train_loader

    return data_loader



if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(), ])

    dset = CustomImageFolder('data/CelebA', transform)
    loader = DataLoader(dset,
                        batch_size=32,
                        shuffle=True,
                        num_workers=1,
                        pin_memory=False,
                        drop_last=True)

    images1 = iter(loader).next()
    import ipdb

    ipdb.set_trace()
