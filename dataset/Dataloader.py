import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4
root_data = '/media/victor/851aa2dd-6b93-4a57-8100-b5253aa4eedd/dataset'

trainset = torchvision.datasets.CIFAR10(root=root_data, train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0, pin_memory=True)

testset = torchvision.datasets.CIFAR10(root=root_data, train=False,
                                        download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=0, pin_memory=True)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np


    # functions to show an image
    def imshow(img):
        img = img / 2 + 0.5 #unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    print(f' '.join(classes[labels[j]] for j in range(batch_size)))