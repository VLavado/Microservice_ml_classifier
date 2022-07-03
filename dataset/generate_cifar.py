from torchvision import datasets
train_folder = '/media/victor/851aa2dd-6b93-4a57-8100-b5253aa4eedd/dataset/cifar10/train'
test_folder = '/media/victor/851aa2dd-6b93-4a57-8100-b5253aa4eedd/dataset/cifar10/test'

dataset_train = datasets.CIFAR10(root=train_folder,
                                 train=True,
                                 download=True)

dataset_test = datasets.CIFAR10(root=test_folder,
                                 train=False,
                                 download=True)