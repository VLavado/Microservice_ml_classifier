import torch
import torch.nn as nn
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset.Dataloader import trainloader


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential1 = nn.Sequential(nn.Conv2d(3, 6, 5),
                                         nn.ReLU(),
                                         nn.Dropout(0.2),
                                         nn.MaxPool2d(2, 2))
        self.sequential2 = nn.Sequential(nn.Conv2d(6, 16, 5),
                                         nn.ReLU(),
                                         nn.Dropout(0.2),
                                         nn.MaxPool2d(2, 2))
        self.sequential3 = nn.Sequential(nn.Linear(16 * 5 * 5, 120),
                                         nn.ReLU(),
                                         nn.Dropout(0.2))
        self.sequential4 = nn.Sequential(nn.Linear(120, 84),
                                         nn.ReLU(),
                                         nn.Dropout(0.2))
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.sequential1(x)
        x = self.sequential2(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.sequential3(x)
        x = self.sequential4(x)
        x = self.fc3(x)
        return x

class Mlmodel(LightningModule):
    def __init__(self, model):
        super(Mlmodel, self).__init__()
        self.classifier = model
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.classifier(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.classifier(x)
        loss = self.loss(y_hat, y)
        self.log("test loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([p for p in self.parameters() if p.requires_grad],
                                     lr = 1e-1)
        return optimizer

if __name__ == '__main__':
    model = Net()
    callback = ModelCheckpoint(dirpath='/media/victor/851aa2dd-6b93-4a57-8100-b5253aa4eedd/cursos/checkpoint_model_microservice',
                           monitor="train_loss")

    trainer = Trainer(max_epochs=10,
                  gpus=-1,
                  profiler='simple',
                  callbacks=callback)
    module = Mlmodel(model)
    trainer.fit(module, train_dataloaders=trainloader)


