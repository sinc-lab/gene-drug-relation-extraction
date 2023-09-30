from torch import nn
import torch as tr
import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import balanced_accuracy_score, \
    f1_score, roc_auc_score
from torch.nn.functional import softmax
from torch import flatten
import time


class InteractionCNN(pl.LightningModule):
    def __init__(self, emb_size, nblocks, nfilters, nclasses,
                 class_weight):

        super(InteractionCNN, self).__init__()
        self.nclasses = nclasses
        cnn = []
        cnn.append(nn.Conv1d(emb_size, int(4*nfilters), kernel_size=3,
                                padding=1))
        cnn.append(nn.ELU())
        cnn.append(nn.BatchNorm1d(int(4*nfilters)))
        cnn.append(nn.Conv1d(int(4*nfilters), nfilters, kernel_size=3,
                                padding=1))
        cnn.append(nn.Dropout(.5))
        for i in range(nblocks):
            cnn.append(ResNet([nfilters, nfilters], [3, 3]))
            cnn.append(nn.MaxPool1d(2))
            cnn.append(nn.Dropout(.3))

        cnn.append(nn.ELU())
        cnn.append(nn.BatchNorm1d(nfilters))
        cnn.append(nn.Dropout(.1))

        cnn.append(nn.Conv1d(nfilters, nclasses, kernel_size=1, padding=0))
        self.globalmax = nn.AdaptiveMaxPool1d(1)

        self.cnn = nn.Sequential(*cnn)

        if class_weight is not None:
            self.loss = nn.CrossEntropyLoss(weight=class_weight)
        else:
            self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        y = self.cnn(x)
        y = self.globalmax(y).squeeze()

        if len(y.shape) < 2:
            y = y.unsqueeze(0)
        return y

    def configure_optimizers(self):
        return tr.optim.Adam(self.parameters(), lr=2e-5)

    def common_step(self, batch, stage):
        x, y, _ = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        self.log(f"{stage}_loss", loss)

        ybin = tr.argmax(y_hat.detach().cpu(), axis=1)
        balacc = balanced_accuracy_score(y.cpu(), ybin)
        f1 = f1_score(y.cpu(), ybin, average="weighted")

        try:
            auc = roc_auc_score(y.cpu(), softmax(y_hat.detach().cpu(), dim=1),
                                average="weighted", multi_class="ovo",
                                labels=np.arange(self.nclasses))
        except ZeroDivisionError:
            auc = -1
        self.log(f"{stage}_balacc", balacc)
        self.log(f"{stage}_f1", f1)
        self.log(f"{stage}_auc", auc)

        return loss

    def training_step(self, batch, batch_id):
        loss = self.common_step(batch, "train")

        return loss

    def validation_step(self, batch, batch_id):
        self.common_step(batch, "valid")

    def test_step(self, batch, batch_id):
        self.common_step(batch, "test")


class ResNet(nn.Module):
    def __init__(self, nfilters, ksizes):
        super(ResNet, self).__init__()
        self.in_dim = nfilters[len(nfilters)-1]
        nfilters.insert(0, self.in_dim)
        layers = []
        for i in range(len(nfilters)-1):
            layers.append(nn.ELU())
            layers.append(nn.BatchNorm1d(nfilters[i]))
            layers.append(nn.Conv1d(nfilters[i], nfilters[i+1],
                kernel_size=ksizes[i], padding=int(ksizes[i]/2)))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x) + x
