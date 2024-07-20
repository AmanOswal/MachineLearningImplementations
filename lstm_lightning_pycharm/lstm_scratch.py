# To load tensorboard, first go to the directory having lightning_logs
# Then use the below command from terminal
# tensorboard --logdir=lightning_logs/
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.optim import SGD, Adam
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from lightning.pytorch.tuner import Tuner
import seaborn as sns
import numpy as np


# n_input_features = 1
# n_samples = 5
# n_time_steps = random
# hidden state size = 1
# output size = 1
# Move data as (n_samples, n_time_steps, n_features) sequence
# Move hidden state as (n_samples, 1)
# Move cell state as (n_samples, 1)
# I have assumed n_features=1 in this

class LSTMfromScratch(L.LightningModule):
    def __init__(self):
        L.seed_everything(seed=42)
        super().__init__()

        mean = torch.tensor(0)
        std = torch.tensor(1)

        self.ow = nn.Parameter(torch.normal(mean, std, size=(2, 1)), requires_grad=True)  # (2,1)
        self.ob = nn.Parameter(torch.normal(mean, std, size=(1, 1)), requires_grad=True)  # (1,1)
        self.iw = nn.Parameter(torch.normal(mean, std, size=(2, 1)), requires_grad=True)  # (2,1)
        self.ib = nn.Parameter(torch.normal(mean, std, size=(1, 1)), requires_grad=True)  # (1,1)
        self.fw = nn.Parameter(torch.normal(mean, std, size=(2, 1)), requires_grad=True)  # (2,1)
        self.fb = nn.Parameter(torch.normal(mean, std, size=(1, 1)), requires_grad=True)  # (1,1)
        self.cw = nn.Parameter(torch.normal(mean, std, size=(2, 1)), requires_grad=True)  # (2,1)
        self.cb = nn.Parameter(torch.normal(mean, std, size=(1, 1)), requires_grad=True)  # (1,1)
        self.NN_1 = torch.nn.Linear(1,2)
        self.NN_2 = torch.nn.Linear(2,1)
        # self.NN_3 = torch.nn.Linear(25,1)
        self.lr = 0.01

    def lstm_unrolled_unit(self, input_cell, input_hidden_state, input_cell_state):
        # Combine [h_t-1 and x_t]
        # output_shape (n_samples, 2)
        # input_hidden_state should be (n_samples, 1)
        # input_cell_state should be (n_samples, 1)
        input_plus_last_hidden_state = torch.cat((input_hidden_state, input_cell), axis=1)  # (n_samples, 2)

        # new_hidden_state = input_plus_last_hidden_state[:, 0]  # (n_samples, 1)

        input_gate = F.sigmoid(torch.matmul(input_plus_last_hidden_state, self.iw) + self.ib)  # (n_samples, 1)
        forget_gate = F.sigmoid(torch.matmul(input_plus_last_hidden_state, self.fw) + self.fb)  # (n_samples, 1)
        output_gate = F.sigmoid(torch.matmul(input_plus_last_hidden_state, self.ow) + self.ob)  # (n_samples, 1)

        new_prospective_cell_state = F.tanh(
            torch.matmul(input_plus_last_hidden_state, self.cw) + self.cb)  # (n_samples, 1)
        new_cell_state = new_prospective_cell_state * input_gate + input_cell_state * forget_gate
        new_hidden_state = output_gate * F.tanh(new_cell_state)
        # return new_prospective_cell_state, new_hidden_state

        return new_cell_state, new_hidden_state

    def forward(self, input_data):
        # print(type(input_data))
        # print(input_data.shape)
        hidden_state_prev = torch.tensor([[0.] for i in range(input_data.shape[0])])
        cell_state_prev = torch.tensor([[0.] for i in range(input_data.shape[0])])
        for time_step in range(len(input_data[0])):
            hidden_state_prev, cell_state_prev = self.lstm_unrolled_unit(input_data[:, time_step].reshape(-1, 1),
                                                                         hidden_state_prev, cell_state_prev)
        NN_out_1 = self.NN_1(hidden_state_prev)
        NN_out_2 = self.NN_2(NN_out_1)

        return NN_out_2

    #         input_x =
    #         return the output from a forward step

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

    def training_step(self, data):
        batch_x = data[0]
        batch_y = data[1]
        loss_func = nn.MSELoss()
        pred = self.forward(batch_x)
        loss = loss_func(pred, batch_y)
        # TODO: rename train_loss to something else
        self.log("train_loss", loss)
        return loss

    def validation_step(self, data):
        batch_x = data[0]
        batch_y = data[1]
        loss_func = nn.MSELoss()
        pred = self.forward(batch_x)
        loss = loss_func(pred, batch_y)
        # TODO: rename train_loss to something else
        self.log("validation_loss", loss)
        return loss

    def test_step(self, data):
        batch_x = data[0]
        batch_y = data[1]
        loss_func = nn.MSELoss()
        pred = self.forward(batch_x)
        loss = loss_func(pred, batch_y)
        self.log("Testing_loss", loss)
        # for i in range(len(pred)):
        self.log(f'Expected_value', batch_y)
        self.log('Found Value', pred)
        return loss

    def get_input_data(self):
        input_features = torch.tensor([([i for i in range(j, j + 4)]) for j in range(1, 100)], dtype=torch.float)
        out_data = torch.tensor([(i+5) for i in range(1, 100)], dtype=torch.float)
        return input_features, out_data

    def get_data_loader_from_data(self, input_features, out_data, batch_size=2):
        # self.log("IF", input_features)
        # self.log("lr", self.lr)
        t_data = TensorDataset(input_features, out_data)
        d_data = DataLoader(t_data, batch_size=batch_size)
        return d_data

    def get_data_loader_from_data_with_validation_split(self, input_features, out_data, batch_size_div=10, val_fraction=0.2, test_fraction=0.2):
        # self.log("IF", input_features)
        # self.log("lr", self.lr)
        t_data = TensorDataset(input_features, out_data)
        len_train = int(len(t_data)*(1-(val_fraction+test_fraction)))
        len_val = int(len(t_data) * val_fraction)
        len_test = len(t_data) - len_train - len_val

        train_data, val_data, test_data = random_split(t_data, [len_train, len_val, len_test])
        train_d_data = DataLoader(train_data, batch_size=max(5, len_train//batch_size_div))
        val_d_data = DataLoader(val_data, batch_size=len_val)
        test_d_data = DataLoader(test_data)

        return train_d_data, val_d_data, test_d_data


if __name__ == "__main__":
    # Normal Training
    model = LSTMfromScratch()

    input_features, out_data = model.get_input_data()
    train_d_data, val_d_data, test_d_data = model.get_data_loader_from_data_with_validation_split(input_features, out_data)
    # d_data = model.get_data_loader_from_data(input_features, out_data)
    # trainer = L.Trainer(max_epochs=1000, log_every_n_steps=2)
    # trainer.fit(model, d_data)

    # Find best LR
    trainer = L.Trainer(max_epochs=1000)
    # tuner_class = Tuner(trainer)
    # lr_finder = tuner_class.lr_find(model, train_dataloaders=train_d_data)

    # Plot and set the learning rate
    # fig = lr_finder.plot(suggest=True)
    # plt.show()
    # suggested_lr = lr_finder.suggestion()
    # print(suggested_lr)
    # if suggested_lr>0.001:
    #     for param_group in model.trainer.optimizers[0].param_groups:
    #         param_group['lr'] = suggested_lr
        #     param_group['lr'] = 0.01


    # Validation and testing
    # model = LSTMfromScratch()
    # input_features, out_data = model.get_input_data()
    # train_d_data, val_d_data, test_d_data = model.get_data_loader_from_data_with_validation_split(input_features, out_data)
    # trainer = L.Trainer(max_epochs=1000, log_every_n_steps=2)
    trainer.fit(model, train_d_data, val_d_data)

    # Training after breaking the whole training process into multiple epoch batches
    # This picks the best point from last epoch
    # path_bst_cp = trainer.checkpoint_callback.best_model_path
    # trainer = L.Trainer(max_epochs=2000)
    # trainer.fit(model, d_data, ckpt_path=path_bst_cp)

    # Starting model training from checkpoint
    # chk_path = "lightning_logs/version_11/checkpoints/epoch=999-step=50000.ckpt"
    # model = LSTMfromScratch.load_from_checkpoint(checkpoint_path=chk_path)
    # input_features, out_data = model.get_input_data()
    # trainer = L.Trainer(max_epochs=1000, log_every_n_steps=2)
    d_data = model.get_data_loader_from_data(input_features, out_data, batch_size=1)
    trainer.test(model, d_data)
