import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from ficc.models.registration import register_model


class LSTMYieldSpreadModel(pl.LightningModule):
    def __init__(
        self,
        num_trade_history_features,
        non_categorical_size,
        category_sizes,
    ):
        super().__init__()

        self.trade_history_lstm = nn.LSTM(
            input_size=num_trade_history_features,
            hidden_size=460,
            num_layers=2,
            dropout=0.0,
            batch_first=True,
        )

        self.CATEGORICAL_START = 2
        self.embed = nn.ModuleList()
        tabular_size = len(non_categorical_size)
        for cat_name in category_sizes:
            cat_size = category_sizes[cat_name]+1
            self.embed.append(nn.Embedding(cat_size, 10))
            tabular_size += 10

        self.tabular_model = nn.Sequential(
            nn.Linear(tabular_size, 260),
            nn.ReLU(),
            nn.BatchNorm1d(260),
            nn.Linear(260, 10),
            nn.ReLU(),
            nn.BatchNorm1d(10),
            nn.Linear(10, 460),
            nn.Tanh(),
        )

        self.final_stage = nn.Sequential(
            nn.Linear(460+460, 250),
            nn.ReLU(),
            nn.BatchNorm1d(250),
            nn.Linear(250, 600),
            nn.Tanh(),
            nn.BatchNorm1d(600),
            nn.Linear(600, 1),
        )

    def forward(self, *inputs):
        if len(inputs) == 1 and (isinstance(inputs[0], tuple) or isinstance(inputs[0], list)):
            inputs = inputs[0]

        trade_history = self.trade_history_lstm(inputs[0])
        
        # [0] to use the output vector, LSTM returns output vector and hidden state as a tuple
        trade_history = trade_history[0]

        # PyTorch returns the output for each timestep, we only use the final one
        trade_history = trade_history[:, -1, :]

        for i in range(self.CATEGORICAL_START, len(inputs)):
            inputs[i] = self.embed[i-self.CATEGORICAL_START](inputs[i])

        tabular = self.tabular_model(torch.cat(inputs[1:], dim=-1))

        return self.final_stage(torch.cat([trade_history, tabular], dim=-1))

    def training_step(self, batch, batch_idx):
        x, y = batch[:-1], batch[-1].squeeze()
        z = self.forward(x).squeeze()
        loss = F.mse_loss(y, z)

        self.log("train_loss", loss.item())
        self.log("train_mae", torch.abs(z - y).mean().item())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[:-1], batch[-1].squeeze()
        z = self.forward(x).squeeze()
        loss = F.mse_loss(y, z)

        self.log("val_loss", loss.item())
        self.log("val_mae", torch.abs(z - y).mean().item())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer


def build_lstm_model_v1(
        hp,
        num_trade_history_features,
        non_categorical_size,
        category_sizes):
    return LSTMYieldSpreadModel(
        num_trade_history_features,
        non_categorical_size,
        category_sizes)


register_model("lstm_yield_spread_model_pytorch", 1, build_lstm_model_v1,
               "Initial LSTM-based yield spread model")
