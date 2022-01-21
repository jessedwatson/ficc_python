import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from ficc.models.registration import register_model


class LSTMCore(pl.LightningModule):
    def __init__(
        self,
        num_trade_history_features,
        non_categorical_size,
        category_sizes,
        num_outputs=1
    ):
        super().__init__()

        self.trade_history_lstm = nn.LSTM(
            input_size=num_trade_history_features,
            hidden_size=460,
            num_layers=2,
            dropout=0.0,
            batch_first=True,
        )

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
            nn.Linear(600, num_outputs),
        )

        self.lr = 1e-4

    def forward(self, trade_history, noncat, *categorical):
        trade_history = self.trade_history_lstm(trade_history)

        # [0] to use the output vector, LSTM returns output vector and hidden state as a tuple
        trade_history = trade_history[0]

        # PyTorch returns the output for each timestep, we only use the final one
        trade_history = trade_history[:, -1, :]

        if len(categorical) == 1 and (isinstance(categorical[0], tuple) or isinstance(categorical[0], list)):
            categorical = categorical[0]
        categorical = [self.embed[i](categorical[i])
                       for i in range(len(categorical))]

        tabular = self.tabular_model(torch.cat([noncat] + categorical, dim=-1))

        return self.final_stage(torch.cat([trade_history, tabular], dim=-1))


class LSTMYieldSpreadModel(LSTMCore):
    def __init__(
        self,
        num_trade_history_features,
        non_categorical_size,
        category_sizes,
    ):
        super().__init__(
            num_trade_history_features,
            non_categorical_size,
            category_sizes)

    def training_step(self, batch, batch_idx):
        x, y = batch[:-1], batch[-1].squeeze()
        z = self.forward(x[0], x[1], x[2:]).squeeze()
        loss = F.mse_loss(y, z)

        self.log("train_loss", loss.item())
        self.log("train_mae", torch.abs(z - y).mean().item())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[:-1], batch[-1].squeeze()
        z = self.forward(x[0], x[1], x[2:]).squeeze()
        loss = F.mse_loss(y, z)

        self.log("val_loss", loss.item())
        self.log("val_mae", torch.abs(z - y).mean().item())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


class LSTMYieldSpreadDistributionModel(LSTMCore):
    def __init__(
        self,
        num_trade_history_features,
        non_categorical_size,
        category_sizes,
    ):
        super().__init__(
            num_trade_history_features,
            non_categorical_size,
            category_sizes,
            num_outputs=2)


    def forward(self, trade_history, noncat, *categorical):
        log_scale = super().forward(trade_history, noncat, *categorical)

        return log_scale[:, 0], log_scale[:, 1].clamp(min=1e-6)

    def loss(self, loc, scale, target):
        var = (scale ** 2)
        log_scale = scale.log()
        log_prob = -((target - loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))
        nll_loss = -log_prob.mean()

        return nll_loss

    def training_step(self, batch, batch_idx):
        x, y = batch[:-1], batch[-1].squeeze()
        loc, scale = self.forward(x[0], x[1], x[2:])

        # Compute NLL
        nll_loss = self.loss(loc, scale, y)

        self.log("train_loss", nll_loss.item())
        self.log("train_mae", torch.abs(loc - y).mean().item())
        return nll_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[:-1], batch[-1].squeeze()
        loc, scale = self.forward(x[0], x[1], x[2:])

        # Compute NLL
        nll_loss = self.loss(loc, scale, y)

        self.log("val_loss", nll_loss.item())
        self.log("val_mae", torch.abs(loc - y).mean().item())
        return nll_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


class YieldSpreadSquaredError(nn.Module):
    def __init__(
        self,
        model
    ):
        super().__init__()

        self.model = model

    def forward(self, ground_truth, trade_history, noncat, *categorical):
        estimate = self.model(trade_history, noncat, *categorical).squeeze()
        diff = (ground_truth - estimate)
        loss = diff.pow(2).mean()
        return loss.unsqueeze(0)


class VarianceWrapper(nn.Module):
    def __init__(
        self,
        model
    ):
        super().__init__()

        self.model = model

    def forward(self, trade_history, noncat, *categorical):
        loc, scale = self.model(trade_history, noncat, *categorical)
        return scale.mean().unsqueeze(0)


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


def build_lstm_dist_model_v1(
        hp,
        num_trade_history_features,
        non_categorical_size,
        category_sizes):
    return LSTMYieldSpreadDistributionModel(
        num_trade_history_features,
        non_categorical_size,
        category_sizes)


register_model("lstm_yield_spread_dist_model_pytorch", 1, build_lstm_dist_model_v1,
               "LSTM-based yield spread distribution model")
