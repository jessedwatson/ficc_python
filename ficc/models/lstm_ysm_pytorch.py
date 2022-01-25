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
        lstm_sizes=[460, 460],
        embed_sizes=10,
        tabular_sizes=[260, 10, 460],
        final_sizes=[250, 600],
        dropout=0.0,
        num_outputs=1,
        **kwargs
    ):
        super().__init__()

        lstm_sizes = [num_trade_history_features] + lstm_sizes
        self.trade_history_lstm = nn.ModuleList()
        for i in range(len(lstm_sizes) - 1):
            self.trade_history_lstm.append(
                nn.LSTM(
                    input_size=lstm_sizes[i],
                    hidden_size=lstm_sizes[i+1],
                    dropout=dropout,
                    batch_first=True,
                )
            )
        self.trade_history_final = nn.Linear(lstm_sizes[-1], lstm_sizes[-1])

        self.embed = nn.ModuleList()
        tabular_size = len(non_categorical_size)
        if isinstance(embed_sizes, list):
            assert len(embed_sizes) == len(
                category_sizes), "embed_sizes must be a list of the same length as category_sizes"
            for emb_size, cat_name in zip(embed_sizes, category_sizes):
                cat_size = category_sizes[cat_name]+1
                self.embed.append(nn.Embedding(cat_size, emb_size))
                tabular_size += emb_size
        else:
            for cat_name in category_sizes:
                cat_size = category_sizes[cat_name]+1
                self.embed.append(nn.Embedding(cat_size, embed_sizes))
                tabular_size += embed_sizes

        self.tabular_model = (
            nn.Linear(tabular_size, tabular_sizes[0]),
            nn.ReLU(),
            nn.BatchNorm1d(tabular_sizes[0]),
            nn.Dropout(dropout),
            nn.Linear(tabular_sizes[0], tabular_sizes[1]),
            nn.ReLU(),
            nn.BatchNorm1d(tabular_sizes[1]),
            nn.Dropout(dropout),
            nn.Linear(tabular_sizes[1], tabular_sizes[2]),
        )
        self.tabular_model = nn.Sequential(
            *self.tabular_model,
            nn.Tanh(),
        )

        self.final_stage = (
            nn.Linear(lstm_sizes[-1]+tabular_sizes[-1], final_sizes[0]),
            nn.ReLU(),
            nn.BatchNorm1d(final_sizes[0]),
            nn.Dropout(dropout),
            nn.Linear(final_sizes[0], final_sizes[1]),
            nn.Tanh(),
            nn.BatchNorm1d(final_sizes[1]),
            nn.Dropout(dropout),
        )
        self.final_stage = nn.Sequential(
            *self.final_stage,
            nn.Linear(final_sizes[-1], num_outputs),
        )

        if 'learning_rate' in kwargs:
            self.lr = kwargs['learning_rate']
        else:
            self.lr = 1e-4

        self.learning_schedule = kwargs["learning_schedule"]
        if self.learning_schedule == "cyclic":
            self.max_factor = kwargs["max_factor"]

    def forward(self, trade_history, noncat, *categorical):
        for lstm in self.trade_history_lstm:
            trade_history, (h, c) = lstm(trade_history)

        # PyTorch returns the output for each timestep, we only use the final one
        trade_history = trade_history[:, -1, :]
        trade_history = F.relu(self.trade_history_final(trade_history))

        if len(categorical) == 1 and (isinstance(categorical[0], tuple) or isinstance(categorical[0], list)):
            categorical = categorical[0]
        categorical = [self.embed[i](categorical[i])
                       for i in range(len(categorical))]

        tabular_input = torch.cat([noncat] + categorical, dim=-1)
        tabular = self.tabular_model(tabular_input)

        return self.final_stage(torch.cat([trade_history, tabular], dim=-1))


class LSTMYieldSpreadModel(LSTMCore):
    def __init__(
        self,
        num_trade_history_features,
        non_categorical_size,
        category_sizes,
        **kwargs
    ):
        super().__init__(
            num_trade_history_features,
            non_categorical_size,
            category_sizes,
            **kwargs)

    def step(self, batch, name):
        x, y = batch[:-1], batch[-1].squeeze()
        z = self.forward(x[0], x[1], x[2:]).squeeze()
        loss = F.mse_loss(y, z)

        self.log(f"{name}_loss", loss.item())
        self.log(f"{name}_mae", torch.abs(z - y).mean().item())
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def configure_optimizers(self):
        if self.learning_schedule == "cyclic":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.lr, max_lr=self.lr * self.max_factor)
            return [optimizer], [scheduler]
        elif self.learning_schedule == "1cycle":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, total_steps=2000)
            return [optimizer], [scheduler]
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
            return optimizer


class LSTMYieldSpreadDistributionModel(LSTMCore):
    def __init__(
        self,
        num_trade_history_features,
        non_categorical_size,
        category_sizes,
        **kwargs
    ):
        super().__init__(
            num_trade_history_features,
            non_categorical_size,
            category_sizes,
            num_outputs=2,
            **kwargs)

    def forward(self, trade_history, noncat, *categorical):
        log_scale = super().forward(trade_history, noncat, *categorical)

        return log_scale[:, 0], log_scale[:, 1].clamp(min=1e-6)

    def loss(self, loc, scale, target):
        var = (scale ** 2)
        log_scale = scale.log()
        log_prob = -((target - loc) ** 2) / (2 * var) - \
            log_scale - math.log(math.sqrt(2 * math.pi))
        nll_loss = -log_prob.mean()

        return nll_loss

    def step(self, batch, name):
        x, y = batch[:-1], batch[-1].squeeze()
        loc, scale = self.forward(x[0], x[1], x[2:])

        # Compute NLL
        nll_loss = self.loss(loc, scale, y)

        self.log(f"{name}_loss", nll_loss.item())
        self.log(f"{name}_mae", torch.abs(loc - y).mean().item())
        return nll_loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


class YieldSpreadSquaredError(nn.Module):
    def __init__(
        self,
        model
    ):
        super().__init__()

        self.wrapped_model = model

    def forward(self, ground_truth, trade_history, noncat, *categorical):
        estimate = self.wrapped_model(
            trade_history, noncat, *categorical).squeeze()
        diff = (ground_truth - estimate)
        loss = diff.pow(2).mean()
        return loss.unsqueeze(0)


class VarianceWrapper(nn.Module):
    def __init__(
        self,
        model
    ):
        super().__init__()

        self.wrapped_model = model

    def forward(self, trade_history, noncat, *categorical):
        loc, scale = self.wrapped_model(trade_history, noncat, *categorical)
        return scale.mean().unsqueeze(0)


def build_lstm_model_v1(
        hp,
        num_trade_history_features,
        non_categorical_size,
        category_sizes,
        **kwargs):
    return LSTMYieldSpreadModel(
        num_trade_history_features,
        non_categorical_size,
        category_sizes,
        **kwargs)


register_model("lstm_yield_spread_model_pytorch", 1, build_lstm_model_v1,
               "Initial LSTM-based yield spread model")


def build_lstm_dist_model_v1(
        hp,
        num_trade_history_features,
        non_categorical_size,
        category_sizes,
        **kwargs):
    return LSTMYieldSpreadDistributionModel(
        num_trade_history_features,
        non_categorical_size,
        category_sizes,
        **kwargs)


register_model("lstm_yield_spread_dist_model_pytorch", 1, build_lstm_dist_model_v1,
               "LSTM-based yield spread distribution model")
