import torch
import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from captum.attr import LayerIntegratedGradients
from captum.attr._utils.input_layer_wrapper import ModelInputWrapper

from ficc.models.lstm_ysm_pytorch import YieldSpreadSquaredError, VarianceWrapper


def compute_integrated_gradient_attributions(
    model,
    x_eval,
    BATCH_SIZE=1000
):
    if not isinstance(model, ModelInputWrapper):
        model = ModelInputWrapper(model)

    ig = LayerIntegratedGradients(model, [model.input_maps["trade_history"],
                                  model.input_maps["noncat"]] + [e for e in model.module.embed.children()])

    if torch.cuda.is_available():
        model = model.cuda()
        x_eval = [x.cuda() for x in x_eval]

        # Set the trade history and non-categorical features to be 32-bit floats
        x_eval[0] = x_eval[0].float()
        x_eval[1] = x_eval[1].float()

    attributes = None
    for start_idx in tqdm.tqdm(range(0, x_eval[0].shape[0], BATCH_SIZE)):
        end_idx = min(start_idx+BATCH_SIZE, x_eval[0].shape[0])
        x = [t[start_idx:end_idx, ...] for t in x_eval]
        ig_attr_test = ig.attribute(
            tuple(x), internal_batch_size=1, n_steps=10)

        if attributes is None:
            attributes = ig_attr_test
        else:
            attributes = [torch.cat((a, i), dim=0)
                          for a, i in zip(attributes, ig_attr_test)]

    # Get the norm of each categorical feature
    for cat_idx in range(2, len(attributes)):
        attributes[cat_idx] = torch.norm(attributes[cat_idx], dim=1)

    return attributes


def compute_integrated_gradient_ysm_error_attributions(
    model,
    x_eval,
    y_eval,
    BATCH_SIZE=1000
):
    if not isinstance(model, ModelInputWrapper):
        model = YieldSpreadSquaredError(model)
        model = ModelInputWrapper(model)

    ig = LayerIntegratedGradients(model, [model.input_maps["trade_history"],
                                  model.input_maps["noncat"]] + [e for e in model.module.model.embed.children()])

    if torch.cuda.is_available():
        model = model.cuda()
        x_eval = [x.cuda() for x in x_eval]
        y_eval = y_eval.cuda()

        # Set the trade history and non-categorical features to be 32-bit floats
        x_eval[0] = x_eval[0].float()
        x_eval[1] = x_eval[1].float()

    attributes = None
    for start_idx in tqdm.tqdm(range(0, x_eval[0].shape[0], BATCH_SIZE)):
        end_idx = min(start_idx+BATCH_SIZE, x_eval[0].shape[0])
        x = [y_eval[start_idx:end_idx]] + [t[start_idx:end_idx, ...] for t in x_eval]
        ig_attr_test = ig.attribute(
            tuple(x), internal_batch_size=1, n_steps=10)

        if attributes is None:
            attributes = ig_attr_test
        else:
            attributes = [torch.cat((a, i), dim=0)
                          for a, i in zip(attributes, ig_attr_test)]

    # Get the norm of each categorical feature
    for cat_idx in range(2, len(attributes)):
        attributes[cat_idx] = torch.norm(attributes[cat_idx], dim=1)

    return attributes

def visualize_trade_history_attribution(attrs):
    fig, ax = plt.subplots(figsize=(15,10))
    xticklabels=list(range(attrs.shape[1]))
    yticklabels=["yield_spread", "par_traded", "trade_type 1", "trade_type 2", "seconds_ago"]
    ax = sns.heatmap(attrs.cpu().numpy().transpose(), xticklabels=xticklabels, yticklabels=yticklabels, linewidth=0.2)
    plt.xlabel('Sequence Index')
    plt.ylabel('Feature Index')
    plt.show()

def visualize_trade_numerical_and_binary_attribution(attrs, numerical_features, binary_features):
    fig, ax = plt.subplots(figsize=(15,2))
    yticklabels=[""]
    xticklabels=numerical_features + [b + " (binary)" for b in binary_features]
    ax = sns.heatmap(attrs.unsqueeze(-2).cpu().numpy(), xticklabels=xticklabels, yticklabels=yticklabels, linewidth=0.2)
    plt.xlabel('Feature')
    plt.show()

def visualize_categorical_attribution(attrs, cat_features):
    fig, ax = plt.subplots(figsize=(15,2))
    yticklabels=[""]
    xticklabels=cat_features
    ax = sns.heatmap(attrs.unsqueeze(-2).cpu().numpy(), xticklabels=xticklabels, yticklabels=yticklabels, linewidth=0.2)
    plt.xlabel('Feature')
    plt.show()
