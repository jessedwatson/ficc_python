import torch
import tqdm

from PIL import Image

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import wandb as W

from captum.attr import LayerIntegratedGradients
from captum.attr._utils.input_layer_wrapper import ModelInputWrapper

from ficc.models.lstm_ysm_pytorch import YieldSpreadSquaredErrorWrapper, VarianceWrapper


def compute_integrated_gradient_attributions(
    model,
    x_eval,
    y_eval=None,
    model_wrapper=None,
    BATCH_SIZE=1000
):
    if not isinstance(model, ModelInputWrapper):
        model = model_wrapper(model) if model_wrapper is not None else model
        model = ModelInputWrapper(model)

    # Build all the inputs for attribution
    inputs = [model.input_maps["trade_history"], model.input_maps["noncat"]]
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            inputs = inputs + [module]

    # Build the integrated gradients model
    ig = LayerIntegratedGradients(model, inputs)

    # Move the model and the inputs to the GPU, if it's available
    if torch.cuda.is_available():
        model = model.cuda()
        x_eval = [x.cuda() for x in x_eval]
        if y_eval is not None:
            y_eval = y_eval.cuda()

        # Set the trade history and non-categorical features to be 32-bit floats
        x_eval[0] = x_eval[0].float()
        x_eval[1] = x_eval[1].float()

    # Compute the integrated gradients in batches
    attributes = None
    for start_idx in tqdm.tqdm(range(0, x_eval[0].shape[0], BATCH_SIZE)):
        end_idx = min(start_idx+BATCH_SIZE, x_eval[0].shape[0])
        if y_eval is None:
            x = [t[start_idx:end_idx, ...] for t in x_eval]
        else:
            x = [y_eval[start_idx:end_idx]] + [t[start_idx:end_idx, ...]
                                               for t in x_eval]
        ig_attr_test = ig.attribute(tuple(x), internal_batch_size=1)

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
    return compute_integrated_gradient_attributions(model, x_eval, y_eval, YieldSpreadSquaredErrorWrapper, BATCH_SIZE)


def compute_integrated_gradient_variance_attributions(
    model,
    x_eval,
    BATCH_SIZE=1000
):
    return compute_integrated_gradient_attributions(model, x_eval, model_wrapper=VarianceWrapper, BATCH_SIZE=BATCH_SIZE)


def visualize_trade_history_attribution(attrs, subtitle=None, wandb=None):
    fig, ax = plt.subplots(figsize=(15, 10))
    xticklabels = list(range(attrs.shape[1]))
    yticklabels = ["yield_spread", "par_traded",
                   "trade_type 1", "trade_type 2", "seconds_ago"]
    ax = sns.heatmap(attrs.cpu().numpy().transpose(
    ), xticklabels=xticklabels, yticklabels=yticklabels, linewidth=0.2)
    if subtitle is None:
        title = "Trade History Attribution"
    else:
        title = f"Trade History Attribution ({subtitle})"
    plt.xlabel('Sequence Index')
    plt.ylabel('Feature Index')
    fig.suptitle(title)

    if wandb is not None:
        pil_image = Image.frombytes(
            'RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        pil_image.info['dpi'] = (fig.dpi, fig.dpi)
        pil_image.save("trade_history_attribution.png")
        image = W.Image(pil_image)
        if subtitle is None:
            wandb.log({title: [image]})
        else:
            wandb.log({subtitle + "/" + title: [image]})

    # plt.show()


def visualize_trade_numerical_and_binary_attribution(attrs, numerical_features, binary_features, subtitle=None, wandb=None):
    fig, ax = plt.subplots(figsize=(15, 5))
    plt.xticks(rotation='vertical')
    fig.subplots_adjust(bottom=0.6)
    yticklabels = [""]
    xticklabels = numerical_features + \
        [b + " (binary)" for b in binary_features]
    ax = sns.heatmap(attrs.unsqueeze(-2).cpu().numpy(),
                     xticklabels=xticklabels, yticklabels=yticklabels, linewidth=0.2)
    if subtitle is None:
        title = "Numerical and Binary Attribution"
    else:
        title = f"Numerical and Binary Attribution ({subtitle})"
    plt.xlabel('Feature')
    fig.suptitle(title)

    if wandb is not None:
        pil_image = Image.frombytes(
            'RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        pil_image.info['dpi'] = (fig.dpi, fig.dpi)
        image = W.Image(pil_image)
        if subtitle is None:
            wandb.log({title: [image]})
        else:
            wandb.log({subtitle + "/" + title: [image]})

    # plt.show()


def visualize_categorical_attribution(attrs, cat_features, subtitle=None, wandb=None):
    fig, ax = plt.subplots(figsize=(15, 2))
    yticklabels = [""]
    xticklabels = cat_features
    ax = sns.heatmap(attrs.unsqueeze(-2).cpu().numpy(),
                     xticklabels=xticklabels, yticklabels=yticklabels, linewidth=0.2)
    if subtitle is None:
        title = "Categorical Attribution"
    else:
        title = f"Categorical Attribution ({subtitle})"
    plt.xlabel('Feature')
    fig.suptitle(title)

    if wandb is not None:
        pil_image = Image.frombytes(
            'RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        pil_image.info['dpi'] = (fig.dpi, fig.dpi)
        image = W.Image(pil_image)
        if subtitle is None:
            wandb.log({title: [image]})
        else:
            wandb.log({subtitle + "/" + title: [image]})

    # plt.show()
