
import tensorflow as tf
import pytorch_lightning as pl
import os
import copy
from pytorch_lightning.loggers import WandbLogger

def eval_pytorch_mae(
        model,
        x,  
        y,
        batch_size
):
    from torch.utils.data import TensorDataset, DataLoader

    test_loader = DataLoader(TensorDataset(*x, y), batch_size=batch_size, num_workers=os.cpu_count())

    evaluator = pl.Trainer(gpus=1)
    results = evaluator.test(model, test_loader, verbose=False)

    return results[0]['test_mae']


def eval_keras_mae(
        model,
        x,  
        y
):
    _, mae = model.evaluate(x, y, verbose=1)

    return mae


def eval_model(
        model,
        df,
        create_input,
        create_labels,
        wandb,
        categories=None,
        batch_size=1000):
    pass

    x_test = create_input(df)
    y_test = create_labels(df)

    if isinstance(model, tf.keras.Model):
        mae = eval_keras_mae(model, x_test, y_test)
    elif isinstance(model, pl.LightningModule):
        mae = eval_pytorch_mae(model, x_test, y_test, batch_size)

    if wandb is not None:
        wandb.log({"Full MAE": mae})
    else:
        print(f"Full MAE: {round(mae, 3)}")

    if categories is not None:
        for category, category_df in df.groupby(by=categories, squeeze=True):
            x_test = create_input(category_df)
            y_test = create_labels(category_df)

            if isinstance(model, tf.keras.Model):
                mae = eval_keras_mae(model, x_test, y_test)
            elif isinstance(model, pl.LightningModule):
                mae = eval_pytorch_mae(model, x_test, y_test, batch_size)

            if isinstance(category, tuple):
                category = ','.join(category)
            
            if wandb is not None:
                wandb.log({f"{category} MAE": mae})
            else:
                print(f"{category} MAE: {round(mae, 3)}")
