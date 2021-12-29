
def eval_model(
        model,
        df,
        create_input,
        create_labels,
        wandb,
        categories=None):
    pass

    x_test = create_input(df)
    y_test = create_labels(df)

    _, mae = model.evaluate(x_test, y_test, verbose=1)
    wandb.log({"Full MAE": mae})

    if categories is not None:
        for category, category_df in df.groupby(by=categories, squeeze=True):
            x_test = create_input(category_df)
            y_test = create_labels(category_df)

            _, mae = model.evaluate(x_test, y_test, verbose=1)
            if isinstance(category, tuple):
                category = ','.join(category)
            
            wandb.log({f"{category} MAE": mae})
