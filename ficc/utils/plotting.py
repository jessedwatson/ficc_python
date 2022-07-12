import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_cusip(sdf, data_type, limit=None):
    if limit is not None:
        sdf = sdf[:limit]

    sdf.loc[:, 'trade_datetime'] = pd.to_datetime(sdf.trade_datetime).apply(lambda date: date.timestamp())

    if data_type.lower() == 'prices':
        ficc = 'price_calc_from_yield'
        target = 'msrb_price'
    elif data_type.lower() == 'yield_spread':
        ficc = 'ficc_spread'
        target = 'yield_spread'
    elif data_type.lower() == 'calc_date':
        ficc = 'calc_date_prediction'
        target = 'calc_date_cat'

    sns.set(rc={'figure.figsize':(12,9)})
    ax1 = plt.subplot()
    ax2 = ax1.twinx()
    sns.scatterplot(x='trade_datetime', y=ficc, data=sdf, color="darkviolet", size= 'quantity',style='trade_type',ax=ax1,) #par_traded
    sns.scatterplot(x='trade_datetime', y=target, data=sdf, color="darkgoldenrod", size= 'quantity',style='trade_type',ax=ax2,legend=False)
    ax2.axis('off')
    # get current axis
    # get current xtick labels
    xticks = ax1.get_xticks()
    # convert all xtick labels to selected format from ms timestamp
    ax1.xaxis.set_major_locator(mticker.FixedLocator(ax1.get_xticks().tolist()))
    ax1.set_xticklabels([pd.to_datetime(tm, unit='s').strftime('%Y-%m-%d\n %H:%M:%S') for tm in xticks],rotation=50)
    ax1.set(xlabel='DATE', ylabel='$ PRICE')
    
    ## Set up colors dictionary
    legend = {'ficc price': 'darkviolet','msrb price': 'darkgoldenrod'}

    fake_handles = [mpatches.Patch(color=item) for item in legend.values()]
    label = legend.keys()
    plt.legend(fake_handles, label, loc='lower right', prop={'size': 10})

    plt.show()
