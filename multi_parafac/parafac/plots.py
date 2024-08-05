import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import pandas as pd
from sklearn.metrics import mean_squared_error

def RMSELoss(yhat, y):
    return torch.sqrt(torch.mean((yhat-y)**2))

def distplots(
    data: dict,
    keyorder: list,
    figsize: tuple = (15, 12),
    hspace: float = 0.5,
    fontsize: float = 18,
    y: float = 0.95,
    **kwargs):
    
    # define subplot grid
    plt.figure(figsize=figsize)
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle("Factor matrices", fontsize=18, y=0.95)

    # loop through tickers and axes
    for n, key in enumerate(keyorder):

        ax = plt.subplot(4, 2, n + 1)
        # filter df for ticker and plot on specified axes
        sample = data[key]
        #TODO: .cpu() might not work with cpu device
        sns.histplot(data=pd.DataFrame(sample.cpu().ravel()), ax=ax, stat="density", **kwargs)

        # chart formatting
        ax.set_title(key.upper())
        ax.get_legend().remove()
        ax.set_xlabel("")
    
    return plt

def RMSE_time(data, x_lab, title, figsize=(20,15)):
    #TODO: size
    fig, ax = plt.subplots(figsize=figsize)
    
    data.R = data.R.astype(str)
    data.n_samples = data.n_samples.astype(str) # find another way with xlab
    grouped_df = data.groupby(x_lab).agg({'RMSE': 'mean', 'time': 'mean'}).reset_index()
    
    sns.boxplot(data=data, x=x_lab, y="RMSE", ax=ax)
    ax.set_xticklabels(sorted(data[x_lab].astype(int).unique()), size = 40)
    ax.set_yticklabels(ax.get_yticks(), size = 40)
    ax.set_xlabel(x_lab, fontsize=40)
    ax.set_ylabel("RMSE",fontsize=40)
    
    ax.text(x=0.5, y=1.1, s=title, fontsize=40, weight='bold', ha='center', va='bottom', transform=ax.transAxes)
    subtitle = "R={}, n_features={}".format(data.R.unique(), data.n_features.unique())
    ax.text(x=0.5, y=1.05, s=subtitle, fontsize=20, alpha=0.75, ha='center', va='bottom', transform=ax.transAxes)
    h,l = ax.get_legend_handles_labels()
    ax.legend(h[:4],l[:4], bbox_to_anchor=(1.05, 1), loc=2)
    
    ax2 = ax.twinx()
    sns.lineplot(data=grouped_df, x=x_lab, y="time", marker="o", ax=ax2, color='r', label="time")
    ax2.set_ylabel("Time", fontsize=40)
    ax2.set_ylim([data["time"].min(), data["time"].max()])
    ax2.set_xticklabels(sorted(data[x_lab].astype(str).unique()), size = 40)
    ax2.set_yticklabels(ax2.get_yticks(), size = 40)

    return(plt)
