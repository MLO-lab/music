import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import pandas as pd
<<<<<<< HEAD
from sklearn.metrics import mean_squared_error
=======
#from sklearn.metrics import mean_squared_error
>>>>>>> dev

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
    
<<<<<<< HEAD
=======
    # Reshape data to long format
    print(data)
    df_long = pd.melt(data, id_vars=[x_lab], value_vars=['Model', 'Tensorly'], 
                      var_name='RMSE', value_name='Y')
    
    # Create boxplot
    sns.set(font_scale=3.5)
    sns.boxplot(x=x_lab, y='Y', hue='RMSE', data=df_long)
    plt.ylim(bottom=0.7)
    #ax.text(x=0.5, y=1.1, s=title, fontsize=20, weight='bold', ha='center', va='bottom', transform=ax.transAxes)
    #subtitle = "R={}, n_features={}".format(data.R.unique(), data.n_features.unique())
    ax.set_xlabel("Number of samples")
    ax.set_ylabel("RMSE")
    # Calculate means for each category
    #print(data)
    #print(df_long)
    #category_means = df_long.groupby('RMSE')['Y'].mean()

    # Plot the means as a line across the categories
    #plt.plot(category_means.index, category_means.values, color='red', marker='o', linestyle='-', linewidth=2, label='Mean')


    '''
>>>>>>> dev
    data.R = data.R.astype(str)
    data.n_samples = data.n_samples.astype(str) # find another way with xlab
    grouped_df = data.groupby(x_lab).agg({'RMSE': 'mean', 'time': 'mean'}).reset_index()
    
    sns.boxplot(data=data, x=x_lab, y="RMSE", ax=ax)
<<<<<<< HEAD
    ax.set_xticklabels(sorted(data[x_lab].astype(int).unique()), size = 40)
    ax.set_yticklabels(ax.get_yticks(), size = 40)
=======
>>>>>>> dev
    ax.set_xlabel(x_lab, fontsize=40)
    ax.set_ylabel("RMSE",fontsize=40)
    
    ax.text(x=0.5, y=1.1, s=title, fontsize=40, weight='bold', ha='center', va='bottom', transform=ax.transAxes)
    subtitle = "R={}, n_features={}".format(data.R.unique(), data.n_features.unique())
    ax.text(x=0.5, y=1.05, s=subtitle, fontsize=20, alpha=0.75, ha='center', va='bottom', transform=ax.transAxes)
    h,l = ax.get_legend_handles_labels()
    ax.legend(h[:4],l[:4], bbox_to_anchor=(1.05, 1), loc=2)
<<<<<<< HEAD
    
    ax2 = ax.twinx()
    sns.lineplot(data=grouped_df, x=x_lab, y="time", marker="o", ax=ax2, color='r', label="time")
    ax2.set_ylabel("Time", fontsize=40)
    ax2.set_ylim([data["time"].min(), data["time"].max()])
    ax2.set_xticklabels(sorted(data[x_lab].astype(str).unique()), size = 40)
    ax2.set_yticklabels(ax2.get_yticks(), size = 40)
=======

    ax2 = ax.twinx()
    sns.boxplot(data=data, x=x_lab, y="RMSE_Tensorly", ax=ax2)
    ax2.set_ylabel("RMSE_Tensorly", fontsize=40)
    df_long = pd.melt(df, id_vars=['X'], value_vars=['Y1', 'Y2'], 
                  var_name='Measure', value_name='Y')0
    '''

>>>>>>> dev

    return(plt)
