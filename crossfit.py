# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 04:53:21 2025

@author: George
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.stats import mannwhitneyu
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression

def compute_vif(df, cols):
    X = df[cols].dropna()
    X = X.assign(intercept=1)

    vif_data = pd.DataFrame({
        'feature': cols,
        'VIF': [variance_inflation_factor(X.values, i) for i in range(len(cols))]
    })
    
    return vif_data

def remove_outliers(df, col_list, lower=0.01, upper=0.99):
    filtered_df = df.copy()
    for col in col_list:
        filtered_df = filtered_df[~filtered_df[col].isna()].copy()
        low = df[col].quantile(lower)
        high = df[col].quantile(upper)
        filtered_df = filtered_df[(filtered_df[col] >= low) & (filtered_df[col] <= high)].copy()
    return filtered_df

def lbs_to_kg(pounds):
    return pounds * 0.45359237

def all_hists(df, title):
    h = df.run5k.hist(bins=20)
    h.set_title(f"{title} - Run 5K")
    h.set_xlabel("minutes")
    plt.show()
    h = df.deadlift.hist(bins=20)
    h.set_title(f"{title} - Deadlift")
    h.set_xlabel("KGs")
    plt.show()
    h = df.age.hist()
    h.set_title(f"{title} - Age")
    h.set_xlabel("Years")
    
def compare_hist(df1, col1, col1_name, df2, col2, col2_name, title, xlabel):
    df1[col1].hist(alpha=0.5, label=col1_name, bins=20)
    df2[col2].hist(alpha=0.5, label=col2_name, bins=20)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.legend()
    plt.show()
    
def smart_scatter(df, col1, col2, title):
    sns.kdeplot(
        data=df, x=col1, y=col2, fill=True, cmap="Blues", thresh=0.05
    )
    plt.scatter(df[col1], df[col2], s=5, color='black', alpha=0.3)
    plt.title(title)
    plt.show()
    
def smarter_scatter(df, col1, col2, title):
    # Calculate the point density
    x = df[col1]
    y = df[col2]
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    
    # Plot
    plt.scatter(x, y, c=z, s=10, cmap='viridis')
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title(title)
    plt.colorbar(label='Density')
    plt.show()
    
def grid_scatter(df, col1, col2, title):
    df.plot.hexbin(x=col1, y=col2, gridsize=30, cmap='Blues')
    plt.title(title)
    plt.show()
    
def plot_ecdf(df, col, title="ECDF"):
    # Drop NA values for safety
    data = df[col].dropna()
    
    # Sort the data
    x = np.sort(data)
    
    # Calculate ECDF values
    y = np.arange(1, len(x) + 1) / len(x)
    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.step(x, y, where='post')
    plt.xlabel(col)
    plt.ylabel('ECDF')
    plt.title(title)
    plt.grid(True)
    plt.show()
    
def normallity_tests(data):
    stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
    
    #stats.anderson(data, dist='norm')
    
    #stats.shapiro(data)
    
    stats.normaltest(data)
    
    
def plot_binned_means(df, x_col, y_col, num_bins=10):
    """
    Plots the mean of `y_col` values in bins of `x_col` values.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        x_col (str): Name of the column to use for x-axis.
        y_col (str): Name of the column to average over for y-axis.
        num_bins (int): Number of bins to split `x_col` into.
    """
    # Drop NA values
    data = df[[x_col, y_col]].dropna()

    # Create bins
    bins = np.linspace(data[x_col].min(), data[x_col].max(), num_bins + 1)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    # Cut x_col into bins and calculate mean y_col per bin
    data['bin'] = pd.cut(data[x_col], bins=bins)
    means = data.groupby('bin')[y_col].mean().values

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(bin_centers, means, marker='o')
    plt.xlabel(x_col)
    plt.ylabel(f'Mean of {y_col}')
    plt.title(f'Mean {y_col} per {x_col} bin')
    plt.grid(True)
    plt.show()


def linreg(df_man):
    import statsmodels.api as sm
    
    x = df_man[["age", "deadlift", "run5k"]]
    x = sm.add_constant(x)
    y = df_man.fran
    
    model = sm.OLS(y, x).fit()
    
    print(model.summary())
    

def howlong2month(howlong):
    replace_dict = {
        "2-4 years": 36,
        "1-2 years": 18,
        "4+ years": 60,
        "6-12 months": 9,
        "Less than 6 months": 3,
        "1-2 years|2-4 years": (18 + 36) / 2,                          # 27.0
        "6-12 months|1-2 years": (9 + 18) / 2,                         # 13.5
        "2-4 years|4+ years": (36 + 60) / 2,                           # 48.0
        "Less than 6 months|1-2 years": (3 + 18) / 2,                  # 10.5
        "Decline to answer": None,
        "Less than 6 months|6-12 months": (3 + 9) / 2,                 # 6.0
        "6-12 months|2-4 years": (9 + 36) / 2,                         # 22.5
        "Less than 6 months|2-4 years": (3 + 36) / 2,                  # 19.5
        "1-2 years|4+ years": (18 + 60) / 2,                           # 39.0
        "6-12 months|1-2 years|2-4 years": (9 + 18 + 36) / 3,          # 21.0
        "Less than 6 months|6-12 months|1-2 years": (3 + 9 + 18) / 3,  # 10.0
        "6-12 months|1-2 years|2-4 years|4+ years": (9 + 18 + 36 + 60) / 4,  # 30.75
        "2-4 years|Decline to answer": None,
        "6-12 months|4+ years": (9 + 60) / 2,                          # 34.5
        "Less than 6 months|6-12 months|1-2 years|2-4 years|4+ years|Decline to answer": None,
        "Less than 6 months|4+ years": (3 + 60) / 2,                   # 31.5
        "Less than 6 months|Decline to answer": None,
        "1-2 years|Decline to answer": None
    }
    
    if howlong is None:
        return None

    if type(howlong) == float:
        return None
    
    if howlong[:-1] in replace_dict:
        return replace_dict[howlong[:-1]]
    return None


def slope_change(df, x, y, col, neg = False, steps=20):
    slopes = []
    values = range(int(df[col].min()), int(df[col].max()))
    v = []
    for i in values:
        dd = df[(df[col] >= i) & (df[col] < i + steps) ].copy()
        if len(dd) > 0.1 * len(df):
            model = LinearRegression()  # fit_intercept=True by default
            model.fit(dd[[x]], dd[y])
            
            slopes.append(model.coef_[0])
            v.append(i)
        
    plt.plot(v, slopes)
    plt.xlabel(col)
    plt.ylabel(f"linreg slope of {x} to {y}")
    plt.title(f"{x} effect on {y} as a function of {col}")
    return v
    
def sample_compare_slopes(df1, df2, x, y, niter = 1000):
    df1_data = []
    df2_data = []
    for i in range(niter):
        dd1 = df1.sample(int(0.2 * len(df1)))
        
        model = LinearRegression()  # fit_intercept=True by default
        model.fit(dd1[[x]], dd1[y])
        
        df1_data.append(model.coef_[0])
        
        dd2 = df2.sample(int(0.2 * len(df2)))
        
        model = LinearRegression()  # fit_intercept=True by default
        model.fit(dd2[[x]], dd2[y])
        
        df2_data.append(model.coef_[0])
    
    ddd = pd.DataFrame({"c1": df1_data, "c2": df2_data})
    ddd["c1"].hist(alpha=0.5, label="Deadlift >= 170", bins=20)
    ddd["c2"].hist(alpha=0.5, label="Deadlift < 170", bins=20)
    
    plt.title("Strong vs Weak age effects")
    plt.xlabel("Slope of the age effect on Fran performance")
    plt.legend()
    plt.show()
        

filename = "C:/Users/George/Dropbox/learning/stats/athletes.csv"
def load_data(filename):
    df = pd.read_csv(filename)
    df = df[~df.gender.isna()].copy()
    df = remove_outliers(df, ["fran", "deadlift", "run5k", "age"])
    df["deadlift"] = df.deadlift.apply(lbs_to_kg)
    df.run5k = df.run5k.apply(lambda x: x / 60.)
    df.fran = df.fran.apply(lambda x: x / 60.)
    df["howlong_num"] = df.howlong.apply(howlong2month)
    df = df.dropna(subset = ["fran", "deadlift", "run5k", "age", "howlong_num"]).copy()
    df["experienced"] = df.howlong_num > 48
    
    all_hists(df, "Everyone")
    
    df_man = df[df.gender == "Male"].copy()
    df_woman = df[df.gender == "Female"].copy()
    
    df_man = remove_outliers(df_man, ["fran", "deadlift", "run5k", "age"])
    df_woman = remove_outliers(df_woman, ["fran", "deadlift", "run5k", "age"])
                             
    all_hists(df_man, "Man")
    all_hists(df_woman, "Woman")
    
    compare_hist(df_man, "deadlift", "Man deadlift", 
                 df_woman, "deadlift", "Woman deadlift", 
                 "Deadlift Man vs Woman", "KGs")
    
    compare_hist(df_man, "run5k", "Man 5K", 
                 df_woman, "run5k", "Woman 5K", 
                 "5K run Man vs Woman", "minutes")
    
    compare_hist(df_man, "age", "Man's age", 
                 df_woman, "age", "Woman's age", 
                 "Age Man vs Woman", "years")
    
    compare_hist(df_man, "fran", "Man's fran time", 
                 df_woman, "fran", "Woman's fran time", 
                 "Fran Man vs Woman", "minutes")
    
    plot_binned_means(df_man, "age", "fran")
    plot_binned_means(df_man, "age", "deadlift")
    plot_binned_means(df_man, "age", "run5k")
    
    df_man_exp = df_man[df_man.howlong_num > 12].copy()
    sns.regplot(data=df_man_exp, x='deadlift', y='fran', x_bins=15)
    sns.regplot(data=df_man_exp, x='run5k', y='fran', x_bins=15)
    sns.regplot(data=df_man_exp, x='age', y='fran', x_bins=15)
    sns.regplot(data=df_man_exp, x='howlong_num', y='fran', x_bins=15)
    
    sns.regplot(data=df_man, x='age', y='fran', x_bins=15)
    sns.regplot(data=df_man_exp, x='age', y='deadlift', x_bins=15)
    sns.regplot(data=df_man, x='age', y='run5k', x_bins=15)
    
    plot_binned_means(df_man, "deadlift", "fran")
    
    plot_binned_means(df_man, "run5k", "fran")
    
    plot_binned_means(df_man, "deadlift", "run5k")
    
    sns.lmplot(x="age", y="fran", data = df_man, line_kws={'color': 'red'})
    
    smart_scatter(df_man, "age", "fran", "Fran per age group for man")
    smarter_scatter(df_man, "age", "fran", "Fran per age group for man")
    grid_scatter(df_man, "age", "fran", "Fran per age group for man")
    
    plot_ecdf(df_man, "fran", "Man's fran ECDF")
    plot_ecdf(df_man, "age", "Man's fran ECDF")
    
    young_man = df_man[(df_man.age < 35) & (df_man.howlong_num >= 12)].copy()
    old_man = df_man[(df_man.age >= 35) & (df_man.howlong_num >= 12)].copy()
    
    compare_hist(young_man, "fran", "Young man's fran time", 
                 old_man, "fran", "Old man's fran time", 
                 "Fran Young man vs Old man", "minutes")
    
    stat, p_value = mannwhitneyu(young_man["fran"], old_man.fran, 
                                 alternative='less')

    # Output the result
    print(f"U statistic = {stat}")
    print(f"p-value = {p_value}")
    
    young_man = df_man[(df_man.age < 35) & (df_man.howlong_num >= 12) & (df_man.deadlift < 180)].copy()
    old_man = df_man[(df_man.age >= 35) & (df_man.howlong_num >= 12) & (df_man.deadlift >= 180)].copy()
    
    compare_hist(young_man, "fran", "Young man's fran time", 
                 old_man, "fran", "Old man's fran time", 
                 "Fran Young man vs Old man", "minutes")
    
    stat, p_value = mannwhitneyu(young_man["fran"], old_man.fran, 
                                 alternative='greater')

    # Output the result
    print(f"U statistic = {stat}")
    print(f"p-value = {p_value}")
    
    young_man = df_man[(df_man.age < 35) & (df_man.howlong_num >= 12) & (df_man.run5k < 180)].copy()
    old_man = df_man[(df_man.age >= 35) & (df_man.howlong_num >= 12) & (df_man.deadlift >= 180)].copy()
    
    compare_hist(young_man, "fran", "Young man's fran time", 
                 old_man, "fran", "Old man's fran time", 
                 "Fran Young man vs Old man", "minutes")
    
    stat, p_value = mannwhitneyu(young_man["fran"], old_man.fran, 
                                 alternative='greater')

    # Output the result
    print(f"U statistic = {stat}")
    print(f"p-value = {p_value}")
    
    stat, p_value = mannwhitneyu(young_man.run5k, old_man.run5k, 
                                 alternative='less')

    # Output the result
    print(f"U statistic = {stat}")
    print(f"p-value = {p_value}")
    
    stat, p_value = mannwhitneyu(old_man.deadlift, young_man.deadlift, 
                                 alternative='less')

    # Output the result
    print(f"U statistic = {stat}")
    print(f"p-value = {p_value}")
    
    
    
    stats.probplot(df_man.fran, dist="norm", plot=plt)
    plt.title('Q-Q Plot of mans fran times')
    plt.show()
    
    stats.probplot(df_man.run5k, dist="norm", plot=plt)
    plt.title('Q-Q Plot of mans 5k times')
    plt.show()
    
    stats.probplot(df_man.deadlift, dist="norm", plot=plt)
    plt.title('Q-Q Plot of mans deadlift')
    plt.show()
    
    
    stats.probplot(df_woman.fran, dist="norm", plot=plt)
    plt.title('Q-Q Plot of womans fran time')
    plt.show()
    
    stats.probplot(df_woman.deadlift, dist="norm", plot=plt)
    plt.title('Q-Q Plot of womans deadlift')
    plt.show()
    
    stats.probplot(df_woman.run5k, dist="norm", plot=plt)
    plt.title('Q-Q Plot of womans run 5k')
    plt.show()
    
    
    x = df_man[["deadlift", "age", "run5k", "fran", "howlong_num"]].corr()
    sns.heatmap(df_man[["deadlift", "age", "run5k", "fran", "howlong_num"]].corr(), 
                annot=True, cmap="coolwarm", fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()
    
    ddf_man_exp = df_man_exp[~pd.isnull(df_man_exp.howlong_num)].copy()
    x = ddf_man_exp[["age", "deadlift", "run5k", "howlong_num"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(x)

    x = sm.add_constant(X_scaled)
    y = ddf_man_exp.fran
    
    model = sm.OLS(y, x).fit()
    
    print(model.summary())
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
