import numpy as np
from scipy import stats
from scipy.stats import variation
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from scipy.optimize import minimize

path = './data/'

# Configuration variables
PLOT_CONFIG = {
    'figsize': (12, 8),
    'dpi': 100,
    'error_bounds': 0.1,  # ±10%
    'marker_size': 60,
    'marker_alpha': 0.5,
    'line_width': 1.5,
    'grid_alpha': 0.3,
    'font_size': 18,
    'title_size': 24,
    'legend_size': 16,
    'tick_size': 18
}

COLOR_SCHEME = {
    'heating_points': "#FF0000",  # Crimson red for heating
    'cooling_points': "#0011FF",  # Professional blue for cooling
    'parity_line': "#000000",  # Deep magenta
    'error_bounds': "#C72E2EFF",  # Orange
    'error_fill': "#FFFFFF00",
    'grid': '#CCCCCC'
}

mpl.rcParams['font.family'] = 'Times New Roman'

def create_frequency_bin_plot(dataset1, dataset2, bins=30, labels=None, title="",bin_range=None):
    """
    Create overlaid frequency bin plots for comparing two datasets.
    
    Parameters:
    -----------
    dataset1, dataset2 : array-like
        Input datasets to compare
    bins : int, default=30
        Number of histogram bins
    labels : tuple, optional
        Labels for (dataset1, dataset2). Default: ("Dataset 1", "Dataset 2")
    title : str
        Plot title
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    
    # Convert to numpy and remove NaN values
    data1 = np.array(dataset1)
    data2 = np.array(dataset2)
    data1 = data1[~np.isnan(data1)]
    data2 = data2[~np.isnan(data2)]
    
    # Set default labels
    if labels is None:
        labels = ("Dataset 1", "Dataset 2")
        
        # Determine bin range if not provided
    if bin_range is None:
        min_val = min(np.min(data1), np.min(data2))
        max_val = max(np.max(data1), np.max(data2))
        bin_range = (min_val, max_val)
    
    # Create figure
    fig, ax = plt.subplots(figsize=PLOT_CONFIG['figsize'], dpi=PLOT_CONFIG['dpi'])
    
# Create histograms
    ax.hist(data1, bins=bins, alpha=PLOT_CONFIG['marker_alpha'], 
            color=COLOR_SCHEME['heating_points'], label=labels[0], edgecolor='white', linewidth=0.5)
    ax.hist(data2, bins=bins, alpha=PLOT_CONFIG['marker_alpha'], 
            color=COLOR_SCHEME['cooling_points'], label=labels[1], edgecolor='white', linewidth=0.5)
    
    # Add mean lines
    mean1 = np.mean(data1)
    mean2 = np.mean(data2)
    
    ax.axvline(mean1, color='red', linestyle='--', linewidth=2, 
               label=f'DC mean: {mean1:.1f} °Ch')
    ax.axvline(mean2, color='blue', linestyle='--', linewidth=2, 
               label=f'AC mean: {mean2:.1f} °Ch')
    
    # Formatting
    ax.set_xlabel('Daily average indoor-outdoor temperature difference [°C]', fontsize=PLOT_CONFIG['font_size'])
    ax.set_ylabel('Frequency', fontsize=PLOT_CONFIG['font_size'])
    plt.xticks(fontsize=PLOT_CONFIG['tick_size'])  # Sets x-axis tick label font size
    plt.yticks(fontsize=PLOT_CONFIG['tick_size']) 
    ax.set_title(title, fontsize=PLOT_CONFIG['title_size'], fontweight='bold')
    ax.grid(True, alpha=PLOT_CONFIG['grid_alpha'], color=COLOR_SCHEME['grid'])
    ax.legend(fontsize=PLOT_CONFIG['legend_size'])
    
    plt.tight_layout()
    return fig, ax

def hdh(year,HP,Tbal):

    AHU_main = pd.read_csv(path+year+HP[0])
    AHU_Aux = pd.read_csv(path+year+HP[1])
    
    AHU = pd.DataFrame()
    AHU['Time'] = pd.to_datetime(AHU_main['Time'])
    AHU['Value'] = (AHU_Aux['Value'] + AHU_main['Value']) / 1000
    AHU_hourly = AHU.groupby(AHU['Time'].dt.floor('h'))['Value'].sum().reset_index()
    AHU_hourly['Value'] = AHU_hourly['Value'] / 12
   
    OU = pd.read_csv(path+year+HP[2])
    OU['Time'] = pd.to_datetime(OU['Time'])
    OU['Value'] = OU['Value'] / 1000
    OU_hourly = OU.groupby(OU['Time'].dt.floor('h'))['Value'].sum().reset_index()
    OU_hourly['Value'] = OU_hourly['Value'] / 12

    Tset = pd.read_csv(path+year+HP[3])
    Tset['Time'] = pd.to_datetime(Tset['Time'])
    Tset_hourly = Tset.groupby(Tset['Time'].dt.floor('h'))['Value'].mean().reset_index()
    
    weather = pd.read_csv(path+year+'/weather_'+year+'.csv')
    Tamb = weather['temperature (degC)']

    HDH = pd.DataFrame()
    HDH['Time'] = weather.T.head(1).T
    HDH['Time'] = pd.to_datetime(HDH['Time'])
    HDH['Tset (C)'] = Tset_hourly['Value']
    HDH['Tamb (C)'] = weather['temperature (degC)']
    HDH['OU (kWh)'] = OU_hourly['Value']
    HDH['AHU (kWh)'] = AHU_hourly['Value']
    HDH['Etot (kWh)'] = OU_hourly['Value'] + AHU_hourly['Value']
    HDH['HDH (C)'] = Tbal - Tamb
    HDH['Eff (kWh/C)'] = HDH['Etot (kWh)'] / HDH['HDH (C)']
    HDH = HDH.dropna()
    HDH = HDH.set_index('Time')
    return HDH

def historical_hdh(data, outlier_method='iqr', outlier_columns=None, z_threshold=3, iqr_factor=1.5):
    """
    Process historical heat pump data with outlier removal.
    
    Parameters:
    - data: CSV file path or DataFrame
    - outlier_method: 'iqr', 'zscore', 'isolation', or 'all'
    - outlier_columns: List of columns to check for outliers (default: energy columns)
    - z_threshold: Z-score threshold for outlier detection
    - iqr_factor: IQR factor for outlier detection
    """
    
    Tbal = 20.556
    ACHP = pd.read_csv(data) if isinstance(data, str) else data.copy()
    
    # Original filtering
    ACHP = ACHP[(ACHP['Mode'] == 0) & 
                (ACHP['Rounded Setpoints'] == 20.5) & 
                (ACHP['OutdoorTemp_C_'] >= -16.5) & 
                (ACHP['OutdoorTemp_C_'] <= 13.85)]
    
    # Feature engineering
    ACHP['Time'] = pd.to_datetime(ACHP['Time'])
    ACHP['HDH (C)'] = Tbal - ACHP['OutdoorTemp_C_']
    ACHP['OU (kWh)'] = ACHP['Outdoor Unit [W]'] / 1000
    ACHP['AHU (kWh)'] = ACHP['AHU and AUX1 [W]'] / 1000 + ACHP['AUX2  [W]'] / 1000
    ACHP['Etot (kWh)'] = ACHP['OU (kWh)'] + ACHP['AHU (kWh)']
    ACHP['Eff (kWh/C)'] = ACHP['Etot (kWh)'] / ACHP['HDH (C)']
    
    ACHP = ACHP.set_index('Time')
    ACHP = ACHP.dropna()
    
    # Define columns to check for outliers
    if outlier_columns is None:
        outlier_columns = ['OU (kWh)', 'AHU (kWh)', 'Etot (kWh)', 'Eff (kWh/C)']
    
    # Remove outliers
    if outlier_method == 'iqr':
        ACHP = remove_outliers_iqr(ACHP, outlier_columns, iqr_factor)
    elif outlier_method == 'zscore':
        ACHP = remove_outliers_zscore(ACHP, outlier_columns, z_threshold)
    elif outlier_method == 'isolation':
        ACHP = remove_outliers_isolation(ACHP, outlier_columns)
    elif outlier_method == 'all':
        ACHP = remove_outliers_combined(ACHP, outlier_columns, z_threshold, iqr_factor)
    
    return ACHP

def remove_outliers_iqr(df, columns, factor=1.5):
    """Remove outliers using Interquartile Range method"""
    df_clean = df.copy()
    
    for col in columns:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    return df_clean

def remove_outliers_zscore(df, columns, threshold=3):
    """Remove outliers using Z-score method"""
    df_clean = df.copy()
    
    for col in columns:
        if col in df_clean.columns:
            z_scores = np.abs(stats.zscore(df_clean[col]))
            df_clean = df_clean[z_scores < threshold]
    
    return df_clean

def remove_outliers_isolation(df, columns, contamination=0.1):
    """Remove outliers using Isolation Forest"""
    try:
        from sklearn.ensemble import IsolationForest
        
        df_clean = df.copy()
        
        # Apply isolation forest to specified columns
        isolation_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_mask = isolation_forest.fit_predict(df_clean[columns])
        
        # Keep only inliers (outlier_mask == 1)
        df_clean = df_clean[outlier_mask == 1]
        
        return df_clean
    
    except ImportError:
        print("Warning: sklearn not available. Using IQR method instead.")
        return remove_outliers_iqr(df, columns)

def remove_outliers_combined(df, columns, z_threshold=3, iqr_factor=1.5):
    """Remove outliers using combination of methods"""
    df_clean = df.copy()
    
    # First apply IQR method
    df_clean = remove_outliers_iqr(df_clean, columns, iqr_factor)
    
    # Then apply Z-score method
    df_clean = remove_outliers_zscore(df_clean, columns, z_threshold)
    
    return df_clean

def outlier_analysis(df, columns):
    """Analyze outliers in the dataset"""
    outlier_info = {}
    
    for col in columns:
        if col in df.columns:
            # IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
            # Z-score method
            z_scores = np.abs(stats.zscore(df[col]))
            zscore_outliers = df[z_scores >= 3]
            
            outlier_info[col] = {
                'iqr_outliers': len(iqr_outliers),
                'zscore_outliers': len(zscore_outliers),
                'min_value': df[col].min(),
                'max_value': df[col].max(),
                'mean': df[col].mean(),
                'std': df[col].std()
            }
    
    return outlier_info

# Example usage with outlier analysis
def process_with_analysis(data_path):
    """Process data and show outlier analysis"""
    # Load and process without outlier removal
    original_data = historical_hdh(data_path, outlier_method=None)
    
    # Analyze outliers
    energy_cols = ['OU (kWh)', 'AHU (kWh)', 'Etot (kWh)', 'Eff (kWh/C)']
    outlier_info = outlier_analysis(original_data, energy_cols)
    
    print("Outlier Analysis:")
    for col, info in outlier_info.items():
        print(f"{col}: IQR outliers: {info['iqr_outliers']}, Z-score outliers: {info['zscore_outliers']}")
    
    # Process with outlier removal
    clean_data = historical_hdh(data_path, outlier_method='iqr')
    
    print(f"\nData points before outlier removal: {len(original_data)}")
    print(f"Data points after outlier removal: {len(clean_data)}")
    print(f"Removed: {len(original_data) - len(clean_data)} points ({((len(original_data) - len(clean_data))/len(original_data)*100):.1f}%)")
    
    return clean_data

def power_integral(hdh,hdh_dc,ac_fit):
    hdh = hdh.to_numpy()
    #print(hdh)
    #print(dc_fit(hdh))
    dc_integral = np.sum(hdh_dc)
    ac_integral = np.sum(ac_fit(hdh))
    print("Percent difference: \n", (dc_integral - ac_integral) / ac_integral)
    
    print("Integrals of DC and AC Power: \n",dc_integral,ac_integral)
    
    plt.figure(figsize=PLOT_CONFIG['figsize'], dpi=PLOT_CONFIG['dpi'])
    name = ['DC Energy\n[Measured]', 'AC Power\n[Modeled]']
    eff = [dc_integral, ac_integral]
    colors = ['#E41A1C', '#377EB8']
    error = 45
    bars = plt.bar(name, eff, color=colors, alpha=0.8, edgecolor='black', linewidth=0.8)
    plt.errorbar(name, eff, yerr=error, fmt='none', 
             color='black', capsize=15, capthick=1.5, 
             elinewidth=1.5, alpha=0.8)
    # Add value labels on bars
    for bar, value in zip(bars, eff):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + error + 5,
                f'{value:.1f}', ha='center', va='bottom', fontsize=PLOT_CONFIG['legend_size'], fontweight='bold')
    plt.xlabel("Power Input Configuration", fontsize=PLOT_CONFIG['font_size'])
    plt.ylabel("Total Energy Consumed [kWh]", fontsize=PLOT_CONFIG['font_size'])
    # Improve tick formatting
    plt.xticks(fontsize=PLOT_CONFIG['tick_size'])
    plt.yticks(fontsize=PLOT_CONFIG['tick_size'])
    #plt.show()
    
    
    return 

def create_parity_plot(ac, dc, config=PLOT_CONFIG, colors=COLOR_SCHEME):
    """
    Create a scientific parity plot with separate heating and cooling data.
    
    Parameters:
    ac, dc: numpy arrays of HP data
    config: dictionary of plot configuration parameters
    colors: dictionary of color scheme
    """
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=config['figsize'], dpi=config['dpi'])

    
    # Calculate plot limits with padding
    min_val = min(min(ac), min(dc)) * 0.9
    max_val = max(max(ac), max(dc)) * 1.1
    
    # Create parity line (y=x)
    parity_x = np.linspace(min_val, max_val, 100)
    parity_y = parity_x
    
    # Plot error bounds (±5%)
    upper_bound = parity_y * (1 + config['error_bounds'])
    lower_bound = parity_y * (1 - config['error_bounds'])
    
    # Fill error bounds region
    ax.fill_between(parity_x, lower_bound, upper_bound, 
                    alpha=0.2, color=colors['error_fill'], 
                    label=f'±{config["error_bounds"]*100:.0f}% bounds')
    
    # Plot error bound lines
    ax.plot(parity_x, upper_bound, '--', color=colors['error_bounds'], 
            linewidth=config['line_width'], alpha=0.8)
    ax.plot(parity_x, lower_bound, '--', color=colors['error_bounds'], 
            linewidth=config['line_width'], alpha=0.8)
    
    # Plot parity line
    ax.plot(parity_x, parity_y, '-', color=colors['parity_line'], 
            linewidth=config['line_width'], label='Perfect parity (y=x)')
    
    # Plot cooling data points
    ax.scatter(ac, dc, s=config['marker_size'], 
                             color=colors['cooling_points'], alpha=config['marker_alpha'],
                             edgecolors='white', linewidth=1, label='Cooling data')
    
    
    # Calculate and display statistics for both datasets
    residuals = dc - ac
    
    overall_rmse = np.sqrt(np.mean(residuals**2))
    
    overall_mape = np.mean(np.abs(residuals / ac)) * 100
    
    overall_r2 = np.corrcoef(ac, dc)[0, 1]**2
    
    # Add statistics text box
    stats_text = f'Overall: n = {len(ac)}, RMSE = {overall_rmse:.3f} kW, MAPE = {overall_mape:.1f}%, R² = {overall_r2:.3f}\n'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='white', alpha=0.8), fontsize=config['legend_size'])
    
    # Formatting
    ax.set_xlabel("AC Heat Pump Thermal Capacity [kW]", fontsize=config['font_size'])
    ax.set_ylabel("DC Heat Pump Thermal Capacity [kW]", fontsize=config['font_size'])
    #ax.set_title("Heat Pump Thermal Capacity Parity Analysis", 
                #fontsize=config['title_size'], fontweight='bold')
    
    # Set equal aspect ratio and limits
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect('equal')
    
    # Add grid
    ax.grid(True, alpha=config['grid_alpha'], color=colors['grid'])
    
    # Add legend
    ax.legend(loc='lower right', fontsize=config['legend_size'])
    
    # Improve layout
    plt.tight_layout()
    
    return fig, ax

year = 'data/2324'
data = '/DC_House_Nov_April.csv'
HDH_AC = historical_hdh(year+data, outlier_method='iqr', iqr_factor=5)
Tbal = 20.556

year = '2425'
DCHP = ['/AHU_main_return.csv','/AHU_Aux_return.csv','/BDinverter_DC_power_OUT_return.csv','/heating_setpoint_return.csv']
HDH_DC = hdh(year,DCHP,Tbal)



# figure - HDH frequency plot
create_frequency_bin_plot(HDH_DC['HDH (C)'], HDH_AC['HDH (C)'], bins=30, labels=['2024 DC data', '2023 AC data'], title="")
plt.savefig("./figs/HDH_frequency_plot.png", dpi=400, bbox_inches='tight')

print("Minimum ambient temperature seen by the DCHP")
print(np.max(HDH_DC['Tamb (C)']))
print("Total Effectiveness (kWh/C) of DCHP")
print(HDH_DC['Eff (kWh/C)'].mean())

print("\nMinimum ambient temperature seen by the ACHP")
print(np.min(HDH_AC['OutdoorTemp_C_']))
print("Total Effectiveness (kWh/C) of ACHP")
print(HDH_AC['Eff (kWh/C)'].mean())

# Marker parameters
alpha_scatter = 0.25 # scatter plot transparency
alpha_line = 1      # line plot transparency
scatter_style = 'o' # marker shape for scatter plots
line_style = '--'   # marker shape for line plots
width = 3.0         # line width setting
s = 30              # marker size
edgecolor = 'none'  # edge style of scatter plot markers

# colors
DC_color = '#E41A1C'
AC_color = '#377EB8'

# Titles and labels
label = 16              # x-y axis label font size
title = 14              # title font size
legend = 10             # legend font size
font = "Times New Roman"         # choice of font
font_fam = "serif" # font family

plt.rcParams["font.family"] = font_fam
plt.rcParams["font.sans-serif"] = [font]

# confidence interval analysis
dc_model = np.polyfit(HDH_DC['HDH (C)'],HDH_DC['OU (kWh)'],2,full=True)
poly_dc = np.poly1d(dc_model[0])
dc_resid = dc_model[1]

ac_model = np.polyfit(HDH_AC['HDH (C)'],HDH_AC['OU (kWh)'],2,full=True)
poly_ac = np.poly1d(ac_model[0])
ac_resid = ac_model[1]

z_star = 1.645
std_dc = stats.tstd(HDH_DC['OU (kWh)'])
std_ac = stats.tstd(HDH_AC['OU (kWh)'])

mean_dc = np.mean(HDH_DC['OU (kWh)'])
mean_ac = np.mean(HDH_AC['OU (kWh)'])

TSS_dc = np.sum((HDH_DC['OU (kWh)'] - mean_dc)**2)
TSS_ac = np.sum((HDH_AC['OU (kWh)'] - mean_ac)**2)

dc_rows, dc_columns = HDH_DC.shape
ac_rows, ac_columns = HDH_AC.shape

CI_dc = z_star * (std_dc/np.sqrt(dc_rows))
CI_ac = z_star * (std_ac/np.sqrt(ac_rows))

Rsq_dc = 1 - dc_resid/TSS_dc
Rsq_ac = 1 - ac_resid/TSS_ac

power_integral(HDH_DC['HDH (C)'],HDH_DC['OU (kWh)'],poly_ac)
print(Rsq_dc)
print(Rsq_ac)

# plot the outdoor unit power consumption over the heating degree hours
plt.figure(figsize=(8, 6), dpi=400)

dc_x = np.linspace(HDH_DC['HDH (C)'].min(), HDH_DC['HDH (C)'].max(), 100)
ac_x = np.linspace(HDH_AC['HDH (C)'].min(), HDH_AC['HDH (C)'].max(), 100)
plt.scatter(HDH_DC['HDH (C)'], HDH_DC['OU (kWh)'],edgecolors=edgecolor, marker=scatter_style,
           color=DC_color, alpha=alpha_scatter-0.1, s=s, label=f'DCHP (R² = {Rsq_dc[0]:.3f})')
#plt.plot(dc_x, dc_slope * dc_x + dc_intercept, line_style, color=DC_color, alpha=alpha_line, linewidth=width)
plt.plot(dc_x, poly_dc(dc_x), line_style, color=DC_color, alpha=alpha_line, linewidth=width)
plt.fill_between(dc_x, poly_dc(dc_x) - CI_dc, poly_dc(dc_x) + CI_dc, color=DC_color, alpha=0.4)
plt.scatter(HDH_AC['HDH (C)'], HDH_AC['OU (kWh)'],edgecolors=edgecolor,marker=scatter_style,
           color=AC_color, alpha=alpha_scatter-0.1, s=s, label=f'ACHP (R² = {Rsq_ac[0]:.3f})')
#plt.plot(ac_x, ac_slope * ac_x + ac_intercept, line_style, color=AC_color, alpha=alpha_line, linewidth=width)
plt.plot(ac_x, poly_ac(ac_x), line_style, color=AC_color, alpha=alpha_line, linewidth=width)
plt.fill_between(ac_x, poly_ac(ac_x) - CI_ac, poly_ac(ac_x) + CI_ac, color=AC_color, alpha=0.4)
plt.xlabel(r'Hourly average indoor-outdoor temperature difference [°C]', fontsize=label)
plt.ylabel('Hourly average outdoor unit power [kW]', fontsize=label)
plt.grid(True, linestyle='--', alpha=0.3)
for spine in plt.gca().spines.values():
    spine.set_linewidth(0.8)
plt.tick_params(direction='in', length=6, width=0.8, labelsize=16)
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(5))
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(1))
plt.legend(frameon=True, fancybox=True, edgecolor='black', 
          fontsize=18, loc='upper left')
plt.tight_layout()
plt.savefig("./figs/OU-hdh.png", dpi=400, bbox_inches='tight')

print("\n===============================Two Sample T-Test Results - HDH===============================\n")

print("Variation of HDH_DC:          ", f"{variation(HDH_DC['OU (kWh)']):.2f}")
print("Mean of HDH_DC:               ", f"{np.mean(HDH_DC['OU (kWh)']):.2f}")
print("Standard Deviation of HDH_DC: ", f"{np.std(HDH_DC['OU (kWh)']):.2f}")

print("\nVariation of HDH_AC:          ", f"{variation(HDH_AC['OU (kWh)']):.2f}")
print("Mean of HDH_AC:               ", f"{np.mean(HDH_AC['OU (kWh)']):.2f}")
print("Standard Deviation of HDH_AC: ", f"{np.std(HDH_AC['OU (kWh)']):.2f}")

t_test = stats.ttest_ind(HDH_DC['OU (kWh)'], HDH_AC['OU (kWh)'], equal_var=True)

print("\nT-Test Statistic:   ", f"{t_test[0]:.2f}")
print("P-Value:            ", t_test[1])

print("\n===============================Two Sample T-Test Results - HDH Model===============================\n")

print("Variation of HDH_DC:          ", f"{variation(poly_dc(dc_x)):.2f}")
print("Mean of HDH_DC:               ", f"{np.mean(poly_dc(dc_x)):.2f}")
print("Standard Deviation of HDH_DC: ", f"{np.std(poly_dc(dc_x)):.2f}")

print("\nVariation of HDH_AC:          ", f"{variation(poly_ac(ac_x)):.2f}")
print("Mean of HDH_AC:               ", f"{np.mean(poly_ac(ac_x)):.2f}")
print("Standard Deviation of HDH_AC: ", f"{np.std(poly_ac(ac_x)):.2f}")

t_test = stats.ttest_ind(poly_dc(dc_x), poly_ac(ac_x), equal_var=True)

print("\nT-Test Statistic:   ", f"{t_test[0]:.2f}")
print("P-Value:            ", t_test[1])

## Figure - Daily average kWh and HDD for OU
plt.figure(figsize=(8, 6), dpi=400)
HDH_DC = HDH_DC.groupby(by=pd.Grouper(freq='D')).mean()
HDH_AC = HDH_AC.groupby(by=pd.Grouper(freq='D')).mean()
HDH_AC = HDH_AC.dropna()

std_dc = stats.tstd(HDH_DC['OU (kWh)'])
std_ac = stats.tstd(HDH_AC['OU (kWh)'])

dc_rows, dc_columns = HDH_DC.shape
ac_rows, ac_columns = HDH_AC.shape

CI_dc = z_star * (std_dc/np.sqrt(dc_rows))
CI_ac = z_star * (std_ac/np.sqrt(ac_rows))
# Define objective function to minimize total squared error
def objective(params):
    shared_intercept, dc_slope, ac_slope = params
    
    # Calculate y-intercepts from shared x-intercept
    dc_y_intercept = -dc_slope * shared_intercept
    ac_y_intercept = -ac_slope * shared_intercept
    
    # Calculate predictions
    dc_pred = dc_slope * HDH_DC['HDH (C)'] + dc_y_intercept
    ac_pred = ac_slope * HDH_AC['HDH (C)'] + ac_y_intercept
    
    # Calculate sum of squared residuals for both datasets
    dc_sse = np.sum((HDH_DC['OU (kWh)'] - dc_pred)**2)
    ac_sse = np.sum((HDH_AC['OU (kWh)'] - ac_pred)**2)
    
    return dc_sse + ac_sse

# Initial guess: [shared_x_intercept, dc_slope, ac_slope]
initial_guess = [4, 0.1, 0.1]

# Optimize
result = minimize(objective, initial_guess, method='Nelder-Mead')

# Extract optimized parameters
shared_intercept, dc_slope, ac_slope = result.x

# Calculate y-intercepts
dc_intercept = -dc_slope * shared_intercept
ac_intercept = -ac_slope * shared_intercept

# Generate fit lines
dc_x = np.linspace(HDH_DC['HDH (C)'].min(), HDH_DC['HDH (C)'].max(), 100)
ac_x = np.linspace(HDH_AC['HDH (C)'].min(), HDH_AC['HDH (C)'].max(), 100)
dc_y = dc_slope * dc_x + dc_intercept
ac_y = ac_slope * ac_x + ac_intercept

# Calculate R² values
dc_ss_res = np.sum((HDH_DC['OU (kWh)'] - (dc_slope * HDH_DC['HDH (C)'] + dc_intercept))**2)
dc_ss_tot = np.sum((HDH_DC['OU (kWh)'] - np.mean(HDH_DC['OU (kWh)']))**2)
dc_r_squared = 1 - (dc_ss_res / dc_ss_tot)

ac_ss_res = np.sum((HDH_AC['OU (kWh)'] - (ac_slope * HDH_AC['HDH (C)'] + ac_intercept))**2)
ac_ss_tot = np.sum((HDH_AC['OU (kWh)'] - np.mean(HDH_AC['OU (kWh)']))**2)
ac_r_squared = 1 - (ac_ss_res / ac_ss_tot)

# Calculate fitted lines
dc_y = dc_slope * dc_x + dc_intercept
ac_y = ac_slope * ac_x + ac_intercept

# Calculate R² values
#dc_r_squared = dc_r_value**2
#ac_r_squared = ac_r_value**2

# Figure - OU HDH 
plt.scatter(HDH_DC['HDH (C)'], HDH_DC['OU (kWh)'], edgecolors=edgecolor, marker=scatter_style,
           color=DC_color, alpha=alpha_scatter, s=s, label=f'DCHP (R² = {dc_r_squared:.3f})')
plt.plot(dc_x, dc_y, line_style, color=DC_color, alpha=alpha_line, linewidth=width)
plt.fill_between(dc_x, dc_y - CI_dc, dc_y + CI_dc, color=DC_color, alpha=0.25)

plt.scatter(HDH_AC['HDH (C)'], HDH_AC['OU (kWh)'], edgecolors=edgecolor, marker=scatter_style,
           color=AC_color, alpha=alpha_scatter, s=s, label=f'ACHP (R² = {ac_r_squared:.3f})')
plt.plot(ac_x, ac_y, line_style, color=AC_color, alpha=alpha_line, linewidth=width)
plt.fill_between(ac_x, ac_y - CI_ac, ac_y + CI_ac, color=AC_color, alpha=0.25)

plt.xlabel(r'Daily average indoor-outdoor temperature difference [°C]', fontsize=label)
plt.ylabel(r'Daily average outdoor unit electric power [kW]', fontsize=label)
plt.grid(True, linestyle='--', alpha=0.3)
for spine in plt.gca().spines.values():
    spine.set_linewidth(0.8)
plt.tick_params(direction='in', length=6, width=0.8, labelsize=16)
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(5))
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(1))
plt.legend(frameon=True, fancybox=True, edgecolor='black', 
          fontsize=18, loc='upper left')
plt.tight_layout()
plt.savefig("./figs/OU-hdd.png", dpi=400, bbox_inches='tight')
#plt.show()

x = np.linspace(10,32,100)
y1 = poly_dc(x)
y2 = poly_ac(x)
diff = 100 * (y1-y2)/y2

y1_lin = dc_slope * x + dc_intercept
y2_lin = ac_slope * x + ac_intercept
diff_lin = 100 * (y1_lin - y2_lin)/y2_lin

X = sm.add_constant(HDH_DC['HDH (C)'])
model = sm.OLS(HDH_DC['OU (kWh)'], X).fit()

print("\n===============================Two Sample T-Test Results - HDD===============================\n")

print("Variation of HDD_DC:          ", f"{variation(HDH_DC['OU (kWh)']):.2f}")
print("Mean of HDD_DC:               ", f"{np.mean(HDH_DC['OU (kWh)']):.2f}")
print("Standard Deviation of HDD_DC: ", f"{np.std(HDH_DC['OU (kWh)']):.2f}")

print("\nVariation of HDD_AC:          ", f"{variation(HDH_AC['OU (kWh)']):.2f}")
print("Mean of HDD_AC:               ", f"{np.mean(HDH_AC['OU (kWh)']):.2f}")
print("Standard Deviation of HDD_AC: ", f"{np.std(HDH_AC['OU (kWh)']):.2f}")

t_test = stats.ttest_ind(HDH_DC['OU (kWh)'], HDH_AC['OU (kWh)'], equal_var=True)

print("\nT-Test Statistic:   ", f"{t_test[0]:.2f}")
print("P-Value:            ", f"{t_test[1]:.4f}")


X_ac = (HDH_AC['OU (kWh)']) / (HDH_AC['HDH (C)'] - 4)
X_dc = (HDH_DC['OU (kWh)']) / (HDH_DC['HDH (C)'] - 4)

print("\n===============================Two Sample T-Test Results - HDD===============================\n")

print("Variation of HDD_DC:          ", f"{variation(X_dc):.2f}")
print("Mean of HDD_DC:               ", f"{np.mean(X_dc):.2f}")
print("Standard Deviation of HDD_DC: ", f"{np.std(X_dc):.2f}")

print("\nVariation of HDD_AC:          ", f"{variation(X_ac):.2f}")
print("Mean of HDD_AC:               ", f"{np.mean(X_ac):.2f}")
print("Standard Deviation of HDD_AC: ", f"{np.std(X_ac):.2f}")

t_test = stats.ttest_ind(X_dc, X_ac, equal_var=False)

print("\nT-Test Statistic:   ", f"{t_test[0]:.2f}")
print("P-Value:            ", f"{t_test[1]:.4f}")

print("\n===============================Shared X-Intercept Model Results===============================\n")

# Load weather data for DC period (2024-2025)
weather_dc = pd.read_csv("./data/2425/weather_2425.csv")
weather_dc['Time'] = pd.to_datetime(weather_dc['Time'])
Tset_daily_dc = weather_dc.groupby(weather_dc['Time'].dt.date)['temperature (degC)'].mean()

# Load weather data for AC period (2023-2024)
weather_ac = pd.read_csv("./data/2324/weather_2324.csv")
weather_ac['Time'] = pd.to_datetime(weather_ac['Time'])
Tset_daily_ac = weather_ac.groupby(weather_ac['Time'].dt.date)['temperature (degC)'].mean()

# Calculate temperature differences
Tbal = 20.5

dc_temp_diff = Tbal - Tset_daily_dc
ac_temp_diff = Tbal - Tset_daily_ac

# Remove NaN values
dc_temp_diff = dc_temp_diff.dropna()
ac_temp_diff = ac_temp_diff.dropna()

# Calculate normalized statistics
dc_hdh_normalized = HDH_DC['HDH (C)'].values / dc_temp_diff.reindex(HDH_DC.index).fillna(method='ffill').fillna(method='bfill').values
ac_hdh_normalized = HDH_AC['HDH (C)'].values / ac_temp_diff.reindex(HDH_AC.index).fillna(method='ffill').fillna(method='bfill').values

# Remove inf and NaN from normalized values
dc_hdh_normalized = dc_hdh_normalized[np.isfinite(dc_hdh_normalized)]
ac_hdh_normalized = ac_hdh_normalized[np.isfinite(ac_hdh_normalized)]

print("Optimized shared x-intercept: ", f"{shared_intercept:.3f} °Ch")

print("\n--- DC Heat Pump ---")
print("Slope:                        ", f"{dc_slope:.4f} kW/°Ch")
print("Y-intercept:                  ", f"{dc_intercept:.4f} kW")
print("R²:                           ", f"{dc_r_squared:.3f}")
print("Mean HDH:                     ", f"{np.mean(HDH_DC['HDH (C)']):.2f} °Ch")
print("Std Dev HDH:                  ", f"{np.std(HDH_DC['HDH (C)']):.2f} °Ch")
print("Mean temp difference (ΔT):   ", f"{np.mean(dc_temp_diff):.2f} °C")
print("Std Dev temp difference:      ", f"{np.std(dc_temp_diff):.2f} °C")
print("Mean normalized HDH:          ", f"{np.mean(dc_hdh_normalized):.4f} kW/°C")
print("Std Dev normalized HDH:       ", f"{np.std(dc_hdh_normalized):.4f} kW/°C")

print("\n--- AC Heat Pump ---")
print("Slope:                        ", f"{ac_slope:.4f} kW/°Ch")
print("Y-intercept:                  ", f"{ac_intercept:.4f} kW")
print("R²:                           ", f"{ac_r_squared:.3f}")
print("Mean HDH:                     ", f"{np.mean(HDH_AC['HDH (C)']):.2f} °Ch")
print("Std Dev HDH:                  ", f"{np.std(HDH_AC['HDH (C)']):.2f} °Ch")
print("Mean temp difference (ΔT):   ", f"{np.mean(ac_temp_diff):.2f} °C")
print("Std Dev temp difference:      ", f"{np.std(ac_temp_diff):.2f} °C")
print("Mean normalized HDH:          ", f"{np.mean(ac_hdh_normalized):.4f} kW/°C")
print("Std Dev normalized HDH:       ", f"{np.std(ac_hdh_normalized):.4f} kW/°C")

print("\n--- Comparison of Slopes ---")
slope_diff = dc_slope - ac_slope
slope_diff_pct = ((dc_slope - ac_slope) / ac_slope) * 100

print("Slope difference (DC - AC):   ", f"{slope_diff:.4f} kW/°Ch")
print("Relative difference:          ", f"{slope_diff_pct:.2f}%")

normalized_diff = np.mean(dc_hdh_normalized) - np.mean(ac_hdh_normalized)
normalized_diff_pct = (normalized_diff / np.mean(ac_hdh_normalized)) * 100
print("\nNormalized HDH difference:    ", f"{normalized_diff:.4f} kW/°C")
print("Normalized relative diff:     ", f"{normalized_diff_pct:.2f}%")

print("\nOptimization success:         ", result.success)
print("Final objective value (SSE):  ", f"{result.fun:.2f}")

