import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

eta_inv = 0.95
eta_rect_HP = 0.955
eta_inv_HP = 0.97
eta_mppt = 0.98
eta_dc_dc = 0.95
eta_nom = 0.95

path = './data/'

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

font = "Times New Roman"         # choice of font
font_fam = "serif" # font family

plt.rcParams["font.family"] = font_fam
plt.rcParams["font.sans-serif"] = [font]

def hdh(year,HP,Tbal,solar):
    """Creates a data frame for the HDH analysis

    Args:
        year (str): selected year to analyze
        HP (list): list of the heat pump data files
        Tbal (int): balance point
        solar (str): file name of the solar data

    Returns:
        df: pandas data frame of the hourly heat pump operation for analysis 
    """

    # read in the air handling unit data
    AHU_main = pd.read_csv(path+year+HP[0])
    AHU_Aux = pd.read_csv(path+year+HP[1])
    AHU = pd.DataFrame()
    AHU['Time'] = pd.to_datetime(AHU_main['Time']) # convert to datetime
    AHU['Value'] = (AHU_Aux['Value'] + AHU_main['Value']) / 1000 # convert to kW
    AHU_hourly = AHU.groupby(AHU['Time'].dt.floor('h'))['Value'].sum().reset_index() # resets index to hourly
    AHU_hourly['Value'] = AHU_hourly['Value'] / 12 # converts the value to kWh
   
    # read in the outdoor unit data
    OU = pd.read_csv(path+year+HP[2])
    OU['Time'] = pd.to_datetime(OU['Time']) # convert to datetime
    OU['Value'] = OU['Value'] / 1000 # convert to kW
    OU_hourly = OU.groupby(OU['Time'].dt.floor('h'))['Value'].sum().reset_index() # resets index to hourly
    OU_hourly['Value'] = OU_hourly['Value'] / 12 # converts the value to kWh since 5 minute data
    
    # read in the full house data
    left_panel = pd.read_csv(path+year+HP[5])
    right_panel = pd.read_csv(path+year+HP[6])
    left_panel['Value'] = left_panel['Value'] / 1000 # convert to kW
    right_panel['Value'] = right_panel['Value'] / 1000 # convert to kW
    left_panel['Time'] = pd.to_datetime(left_panel['Time']) # convert to datetime
    right_panel['Time'] = pd.to_datetime(right_panel['Time']) # convert to datetime
    left_panel_hourly = left_panel.groupby(left_panel['Time'].dt.floor('h'))['Value'].sum().reset_index() # resets index to hourly
    right_panel_hourly = right_panel.groupby(right_panel['Time'].dt.floor('h'))['Value'].sum().reset_index() # resets index to hourly
    left_panel_hourly['Value'] = left_panel_hourly['Value'] / 12 # converts the value to kWh
    right_panel_hourly['Value'] = right_panel_hourly['Value'] / 12 # converts the value to kWh

    # read in the set point data
    Tset_cool = pd.read_csv(path+year+HP[3])
    Tset_heat = pd.read_csv(path+year+HP[4])
    Tset =  pd.merge(Tset_cool, Tset_heat, on='Time', how='outer')
    Tset['Value'] = Tset['Value_x'].fillna(Tset['Value_y'])
    Tset['Time'] = pd.to_datetime(Tset['Time']) # convert to datetime
    Tset_hourly = Tset.groupby(Tset['Time'].dt.floor('h'))['Value'].mean().reset_index() # resets index to hourly
    
    # read in the weather data
    weather = pd.read_csv(path+year+'/weather_'+year+'.csv')
    Tamb = weather['temperature (degC)']

    # read in the solar data
    PV = pd.read_csv(path+year+solar)
    PV_power = PV['p_mp']

    # create the HDH data frame
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
    HDH['PV Power'] = PV_power / 1000
    HDH['Home load (kWh)'] = left_panel_hourly['Value'] + right_panel_hourly['Value'] - HDH['Etot (kWh)']
    HDH = HDH.fillna(5)
    HDH = HDH.set_index('Time')
        
    return HDH

def bar_plot(dc_use, ac_use, dc_ideal_use):
    """bar plot of the net exports from the simulation

    Args:
        dc_use (int): net exports from the DCHP simulation
        ac_use (int): net exports from the ACHP simulation
        dc_ideal_use (int): net exports from the ideal DCHP simulation
    """
    
    # Configuration names and efficiency values
    names = ['AC-coupled HP','DC-coupled HP', 'Ideal DC HP']
    exports = [ac_use, dc_use, dc_ideal_use]
    fig, ax = plt.subplots(figsize=(8, 6), dpi=400)
    colors = [ '#377EB8','#E41A1C', '#9A7FBF',]
    bar_width = 0.6
    bars = ax.bar(names, exports, width=bar_width, color=colors, 
                edgecolor='black', linewidth=1, alpha=0.9, 
                capsize=7)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_xlabel("Heat Pump Configuration", fontsize=12)
    ax.set_ylabel("Net annual cost of energy [USD]", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.set_ylim(0, max(exports) * 1.15)
    ax.set_facecolor('#f8f8f8')
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    plt.tight_layout()

    #plt.show()
    plt.savefig("./figs/net-home-use_ideal.png", dpi=400, bbox_inches='tight')
    return

def monthly_bar_plot(HDH, E_house_dc, E_house_ac, E_house_ideal):
    """monthly bar plot of the net exported power from the simulations

    Args:
        HDH (df): pandas data frame of the hourly heat pump operation for analysis 
        E_house_dc (list): list of the net exported power from the DCHP simulation
        E_house_ac (list): list of the net exported power from the ACHP simulation
        E_house_ideal (list): list of the net exported power from the ideal DCHP simulation
    """

    HDH['E_house_dc'] = E_house_dc[:len(HDH)]
    HDH['E_house_ac'] = E_house_ac[:len(HDH)]
    HDH['E_house_ideal'] = E_house_ideal[:len(HDH)]

    # Add month column to HDH dataframe
    HDH['Month'] = HDH.index.to_period('M')

    # Calculate monthly net exported power for each configuration
    dc_monthly = HDH['E_house_dc'].groupby(HDH.index.month).sum() * 0.14
    ac_monthly = HDH['E_house_ac'].groupby(HDH.index.month).sum() * 0.14
    ideal_monthly = HDH['E_house_ideal'].groupby(HDH.index.month).sum() * 0.14

    # Create the plot with modern styling
    fig, ax = plt.subplots(figsize=(14, 8), dpi=400)

    # Fix: Use the actual number of months in the data
    x = np.arange(len(dc_monthly))  # or len(ac_monthly) or len(ideal_monthly)
    width = 0.28

    # Modern color palette with gradients
    dc_color = '#E41A1C'  # Modern coral red
    ac_color = '#377EB8'  # Modern teal
    ideal_color = '#9A7FBF'  # Modern purple

    bars1 = ax.bar(x, dc_monthly, width, label='DC-coupled HP', 
                color=dc_color, edgecolor='white', linewidth=1.5, alpha=1,
                capstyle='round')
    bars2 = ax.bar(x - width, ac_monthly, width, label='AC-coupled HP', 
                color=ac_color, edgecolor='white', linewidth=1.5, alpha=1,
                capstyle='round')
    bars3 = ax.bar(x + width, ideal_monthly, width, label='Ideal DC HP', 
                color=ideal_color, edgecolor='white', linewidth=1.5, alpha=1,
                capstyle='round')

    # Enhanced value labels with better positioning
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            # Position labels above positive bars, below negative bars
            y_offset = 8 if height >= 0 else -8  # noqa: F841
            va = 'bottom' if height >= 0 else 'top'  # noqa: F841

    # Enhanced styling
    ax.set_ylabel("Monthly energy bill [USD]", fontsize=14, color='#2C3E50')

    # Fix: Create proper month labels from the data
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    display_labels = [month_names[month-1] for month in dc_monthly.index]

    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, fontsize=16, fontweight='bold')

    # Enhanced legend
    legend = ax.legend(loc='best', fontsize=14, framealpha=0.95, 
                    fancybox=True, shadow=True, borderpad=1)
    legend.get_frame().set_facecolor('#FFFFFF')
    legend.get_frame().set_edgecolor('#E0E0E0')

    # Modern grid
    ax.grid(axis='y', linestyle='-', alpha=0.3, color='#BDC3C7', linewidth=0.8)
    ax.set_axisbelow(True)

    # Clean background
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('#FFFFFF')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#BDC3C7')
    ax.spines['bottom'].set_color('#BDC3C7')
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    # Add zero line for reference
    ax.axhline(y=0, color='#34495E', linewidth=1.5, alpha=0.8)

    # Improve tick styling
    ax.tick_params(axis='both', which='major', labelsize=16, colors='#2C3E50')
    ax.tick_params(axis='y', which='major', length=6, width=1.2, color='#BDC3C7')
    ax.tick_params(axis='x', which='major', length=6, width=1.2, color='#BDC3C7')

    plt.tight_layout()
    #plt.show()
    plt.savefig("./figs/monthly_home_ideal.png", bbox_inches='tight')

def efficiency(x, a=20, b=17.2, c=1.5, max_val=98):
    """function to calculate the efficiency of a component using the percentage full load as an input

    Args:
        x (float): percentage of the full load on a component
        a (int, optional): Defaults to 12.
        b (int, optional): Defaults to 6.
        c (int, optional): Defaults to 3.
        max (int, optional): Defaults to 98.

    Returns:
        float: efficiency of the component
    """    
        # Calculate d so that f(0) = 0
    # Solve: a * (1 - exp(b*d)) + c*d^2 + max_val - 10.65888 = 0
    from scipy.optimize import fsolve
    
    def equation(d):
        return a * (1 - np.exp(b * d)) + c * d**2 + max_val - 19.8
    
    d = fsolve(equation, 0.08)[-1]
    
    if x < 0.0:
        eta = 0
    else:
        eta = (a * (1 - np.exp(-b * (x-d))) - c * (x-d)**2 + max_val - 19.8) / 100
        
    eta = max(eta, 0.9)
    
    return np.abs(eta)

def solar_use(weather,HDH,mode):
    """function to calculate the solar use of a nanogrid

    Args:
        weather (str): path to the weather data
        HDH (df): pandas data frame of the hourly heat pump operation for analysis 
        mode (str): type of nanogrid configuration

    Returns:
        list: list of the net exported power from the simulation
    """
    df = pd.read_csv(weather)
    dt = 1 # time step duration, h
    hours = int((df.index[-1] - df.index[0]))
    xMax = 20 # battery energy capacity, kWh
    xMin = 0
    x0 = 0.5 * xMax
    pCap = 12.5 # battery charging/discharging power capacity, kW
    
    tau = 1600 # self-dissipation time constant, h
    a = np.exp(-dt / tau)

    # Data storage
    x_all = np.zeros(hours+1)  # +1 to prevent index out of bounds
    x_all[0] = x0
    Pchem_all = np.zeros(hours+1)
    E_house = np.zeros(hours+1)
    data_length = min(hours-1, len(HDH['PV Power']))
    b = np.zeros(hours) # battery electric charging power, kW
    
    # Max power
    P_s_max = 15
    P_hp_max = 20
    
    P_hp = np.zeros(hours)

    # DC Nanogrid configuration
    for i in range(data_length):  # Stop at hours-1 to prevent index out of bounds    
                
        HP_load = HDH['Etot (kWh)'].iloc[i] / P_hp_max
        P_s_load = HDH['PV Power'].iloc[i] / P_s_max
        
        # Calculate the conversion efficiencies
        if mode == 'ac':
            eta = eta_nom * eta_inv
            eta_sol_nom = efficiency(P_s_load,max_val=eta_inv*100) #* efficiency(P_s_load, max_val=eta_mppt*100)
            eta_hp_nom = efficiency(HP_load, max_val=eta_rect_HP*100) * efficiency(HP_load, max_val=eta_inv_HP*100)
            eta_home = 1
        elif mode == 'dc':
            eta = eta_nom * eta_dc_dc
            eta_sol_nom = efficiency(P_s_load, max_val=eta_dc_dc*100) # * efficiency(P_s_load, max_val=eta_mppt*100)
            eta_hp_nom = efficiency(HP_load, max_val=eta_inv_HP*100)
            eta_home = eta_inv
        elif mode == 'dc-ideal':
            eta = eta_nom * eta_dc_dc
            eta_sol_nom = efficiency(P_s_load, max_val=eta_dc_dc*100) # * efficiency(P_s_load, max_val=eta_mppt*100)
            eta_hp_nom = efficiency(HP_load, max_val=100)
            eta_home = eta_inv
            
        # Calculate the battery charging/discharging power
        if eta_sol_nom * HDH['PV Power'].iloc[i] > HDH['Etot (kWh)'].iloc[i] / eta_hp_nom:  # Excess solar power
            Pchem_all[i] = eta*np.min([eta_sol_nom * HDH['PV Power'].iloc[i] - HDH['Etot (kWh)'].iloc[i] / eta_hp_nom, pCap, (xMax - a*x_all[i])/((1-a)*tau)])  # Limit charging to available power
        else:
            Pchem_all[i] = (1/eta)*np.max([eta_sol_nom * HDH['PV Power'].iloc[i] - HDH['Etot (kWh)'].iloc[i] / eta_hp_nom, -pCap, (xMin - a*x_all[i])/((1-a)*tau)])
        
        Pchem_load = np.abs(Pchem_all[i] / pCap)
        eta = efficiency(Pchem_load, max_val=eta_nom*100)
        
        x_all[i+1] = a*x_all[i] + (1-a)*tau*Pchem_all[i]
        b[i] = max([eta*Pchem_all[i], Pchem_all[i]/eta])
        E_house[i] = - (eta_sol_nom * HDH['PV Power'].iloc[i] - HDH['Etot (kWh)'].iloc[i] / eta_hp_nom - b[i] - HDH['Home load (kWh)'].iloc[i] / eta_home)
        
        P_hp[i] = HDH['Etot (kWh)'].iloc[i] / eta_hp_nom

    print("Max P_HP: ", np.max(P_hp))
    return E_house, Pchem_all, x_all, xMax, xMin, df

def payback(rev,DPP):
    """function to calculate the payback period of a nanogrid

    Args:
        rev (float): revenue from the nanogrid
        DPP (float): discounted present price of the nanogrid

    Returns:
        float: payback period of the nanogrid
    """    
    r = 0.07
    return (rev/r)*(1 - (1+r)**(-DPP))

Tbal = 24

year = '2024_full'
ACHP = ['/AHU_main_return_2024.csv','/AHU_Aux_return_2024.csv','/AC_unitout_return_2024.csv','/cooling_setpoint_return_2024.csv',
        '/heating_setpoint_return_2024.csv','/main_left_2024.csv','/main_right_2024.csv']
AC_solar = '/solar_output_2024_full.csv'

HDH_AC = hdh(year,ACHP,Tbal,AC_solar)

weather = './data/2024_full/weather_2024_full.csv'

E_house, Pchem_all, x_all, xMax, xMin, df = solar_use(weather, HDH_AC,'dc')
E_house_ac, Pchem_all_ac, x_all_ac, xMax, xMin, df = solar_use(weather, HDH_AC,'ac')
E_house_ideal, Pchem_all_ideal, x_all_ideal, xMax, xMin, df = solar_use(weather, HDH_AC,'dc-ideal')

# Calculate available power for the whole timeseries
available_power = eta_mppt * eta_inv_HP * HDH_AC['PV Power'] - HDH_AC['Etot (kWh)']
date_format = mdates.DateFormatter('%b %d')

dc_batt = x_all.sum()
ac_batt = x_all_ac.sum()
ideal_batt = x_all_ideal.sum()

dc_net_price = (np.sum(E_house))*0.14
ac_net_price = (np.sum(E_house_ac)) * 0.14
ideal_net_price = (np.sum(E_house_ideal)) * 0.14

bar_plot(dc_net_price,ac_net_price,ideal_net_price)
monthly_bar_plot(HDH_AC, E_house, E_house_ac, E_house_ideal)
