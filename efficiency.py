import numpy as np
import matplotlib.pyplot as plt

# Generate x values (0% to 100% load)
x = np.linspace(0, 1, 100)


# Model 4: Exponential saturation
def exp_saturation(x, a=20, b=17.2, c=1.5, max=98):
    # Calculate d so that f(0) = 0
    # Solve: a * (1 - exp(b*d)) + c*d^2 + max_val - 10.65888 = 0
    from scipy.optimize import fsolve
    
    def equation(d):
        return a * (1 - np.exp(b * d)) + c * d**2 + max - 19.8
    
    d = fsolve(equation, 0.08)[0]
    
    eta = a * (1 - np.exp(-b * (x - d))) - c * (x - d)**2 + max - 19.8
    
    #for i in range(len(eta)):
    #    if eta[i] < 0.9:
    #        eta[i] = 0.9

    return np.abs(eta)


# ==================== CONFIGURABLE SETTINGS ====================

# Figure settings
FIGURE_WIDTH = 8           # inches
FIGURE_HEIGHT = 6          # inches
DPI = 400                  # resolution

# Line settings
LINE_WIDTH = 2.0
LINE_ALPHA = 0.9

# Axis labels
XLABEL = 'Part load ratio [%]'
YLABEL = 'Efficiency [%]'
XLABEL_SIZE = 12
YLABEL_SIZE = 12

# Title (set to empty string if not needed)
TITLE = ''
TITLE_SIZE = 14

# Tick settings
TICK_LABELSIZE = 11
TICK_WIDTH = 1.0
TICK_LENGTH = 4

# Spine settings
SPINE_WIDTH = 0.8
SHOW_TOP_SPINE = True
SHOW_RIGHT_SPINE = True

# Grid settings
SHOW_GRID = True
GRID_AXIS = 'both'        # 'x', 'y', or 'both'
GRID_ALPHA = 0.3
GRID_LINESTYLE = '--'
GRID_LINEWIDTH = 0.5

# Font settings
FONT_FAMILY = 'Times New Roman'

# Set font globally
plt.rcParams['font.family'] = FONT_FAMILY

# Layout
TIGHT_LAYOUT = True

# Y-axis limits (set to None for automatic)
YLIM_BOTTOM = 90
YLIM_TOP = 100

# X-axis limits
XLIM_LEFT = 0
XLIM_RIGHT = 110

# Legend settings
LEGEND_FONTSIZE = 10
LEGEND_LOCATION = 'best'

# Calculate y values for each model
y_inv = exp_saturation(x,max=95)
y_rect = exp_saturation(x,max=94.5)
y_dcdc = exp_saturation(x,max=98)
y_hp_inv = exp_saturation(x,max=97)

a=20
b=17.2
c=1.5

# Create plot
fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), dpi=DPI)

# Plot lines
ax.plot(x * 100, y_inv, label='Inverter', 
        linewidth=LINE_WIDTH, alpha=LINE_ALPHA)
ax.plot(x * 100, y_rect, label='Rectifier', 
        linewidth=LINE_WIDTH, alpha=LINE_ALPHA)
ax.plot(x * 100, y_dcdc, label='DC-DC Converter', 
        linewidth=LINE_WIDTH, alpha=LINE_ALPHA)
ax.plot(x * 100, y_hp_inv, label='Heat Pump Inverter', 
        linewidth=LINE_WIDTH, alpha=LINE_ALPHA)

# Set labels
ax.set_xlabel(XLABEL, fontsize=XLABEL_SIZE)
ax.set_ylabel(YLABEL, fontsize=YLABEL_SIZE)
if TITLE:
    ax.set_title(TITLE, fontsize=TITLE_SIZE)

# Set limits
if YLIM_BOTTOM is not None or YLIM_TOP is not None:
    ax.set_ylim(bottom=YLIM_BOTTOM, top=YLIM_TOP)
ax.set_xlim(left=XLIM_LEFT, right=XLIM_RIGHT)

# Configure ticks
ax.tick_params(axis='both', labelsize=TICK_LABELSIZE, width=TICK_WIDTH, length=TICK_LENGTH)

# Configure spines
for spine in ax.spines.values():
    spine.set_linewidth(SPINE_WIDTH)
    
if not SHOW_TOP_SPINE:
    ax.spines['top'].set_visible(False)
if not SHOW_RIGHT_SPINE:
    ax.spines['right'].set_visible(False)

# Grid
if SHOW_GRID:
    ax.grid(True, axis=GRID_AXIS, alpha=GRID_ALPHA, 
            linestyle=GRID_LINESTYLE, linewidth=GRID_LINEWIDTH)

# Legend
ax.legend(loc=LEGEND_LOCATION, fontsize=LEGEND_FONTSIZE)

# Add equation as text annotation
equation_text = r'$\eta(x) = a(1 - e^{-b(x-d)}) - c(x-d)^2 + \mathrm{max} - 19.8$' + '\n' + \
                r'$a=' + f'{a}' + r', b=' + f'{b}' + r', c=' + f'{c}' + r', \mathrm{max}=input$'
ax.text(0.3, 0.2, equation_text, transform=ax.transAxes, 
        fontsize=10, verticalalignment='bottom', fontfamily='Times New Roman',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

# Layout
if TIGHT_LAYOUT:
    plt.tight_layout()

#
plt.savefig('efficiency.png', dpi=DPI, bbox_inches='tight')