import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk

# --- Problem Fix Code Block [Start] ---

# 1. Solve font display issues: Set Matplotlib to use fonts that support various characters
#    The program will try to use standard fonts. If you need specific language support,
#    you can install appropriate fonts like 'Arial Unicode MS' or system fonts.
try:
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
except:
    try:
        plt.rcParams['font.sans-serif'] = ['Helvetica']
    except:
        print("Warning: Standard fonts not found. Using default system font.")
        print("Consider installing standard fonts if you encounter display issues.")
        plt.rcParams['font.sans-serif'] = ['sans-serif']

# 2. Solve negative sign display issue: Ensure proper minus sign display
plt.rcParams['axes.unicode_minus'] = False


# --- Problem Fix Code Block [End] ---


class StockVisualizer:
    def __init__(self, master=None):
        """Initialize Visualizer, create figure and axes"""
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        # If in Tkinter main window, embed the chart
        if master:
            self.canvas = FigureCanvasTkAgg(self.fig, master=master)
            self.canvas_widget = self.canvas.get_tk_widget()
            self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            toolbar = NavigationToolbar2Tk(self.canvas, master)
            toolbar.update()
            self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def plot_price_history(self, df_dict, title="Stock Price History", indicator_col=None):
        """Plot stock price history and optional indicators"""
        self.ax.clear()
        for ticker, df in df_dict.items():
            if not df.empty:
                self.ax.plot(df.index, df['Close'], label=f'{ticker} Close Price')
                if indicator_col and indicator_col in df.columns:
                    self.ax.plot(df.index, df[indicator_col], label=f'{ticker} {indicator_col}', linestyle='--')
        self._format_plot(title, "Date", "Price")

    def plot_multi_stock_comparison(self, df_dict, metric="Close", title="Stock Comparison"):
        """Compare the same metric across multiple stocks"""
        self.ax.clear()
        for ticker, df in df_dict.items():
            if not df.empty and metric in df.columns:
                self.ax.plot(df.index, df[metric], label=f'{ticker} {metric}')
        self._format_plot(title, "Date", metric)

    def _format_plot(self, title, xlabel, ylabel):
        """
        Uniformly format the chart.
        This is the core location for font size adjustments.
        """
        # --- Font Size Adjustments [Start] ---

        # Set title font size to 20
        self.ax.set_title(title, fontsize=20, weight='bold')

        # Set X-axis and Y-axis label font size to 16
        self.ax.set_xlabel(xlabel, fontsize=16)
        self.ax.set_ylabel(ylabel, fontsize=16)

        # Set legend font size to 14
        self.ax.legend(fontsize=14)

        # Set axis tick font size to 12
        self.ax.tick_params(axis='both', which='major', labelsize=12)

        # --- Font Size Adjustments [End] ---

        # Keep grid lines and date formatting
        self.ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        self.fig.tight_layout()  # Auto-adjust layout to prevent label overlap
        self.fig.autofmt_xdate()

        # Refresh canvas
        if hasattr(self, 'canvas'):
            self.canvas.draw()
        else:
            plt.show()