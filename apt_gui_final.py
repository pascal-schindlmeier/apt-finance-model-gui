import tkinter as tk
from tkinter import messagebox, filedialog
from datetime import datetime
from pathlib import Path
import os
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import statsmodels.api as sm
import matplotlib.pyplot as plt
from fpdf import FPDF
import requests
import json
import logging
from dotenv import load_dotenv

# Load API keys from .env
load_dotenv()
DEFAULT_ALPHA_KEY = os.environ.get("ALPHAVANTAGE_API_KEY", "")
DEFAULT_FRED_KEY = os.environ.get("FRED_API_KEY", "")

# Define base and output directories
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Logging
LOG_FILE = OUTPUT_DIR / "apt_log.txt"
logging.basicConfig(
    filename=LOG_FILE,
    filemode="w",
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logging.getLogger().addHandler(logging.StreamHandler())

def normalize_monthly_index(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index).to_period('M').to_timestamp('M')
    return df

def align_data(returns_dict):
    common_index = set.intersection(*(set(df.index) for df in returns_dict.values()))
    aligned = {key: df[df.index.isin(common_index)] for key, df in returns_dict.items()}
    return aligned

class APTApp:
    def __init__(self, root):
        self.root = root
        self.root.title("APT Model Tool")
        self.setup_gui()

    def setup_gui(self):
        tk.Label(self.root, text="Asset Symbol:").grid(row=0, column=0)
        self.asset_entry = tk.Entry(self.root)
        self.asset_entry.insert(0, "AAPL")
        self.asset_entry.grid(row=0, column=1)

        tk.Label(self.root, text="Start Date (YYYY-MM-DD):").grid(row=1, column=0)
        self.start_entry = tk.Entry(self.root)
        self.start_entry.insert(0, "2020-01-01")
        self.start_entry.grid(row=1, column=1)

        tk.Label(self.root, text="End Date (YYYY-MM-DD):").grid(row=2, column=0)
        self.end_entry = tk.Entry(self.root)
        self.end_entry.insert(0, datetime.today().strftime('%Y-%m-%d'))
        self.end_entry.grid(row=2, column=1)

        tk.Label(self.root, text="Alpha Vantage API Key:").grid(row=3, column=0)
        self.api_entry = tk.Entry(self.root, width=40)
        self.api_entry.insert(0, DEFAULT_ALPHA_KEY)
        self.api_entry.grid(row=3, column=1)

        tk.Label(self.root, text="FRED API Key:").grid(row=4, column=0)
        self.fred_api_entry = tk.Entry(self.root, width=40)
        self.fred_api_entry.insert(0, DEFAULT_FRED_KEY)
        self.fred_api_entry.grid(row=4, column=1)

        tk.Label(self.root, text="Factors (1 per line):").grid(row=5, column=0, columnspan=2)
        self.factor_text = tk.Text(self.root, height=6, width=40)
        self.factor_text.insert("1.0", "^GSPC\nDCOILWTICO\nFEDFUNDS\nAV:TSLA")
        self.factor_text.grid(row=6, column=0, columnspan=2)

        tk.Label(self.root, text="Custom Horizon (months):").grid(row=7, column=0)
        self.horizon_entry = tk.Entry(self.root)
        self.horizon_entry.insert(0, "6")
        self.horizon_entry.grid(row=7, column=1)

        tk.Button(self.root, text="Run APT Model", command=self.run_apt).grid(row=8, column=0)
        tk.Button(self.root, text="Export Report", command=self.export_report).grid(row=8, column=1)
        tk.Button(self.root, text="View Logs", command=lambda: os.system(f'notepad {LOG_FILE}')).grid(row=9, column=0, columnspan=2)

    def fetch_yfinance_data(self, symbol, start, end):
        df = yf.download(symbol, start=start, end=end, interval="1mo", progress=False)
        if df.empty:
            raise ValueError(f"No data for symbol: {symbol}")
        price = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']
        df = pd.DataFrame(price)
        df.columns = ['price']
        df = normalize_monthly_index(df)
        df['return'] = np.log(df['price'] / df['price'].shift(1))
        return df[['return']].dropna()

    def fetch_fred_data(self, fred, symbol, start, end):
        series = fred.get_series(symbol, observation_start=start, observation_end=end)
        df = series.to_frame(name='price')
        df = normalize_monthly_index(df)
        df['return'] = np.log(df['price'] / df['price'].shift(1))
        return df[['return']].dropna()

    def fetch_alphavantage_data(self, symbol, api_key, start):
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY_ADJUSTED&symbol={symbol}&apikey={api_key}&datatype=json"
        response = requests.get(url)
        data = response.json()
        if "Monthly Adjusted Time Series" not in data:
            raise ValueError(f"Alpha Vantage error: {data.get('Note') or data.get('Error Message') or 'No data found'}")
        records = data["Monthly Adjusted Time Series"]
        df = pd.DataFrame.from_dict(records, orient='index')
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        df = df[df.index >= pd.to_datetime(start)]
        df['price'] = df['5. adjusted close'].astype(float)
        df = normalize_monthly_index(df)
        df['return'] = np.log(df['price'] / df['price'].shift(1))
        return df[['return']].dropna()

    def run_apt(self):
        try:
            asset = self.asset_entry.get().strip()
            start = self.start_entry.get()
            end = self.end_entry.get()
            api_key = self.api_entry.get().strip()
            fred_key = self.fred_api_entry.get().strip()
            horizon_months = int(self.horizon_entry.get().strip())
            factors = [f.strip() for f in self.factor_text.get("1.0", tk.END).splitlines() if f.strip()]

            if not asset or not api_key:
                raise ValueError("Asset symbol and API key are required.")
            if not factors:
                raise ValueError("Please enter at least one factor.")

            fred = Fred(api_key=fred_key) if any(not s.startswith(("^", "AV:")) for s in factors) else None
            returns = {'asset': self.fetch_yfinance_data(asset, start, end)}

            for factor in factors:
                if factor.startswith("^"):
                    returns[factor] = self.fetch_yfinance_data(factor, start, end)
                elif factor.startswith("AV:"):
                    returns[factor] = self.fetch_alphavantage_data(factor.split("AV:")[-1], api_key, start)
                else:
                    returns[factor] = self.fetch_fred_data(fred, factor, start, end)

            returns = align_data(returns)
            combined = pd.concat([df.rename(columns={'return': key}) for key, df in returns.items()], axis=1)

            if len(combined) < 10:
                raise ValueError("Not enough data for regression.")

            X = sm.add_constant(combined[factors])
            y = combined['asset']
            self.model = sm.OLS(y, X).fit()
            self.combined = combined
            self.factors = factors
            self.expected = self.model.params[1:].dot(combined[factors].mean()) + self.model.params[0]
            self.annualized = (1 + self.expected)**12 - 1
            self.custom_return = (1 + self.expected)**horizon_months - 1
            self.horizon_months = horizon_months
            self.factor_stats = {
                f: {
                    "beta": self.model.params[f],
                    "t": self.model.tvalues[f],
                    "p": self.model.pvalues[f],
                    "mean": combined[f].mean()
                }
                for f in factors
            }
            self.show_results()
        except Exception as e:
            logging.exception("APT model failed.")
            messagebox.showerror("Error", str(e))

    def show_results(self):
        # Plot each factor's regression chart
        for factor in self.factors:
            fig = plt.figure(figsize=(10, 6))
            sm.graphics.plot_regress_exog(self.model, factor, fig=fig)
            plt.tight_layout()
            fig_path = OUTPUT_DIR / f"reg_plot_{factor.replace(':', '')}.png"
            plt.savefig(fig_path)
            plt.close()

        factor_summary = "\n--- Factor Contributions ---\n"
        for f, stats in self.factor_stats.items():
            factor_summary += (
                f"{f:<12} | β = {stats['beta']:.3f}  "
                f"| t = {stats['t']:.2f}  "
                f"| p = {stats['p']:.4f}  "
                f"| Mean: {stats['mean']:.4f}\n"
            )

        legend = (
            "\n--- Legend ---\n"
            "Monthly Return: Return estimated for a typical month.\n"
            "Annualized Return: Compounded over 12 months.\n"
            "Custom Horizon: Compounded over your chosen time frame.\n"
            "R2: Higher is better. How well do the chosen factors explain the movement of the asset? Ranges from 0 - 1 (or 1% to 100%).\n"
            "Adjusted R2: Like R2, but penalizes for unnecessary variables.\n"
            "t: Strength of effect. |t| > 2 is usually strong.\n"
            "p: Significance. p < 0.05 is usually considered meaningful.\n"
        )

        summary = (
            f"Expected Monthly Return: {self.expected:.4f}\n"
            f"Expected Annualized Return: {self.annualized:.2%}\n"
            f"Return over {self.horizon_months} months: {self.custom_return:.2%}\n\n"
            f"R²: {self.model.rsquared:.4f}\n"
            f"Adjusted R²: {self.model.rsquared_adj:.4f}\n"
            f"F-statistic: {self.model.fvalue:.2f} (p={self.model.f_pvalue:.4f})\n"
            f"{factor_summary}{legend}"
        )

        messagebox.showinfo("APT Results", summary)

    def export_report(self):
        if not hasattr(self, 'model'):
            messagebox.showerror("No results", "Run the model first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
        if not path:
            return

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=10)

        pdf.multi_cell(0, 5, self.model.summary().as_text())
        pdf.ln()

        pdf.set_font("Arial", 'B', size=12)
        pdf.cell(0, 10, "Return Summary", ln=True)
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 8, f"Expected Monthly Return: {self.expected:.4f}", ln=True)
        pdf.cell(0, 8, f"Expected Annualized Return: {self.annualized:.2%}", ln=True)
        pdf.cell(0, 8, f"Expected {self.horizon_months}-Month Return: {self.custom_return:.2%}", ln=True)

        pdf.ln()
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Factor Contributions", ln=True)
        pdf.set_font("Arial", size=10)
        for f, stats in self.factor_stats.items():
            pdf.cell(0, 8,
                     f"{f:<12} | β = {stats['beta']:.3f}  "
                     f"| t = {stats['t']:.2f}  "
                     f"| p = {stats['p']:.4f}  "
                     f"| Mean: {stats['mean']:.4f}", ln=True)

        for factor in self.factors:
            fig_path = OUTPUT_DIR / f"reg_plot_{factor.replace(':', '')}.png"
            if fig_path.exists():
                pdf.add_page()
                pdf.image(str(fig_path), x=10, y=20, w=190)

        pdf.output(path)
        messagebox.showinfo("Exported", f"Report saved to: {path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = APTApp(root)
    root.mainloop()
