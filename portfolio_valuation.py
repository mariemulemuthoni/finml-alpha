"""
    Portfolio Valuation via Linear Transformations
    ================================================
    This script demonstrates a core application of linear algebra in financial modeling:
        computing the daily USD equity of a multi-asset forex portfolio using matrix-vector
        multiplication.

    The key insight is that portfolio valuation is a linear transformation:
        - A positions vector p ∈ Rⁿ holds the quantity of each asset
        - A price matrix P ∈ Rᵐˣⁿ holds the price of each asset across m trading days
        - The dot product P · p yields a value vector v ∈ Rᵐ — the portfolio's total
            USD equity for each day

    The same matrix-vector structure underpins more advanced techniques in quantitative
        finance. In mean-variance portfolio optimization (Markowitz), the covariance matrix
        of asset returns is used to compute portfolio variance (pᵀΣp), and the expected
        return is the dot product of weights with return estimates (μᵀp). Understanding
        this valuation primitive is the foundation for those methods.
            
    Structure:
        Section I  — Static proof using hard-coded data (2 assets, 2 days)
        Section II — Live implementation using 30 days of historical forex data
                    from the Yahoo Finance API (3 assets: EUR/USD, GBP/USD, USD/JPY)
"""

import yfinance as yf
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# SECTION I: PROOF OF CONCEPT
# ------------------------------------------------------------
# Verify the core P · p transformation with traceable numbers before scaling.

print("=" * 60)
print("SECTION I: Static Linear Transformation (Proof of Concept)")
print("=" * 60)

# --- Position Vector (Lots): What You Own ---
# Each element represents how many lots (units) you hold of each currency pair.
# Index 0 → EUR/USD (2 lots), Index 1 → GBP/USD (1 lot)
positions_vector_static = np.array([2, 1])

# --- Price Matrix: What Those Assets Cost Over Time ---
# Row = A trading day; Column = An asset's closing price in USD.
# Shape: (days × assets) → here, (2 × 2)
# Row 0 → Day 1: EUR/USD = 1.08, GBP/USD = 1.26
# Row 1 → Day 2: EUR/USD = 1.09, GBP/USD = 1.25
price_matrix_static = np.array([
    [1.08, 1.26],  # Day 1: EUR/USD, GBP/USD
    [1.09, 1.25]   # Day 2
])

# --- Linear Transformation: Compute Daily Portfolio Value ---
# Matrix multiplication (dot product) maps the (2×2) price matrix and the (2,)
#   position vector to a (2,) output vector — one equity value per day.
portfolio_value_static = price_matrix_static.dot(positions_vector_static)

print(f"\nPositions Vector (lots per asset):\n  {positions_vector_static}")
print(f"\nPrice Matrix (USD, rows = days, columns = assets):\n{price_matrix_static}")
print(f"\nResulting Portfolio Value Vector (P · p):")
print(f"  Day 1: ${portfolio_value_static[0]:.2f}")
print(f"  Day 2: ${portfolio_value_static[1]:.2f}")


# ------------------------------------------------------------
# SECTION II: LIVE MARKET DATA — Full 3-Asset Implementation
# ------------------------------------------------------------
# Scale the same transformation to 3 real forex pairs over 30 days.

print("\n\n" + "=" * 60)
print("SECTION II: Live Market Data Linear Transformation")
print("=" * 60)

# --- Fetch Historical Closing Prices ---
tickers = ['EURUSD=X', 'GBPUSD=X', 'JPY=X']

market_data = yf.download(tickers, period="30d", interval="1d", progress=False, auto_adjust=True)

# Drop any missing data; E.g., An incomplete row meaning an asset hasn't fully updated/settled yet may have NaN values.
closing_prices = market_data['Close'].dropna()
price_matrix_live = closing_prices.to_numpy()
dates = closing_prices.index

# --- Unit Normalization (How many USD is one unit of this asset worth today?) ---
# EUR/USD & GBP/USD are expressed as "USD per 1 unit of foreign currency".
# USD/JPY is the INVERSE and expresses how many Yen = 1 USD.
# To keep all columns in USD per unit, invert the JPY column such that 
#   if USD/JPY = 149.50, the inverse is 1 / 149.50 ≈ 0.006689 USD per 1 JPY

normalized_matrix = price_matrix_live.copy()
normalized_matrix[:, 2] = 1 / normalized_matrix[:, 2]  # Invert column 2 (JPY)

print(f"\nRaw USD/JPY rate (row 0):      {price_matrix_live[0, 2]:.2f} JPY per USD")
print(f"Normalized JPY value (row 0):  {normalized_matrix[0, 2]:.6f} USD per JPY")

# --- Define the Position Vector ---
# A standard forex lot is 100,000 units of the BASE currency.
# To demonstrate the matrix transformation, we simulate holding 100,000 units 
#   of the respective foreign currency for each pair:
#       EUR/USD: holding 100,000 EUR (Base)
#       GBP/USD: holding 100,000 GBP (Base)
#       USD/JPY: holding 100,000 JPY (Quote - sized for matrix symmetry)
positions_vector_live = np.array([100000, 100000, 100000])

# ---- VIEW 1: MACRO — Total Portfolio Equity ----
# Dot product aggregates the normalized price matrix (30×3) with the position vector (3,)
#   produces a (30,) equity vector, that is, total daily equity.
total_portfolio_equity = normalized_matrix.dot(positions_vector_live)

print("\n" + "=" * 60)
print("MACRO VIEW: Total Daily Portfolio Equity (USD)")
print("=" * 60)
for i in range(len(total_portfolio_equity)):
    date_str = dates[i].strftime('%Y-%m-%d')
    print(f"  {date_str}: ${total_portfolio_equity[i]:,.2f}")

# ---- VIEW 2: MICRO ----
# Element-wise multiplication (*) preserves the 3-column structure to show 
#   the resulting individual USD contribution of each asset per day prior to aggregation.
individual_equities = normalized_matrix * positions_vector_live

results_table = pd.DataFrame(
    individual_equities,
    index=dates.strftime('%Y-%m-%d'),
    columns=['EUR/USD ($)', 'GBP/USD ($)', 'USD/JPY ($)']
)

# Append the macro total as a verification column ("should equal the sum of the three columns")
results_table['Total Equity ($)'] = total_portfolio_equity

pd.options.display.float_format = '${:,.2f}'.format

print("\n" + "=" * 60)
print("MICRO VIEW: Per-Asset Equity Breakdown + Total Equity")
print("=" * 60)
print(results_table)