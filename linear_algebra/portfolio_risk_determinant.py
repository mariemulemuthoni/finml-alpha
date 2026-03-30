"""
    ================================================================
    II. Portfolio Risk and Diversification via Matrix Determinants
    ================================================================

    Measures the geometric "volume" of diversification in a multi-asset
    portfolio using the determinant of its correlation matrix.

    The key insight is that a correlation matrix is a linear transformation
    of asset return space. Its determinant measures the volume of that space:
        - Det(R) ≈ 1.0 : Assets are orthogonal — each contributes an
                         independent dimension of risk (maximum diversification)
        - Det(R) ≈ 0.0 : Assets are collinear — they collapse the space
                         toward a flat plane or line (redundant, undiversified)

    In geometric terms, a 3x3 identity correlation matrix maps to a perfect
    unit cube (volume = 1). High cross-correlations progressively squish that
    cube flat. The determinant measures how much volume remains.

    Note on methodology: This script uses the correlation matrix for clarity
    and comparability across assets. In production risk systems, the covariance
    matrix of returns is preferred — its determinant is the "generalized variance"
    and preserves magnitude information that correlation matrices normalize away.

    Structure:
        Section I  — Static proof comparing an identity matrix to a highly
                     correlated matrix (with hand-verifiable arithmetic)
        Section II — Live implementation comparing a diversified portfolio
                     (Gold, BTC, S&P 500) against a redundant one
                     (SPY, IVV, VOO) using 2 years of Yahoo Finance data
"""

import numpy as np
import pandas as pd
import yfinance as yf

# ============================================================
# SECTION I: THE MATH
# ============================================================
# Verify the determinant as a diversification measure with traceable numbers.

print("=" * 65)
print("SECTION I: Static Determinant Calculation (Proof of Concept)")
print("=" * 65)

# --- Perfectly Diversified: Identity Matrix ---
# Each asset correlates only with itself (diagonal = 1, off-diagonal = 0).
# Geometrically: a perfect orthogonal unit cube in 3D space.
# Det(I) = 1.0 by definition — the transformation preserves all volume.
matrix_diversified = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
])

# --- Highly Correlated: 90% Cross-Correlation Matrix ---
# All three assets move together 90% of the time.
# Geometrically: the unit cube is heavily squished toward a flat plane.
# Hand verification (cofactor expansion along row 0):
#   Det = 1*(1*1 - 0.9*0.9) - 0.9*(0.9*1 - 0.9*0.9) + 0.9*(0.9*0.9 - 1*0.9)
#       = 1*(0.19) - 0.9*(0.09) + 0.9*(-0.09)
#       = 0.19 - 0.081 - 0.081 = 0.028
matrix_redundant = np.array([
    [1.0, 0.9, 0.9],
    [0.9, 1.0, 0.9],
    [0.9, 0.9, 1.0]
])

det_div = np.linalg.det(matrix_diversified)
det_red = np.linalg.det(matrix_redundant)

print("\n1. Perfectly Diversified Portfolio (Identity Matrix):")
print(matrix_diversified)
print(f"   -> Determinant (3D Volume) = {det_div:.4f}  [Maximal Diversification]")

print("\n2. Redundant Portfolio (90% Cross-Correlation):")
print(matrix_redundant)
print(f"   -> Determinant (3D Volume) = {det_red:.4f}  [Heavily Squished / Collinear]")


# ============================================================
# SECTION II: LIVE MARKET DATA — Measuring Real Portfolio Risk
# ============================================================
# Apply the same determinant test to two real portfolios with contrasting correlation structures. 
# We use daily percentage returns, not raw prices — 
#   correlation of price levels is spurious; correlation of returns is meaningful.

print("\n\n" + "=" * 65)
print("SECTION II: Live Market Data (Measuring Real Risk Volume)")
print("=" * 65)

# Portfolio A: Historically distinct asset classes.
# Gold (GC=F) is a macro hedge, BTC is a risk asset with low equity beta, S&P 500 (^GSPC) is broad equity — three genuinely different risk drivers.
tickers_diversified = ['GC=F', 'BTC-USD', '^GSPC']

# Portfolio B: Three ETFs tracking the exact same index (S&P 500).
# SPY, IVV, and VOO are functionally identical — maximum possible collinearity.
# A near-zero determinant here is the expected and correct result.
tickers_redundant = ['SPY', 'IVV', 'VOO']

print("\nFetching 2 years of daily closing prices for both portfolios...")

data_div = yf.download(tickers_diversified, period="2y", interval="1d", progress=False, auto_adjust=True)['Close']
data_red = yf.download(tickers_redundant, period="2y", interval="1d", progress=False, auto_adjust=True)['Close']

# --- Compute Daily Percentage Returns ---
# pct_change() gives day-over-day return: (P_t - P_{t-1}) / P_{t-1}
# fill_method=None prevents deprecated forward-fill behavior before differencing.
returns_div = data_div.pct_change(fill_method=None).dropna()
returns_red = data_red.pct_change(fill_method=None).dropna()

# --- Build Correlation Matrices ---
# Each cell (i, j) is the Pearson correlation between asset i and asset j's daily returns.
# Diagonal is always 1.0 (every asset is perfectly correlated with itself).
corr_matrix_div = returns_div.corr()
corr_matrix_red = returns_red.corr()

# --- Compute Determinants ---
# Extract the underlying NumPy array before passing to linalg.det — numpy operates on arrays, not DataFrames.
det_live_div = np.linalg.det(corr_matrix_div.to_numpy())
det_live_red = np.linalg.det(corr_matrix_red.to_numpy())

# --- Evaluate Diversification Status ---
# A threshold of 0.1 is a reasonable heuristic: 
#   below it, the portfolio's risk space has collapsed to the point where assets are near-redundant.
DIVERSIFICATION_THRESHOLD = 0.1
status_div = "PASS [Orthogonal Volume Intact]" if det_live_div >= DIVERSIFICATION_THRESHOLD else "FAIL [Collinear/Redundant Assets Detected]"
status_red = "PASS [Orthogonal Volume Intact]" if det_live_red >= DIVERSIFICATION_THRESHOLD else "FAIL [Collinear/Redundant Assets Detected]"

print("\n" + "-" * 65)
print("PORTFOLIO A: Diversified Assets (Gold, BTC, S&P 500)")
print("-" * 65)
print(corr_matrix_div.round(3))
print(f"\n-> Determinant (3D Volume): {det_live_div:.6f}")
print(f"-> Status: {status_div}")

print("\n" + "-" * 65)
print("PORTFOLIO B: Redundant Assets (SPY, IVV, VOO)")
print("-" * 65)
print(corr_matrix_red.round(3))
print(f"\n-> Determinant (3D Volume): {det_live_red:.12f}")
print(f"-> Status: {status_red}")