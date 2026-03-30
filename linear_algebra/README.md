# Linear Algebra in Financial Modeling

This module applies foundational linear algebra concepts to quantitative finance, currently focusing on portfolio valuation and risk modeling.

## 1. Portfolio Valuation (`portfolio_valuation.py`)
- Computes the daily USD equity of a multi-asset forex portfolio using matrix-vector multiplication ($P \cdot p = v$). 
- It handles real-world API data constraints, including missing data drops and quote currency normalization (e.g., inverting USD/JPY).

### Example Output
```text
============================================================
SECTION I: Static Linear Transformation (Proof of Concept)
============================================================
Positions Vector (lots per asset):
  [2 1]

Price Matrix (USD, rows = days, columns = assets):
[[1.08 1.26]
 [1.09 1.25]]

Resulting Portfolio Value Vector (P · p):
  Day 1: $3.42
  Day 2: $3.43

============================================================
SECTION II: Live Market Data Linear Transformation
============================================================
Raw USD/JPY rate (row 0):      153.61 JPY per USD
Normalized JPY value (row 0):  0.006510 USD per JPY

============================================================
MACRO VIEW: Total Daily Portfolio Equity (USD)
============================================================
  2026-02-17: $255,426.77
  2026-02-18: $254,816.20
  ...
  2026-03-26: $249,878.61
  2026-03-27: $249,333.51

============================================================
MICRO VIEW: Per-Asset Equity Breakdown + Total Equity
============================================================
            EUR/USD ($)  GBP/USD ($)  USD/JPY ($)  Total Equity ($)
Date
2026-02-17  $118,498.85  $136,276.91      $651.00       $255,426.77
2026-02-18  $118,518.52  $135,644.72      $652.96       $254,816.20
...
2026-03-26  $115,614.96  $133,636.24      $627.42       $249,878.61
2026-03-27  $115,350.90  $133,356.45      $626.16       $249,333.51
```

## 2. Portfolio Risk Determinant (`portfolio_risk_determinant.py`)
- Measures the geometric "volume" of portfolio diversification. 
- By calculating the determinant of a $3 \times 3$ correlation matrix, we can mathematically prove whether a portfolio is orthogonally diversified (Det $\approx$ 1) or dangerously collinear (Det $\approx$ 0).

### Example Output
```text
=================================================================
SECTION I: Static Determinant Calculation (Proof of Concept)
=================================================================

1. Perfectly Diversified Portfolio (Identity Matrix):
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
   -> Determinant (3D Volume) = 1.0000  [Maximal Diversification]

2. Redundant Portfolio (90% Cross-Correlation):
[[1.  0.9 0.9]
 [0.9 1.  0.9]
 [0.9 0.9 1. ]]
   -> Determinant (3D Volume) = 0.0280  [Heavily Squished / Collinear]

=================================================================
SECTION II: Live Market Data (Measuring Real Risk Volume)
=================================================================
Fetching 2 years of daily closing prices for both portfolios...
-----------------------------------------------------------------
PORTFOLIO A: Diversified Assets (Gold, BTC, S&P 500)
-----------------------------------------------------------------
Ticker   BTC-USD   GC=F  ^GSPC
Ticker
BTC-USD    1.000  0.118  0.441
GC=F       0.118  1.000  0.138
^GSPC      0.441  0.138  1.000

-> Determinant (3D Volume): 0.787068
-> Status: PASS [Orthogonal Volume Intact]

-----------------------------------------------------------------
PORTFOLIO B: Redundant Assets (SPY, IVV, VOO)
-----------------------------------------------------------------
Ticker    IVV    SPY    VOO
Ticker
IVV     1.000  0.998  0.999
SPY     0.998  1.000  0.998
VOO     0.999  0.998  1.000

-> Determinant (3D Volume): 0.000003725632
-> Status: FAIL [Collinear/Redundant Assets Detected]
```