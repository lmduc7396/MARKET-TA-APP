# Database Schema Reference Guide
**Dragon Capital Financial Data Pipeline**

## Overview
This document provides a comprehensive reference for all database tables created and maintained by the financial data pipeline. Tables are organized by functional area with detailed schema information, relationships, and data characteristics.

---

## Table of Contents
1. [Financial Statement Tables](#financial-statement-tables)
2. [Market Data Tables](#market-data-tables)
3. [Banking Analytics Tables](#banking-analytics-tables)
4. [Reference Data Tables](#reference-data-tables)
5. [Table Relationships](#table-relationships)
6. [Data Update Patterns](#data-update-patterns)

---

## Financial Statement Tables

### FA_Quarterly
**Purpose**: Quarterly financial statement data for all listed companies
**Update Frequency**: Weekly
**Data Range**: 2016 - Present
**Row Count**: ~500,000+

| Column | Data Type | Nullable | Description | Example |
|--------|-----------|----------|-------------|---------|
| **TICKER** | NVARCHAR(50) | NO | Stock symbol (3 letters) | 'VNM' |
| **KEYCODE** | NVARCHAR(50) | NO | Financial metric identifier | 'Net_Revenue' |
| **DATE** | NVARCHAR(50) | NO | Quarter in YYYYQX format | '2024Q3' |
| VALUE | FLOAT | YES | Metric value in VND | 15234567890.0 |
| YEAR | BIGINT | YES | Extracted year for filtering | 2024 |
| YoY | FLOAT | YES | Year-over-year growth rate | 0.085 (8.5%) |

**Primary Key**: TICKER + KEYCODE + DATE
**Common KEYCODE Values**:
- Income Statement: Net_Revenue, COGS, Gross_Profit, EBIT, EBITDA, NPAT, NPATMI
- Balance Sheet: Total_Asset, Total_Liabilities, TOTAL_Equity, Cash, ST_Debt, LT_Debt
- Cash Flow: Operating_CF, Inv_CF, Fin_CF, FCF, Capex
- Margins: Gross_Margin, EBIT_Margin, EBITDA_Margin, NPAT_Margin

---

### FA_Annual
**Purpose**: Annual financial statement data
**Update Frequency**: Weekly
**Data Range**: 2016 - Present
**Row Count**: ~125,000+

| Column | Data Type | Nullable | Description | Example |
|--------|-----------|----------|-------------|---------|
| **TICKER** | NVARCHAR(50) | NO | Stock symbol | 'VNM' |
| **KEYCODE** | NVARCHAR(50) | NO | Financial metric identifier | 'Net_Revenue' |
| **DATE** | NVARCHAR(50) | NO | Year as string | '2024' |
| VALUE | FLOAT | YES | Annual metric value in VND | 61234567890.0 |
| YEAR | BIGINT | YES | Year as integer | 2024 |
| YoY | FLOAT | YES | Year-over-year growth | 0.092 |

**Primary Key**: TICKER + KEYCODE + DATE
**Note**: Contains same KEYCODE values as FA_Quarterly but with annual aggregations

---

## Market Data Tables

### Market_Data
**Purpose**: Comprehensive daily market data including OHLC prices, valuation multiples, and EV/EBITDA
**Update Frequency**: Daily
**Data Range**: 2018 - Present
**Row Count**: ~1,400,000+

| Column | Data Type | Nullable | Description | Example |
|--------|-----------|----------|-------------|---------|
| **TICKER** | VARCHAR(10) | NO | Stock symbol (extracted from PRIMARYSECID) | 'VNM' |
| **TRADE_DATE** | DATE | NO | Trading date | '2024-09-23' |
| PE | FLOAT | YES | Price-to-Earnings ratio | 18.5 |
| PB | FLOAT | YES | Price-to-Book ratio | 3.2 |
| PS | FLOAT | YES | Price-to-Sales ratio | 2.8 |
| PX_OPEN | FLOAT | YES | Opening price | 67500 |
| PX_HIGH | FLOAT | YES | Daily high price | 68200 |
| PX_LOW | FLOAT | YES | Daily low price | 67000 |
| PX_LAST | FLOAT | YES | Closing/Last price | 67800 |
| MKT_CAP | FLOAT | YES | Market capitalization | 145678.5 |
| EV_EBITDA | FLOAT | YES | Enterprise Value/EBITDA ratio | 12.3 |
| UPDATE_TIMESTAMP | DATETIME | YES | Last update timestamp | '2024-09-23 18:30:00' |

**Primary Key**: TICKER + TRADE_DATE
**Data Sources**:
- Bloomberg (SIL.S_BBG_DATA_DWH_ADJUSTED): PE, PB, PS, PX_OPEN, PX_HIGH, PX_LOW, PX_LAST, MKT_CAP
- IRIS (SIL.W_F_IRIS_CALCULATE): EV_EBITDA
**Data Quality Notes**:
- PX_ prefix used for price columns to avoid SQL reserved keywords
- NULL values indicate data not available or not calculable
- Price relationships validated: PX_HIGH >= PX_LAST >= PX_LOW
- Extreme valuation ratios capped (PE < 1000, PB < 100, PS < 100)
- Updated via standalone valuation_ohlc_extractor script

---

### MarketCap
**Purpose**: Latest market capitalization snapshot
**Update Frequency**: Daily
**Data Range**: Current snapshot only
**Row Count**: ~1,700 (all listed stocks)

| Column | Data Type | Nullable | Description | Example |
|--------|-----------|----------|-------------|---------|
| **TICKER** | NVARCHAR(50) | NO | Stock symbol | 'VNM' |
| CUR_MKT_CAP | FLOAT | YES | Market cap in billions VND | 145678.5 |
| **TRADE_DATE** | DATETIME | YES | Date of snapshot | '2024-09-23' |

**Primary Key**: TICKER + TRADE_DATE
**Note**: Only contains latest values, historical data in separate archive

---

### MarketIndex
**Purpose**: Stock market index historical data (HOSE)
**Update Frequency**: Daily
**Data Range**: 2016 - Present
**Row Count**: ~2,000+

| Column | Data Type | Nullable | Description | Example |
|--------|-----------|----------|-------------|---------|
| **COMGROUPCODE** | NVARCHAR(50) | NO | Index identifier | 'VNINDEX' |
| **TRADINGDATE** | DATETIME | NO | Trading date | '2024-09-23' |
| INDEXVALUE | FLOAT | YES | Closing index value | 1285.67 |
| PRIORINDEXVALUE | FLOAT | YES | Previous day's close | 1278.45 |
| HIGHEST | FLOAT | YES | Intraday high | 1290.12 |
| LOWEST | FLOAT | YES | Intraday low | 1275.30 |
| TOTALSHARE | BIGINT | YES | Total shares traded | 567890123 |
| TOTALVALUE | FLOAT | YES | Total value traded (VND) | 12345678901234 |
| FOREIGNBUYVOLUME | BIGINT | YES | Foreign buying volume | 12345678 |
| FOREIGNSELLVOLUME | BIGINT | YES | Foreign selling volume | 11234567 |

**Primary Key**: COMGROUPCODE + TRADINGDATE

---

## Banking Analytics Tables

### BankingMetrics
**Purpose**: Comprehensive banking metrics including 26 calculated ratios (CA.1-CA.26)
**Update Frequency**: Quarterly
**Data Range**: 2017 - Present
**Row Count**: ~10,000+

| Column | Data Type | Nullable | Description | Example |
|--------|-----------|----------|-------------|---------|
| **TICKER** | NVARCHAR(20) | NO | Bank ticker or tier aggregate | 'VCB' or 'SOCB' |
| **YEARREPORT** | INT | NO | Reporting year | 2024 |
| **LENGTHREPORT** | INT | NO | 1-4 for Q1-Q4, 5 for annual | 3 |
| DATE | DATE | YES | End date of period | '2024-09-30' |
| DATE_STRING | NVARCHAR(20) | YES | Formatted period | '2024-Q3' |
| BANK_TYPE | NVARCHAR(20) | YES | Classification or 'Aggregate' | 'SOCB' |
| PERIOD_TYPE | NVARCHAR(10) | YES | 'Q' or 'Y' | 'Q' |

**Source Data Columns** (Examples):
| Column | Description | Typical Range |
|--------|-------------|---------------|
| TOI | Total Operating Income | 1000-50000 (bn VND) |
| Net Interest Income | Interest revenue | 500-30000 |
| OPEX | Operating expenses | -500 to -20000 |
| Loan | Customer loans | 10000-1000000 |
| Deposit | Customer deposits | 10000-1000000 |
| Total Assets | Balance sheet total | 50000-2000000 |

**Calculated Metrics (CA.1-CA.26)**:
| Metric | Description | Formula | Typical Range |
|--------|-------------|---------|---------------|
| LDR (CA.1) | Loan-to-Deposit Ratio | Loan/Deposit | 0.7-1.0 |
| CASA (CA.2) | Current/Savings ratio | (Nt.121+124+125)/Deposit | 0.15-0.40 |
| NPL (CA.3) | Non-performing loans | (Nt.68+69+70)/Loan | 0.005-0.03 |
| CIR (CA.6) | Cost-to-Income | -OPEX/TOI | 0.3-0.6 |
| NIM (CA.13) | Net Interest Margin | NII/Avg(Assets) | 0.02-0.05 |
| ROA (CA.16) | Return on Assets | NetProfit/Avg(Assets) | 0.005-0.02 |
| ROE (CA.17) | Return on Equity | NPATMI/Avg(Equity) | 0.10-0.25 |

**Primary Key**: TICKER + YEARREPORT + LENGTHREPORT

**Special TICKER Values for Aggregates**:
- 'SOCB': State-owned commercial banks aggregate
- 'Private_1': Tier 1 private banks
- 'Private_2': Tier 2 private banks
- 'Private_3': Tier 3 private banks
- 'Sector': Entire banking sector

---

### Banking_Comments
**Purpose**: Qualitative commentary and analysis notes for banks
**Update Frequency**: Quarterly (manual)
**Data Range**: As available
**Row Count**: Variable

| Column | Data Type | Nullable | Description | Example |
|--------|-----------|----------|-------------|---------|
| **TICKER** | NVARCHAR(50) | NO | Bank ticker | 'VCB' |
| SECTOR | NVARCHAR(50) | YES | Banking sector/type | 'SOCB' |
| **DATE** | NVARCHAR(50) | NO | Quarter in YYYYQX format | '2024Q3' |
| COMMENT | NVARCHAR(MAX) | YES | Analysis text | 'Strong credit growth...' |

**Primary Key**: TICKER + DATE

---

## Reference Data Tables

### Sector_Map
**Purpose**: Master reference for ticker classification and index membership
**Update Frequency**: As needed
**Data Range**: All listed tickers
**Row Count**: 433

| Column | Data Type | Nullable | Description | Example |
|--------|-----------|----------|-------------|---------|
| OrganCode | NVARCHAR(20) | YES | Organization code | 'VNMILK' |
| **Ticker** | NVARCHAR(10) | NO | Stock ticker | 'VNM' |
| ExportClassification | NVARCHAR(10) | YES | Export flag | 'Export' |
| Sector | NVARCHAR(20) | NO | Primary sector | 'Consumer' |
| L1 | NVARCHAR(30) | NO | Level 1 industry | 'Consumer Staples' |
| L2 | NVARCHAR(25) | NO | Level 2 industry | 'Food & Beverage' |
| L3 | NVARCHAR(25) | YES | Level 3 sub-industry | 'Dairy' |
| VNI | NVARCHAR(1) | YES | VN30 Index member | 'Y' or NULL |

**Primary Key**: Ticker

**Sector Distribution**:
- Consumer: ~100 tickers
- Industrial: ~150 tickers
- Service: ~80 tickers
- Financial: ~40 tickers
- Resources: ~60 tickers

**VNI Membership**: 37 tickers marked with 'Y'

---

## Table Relationships

### Primary Relationships
```
FA_Quarterly/FA_Annual
    ↓ [TICKER]
Sector_Map ← [TICKER] → MarketCap
    ↓ [TICKER]           ↓ [TICKER]
Valuation ← [TICKER] → BankingMetrics
                         ↓ [TICKER]
                    Banking_Comments
```

### Key Relationships
1. **TICKER** is the universal join key across all tables
2. **DATE** formats vary by table:
   - Financial: 'YYYYQX' or 'YYYY'
   - Banking: YEARREPORT + LENGTHREPORT
   - Market: DATETIME
3. **Sector_Map** provides classification for all tickers
4. **BankingMetrics** includes both individual banks and aggregates

---

## Data Update Patterns

### Daily Updates
- **MarketCap**: Full replacement with latest snapshot
- **Valuation**: Incremental addition of new trading day
- **MarketIndex**: Incremental addition of new trading day

### Weekly Updates
- **FA_Quarterly**: Incremental update for reporting companies
- **FA_Annual**: Incremental update (mainly during annual reporting season)

### Quarterly Updates
- **BankingMetrics**: Full refresh with new quarter data
- **Banking_Comments**: Manual updates as analysis completed

### On-Demand Updates
- **Sector_Map**: When new listings or reclassifications occur

---

## Query Examples

### 1. Get latest financials for a ticker
```sql
SELECT KEYCODE, VALUE, YoY
FROM FA_Quarterly
WHERE TICKER = 'VNM'
  AND DATE = (SELECT MAX(DATE) FROM FA_Quarterly WHERE TICKER = 'VNM')
ORDER BY KEYCODE
```

### 2. Banking peer comparison
```sql
SELECT TICKER, BANK_TYPE,
       [LDR] as LoanDeposit,
       [NPL] as NPL_Ratio,
       [ROE] as ReturnEquity
FROM BankingMetrics
WHERE YEARREPORT = 2024 AND LENGTHREPORT = 2
  AND TICKER IN ('VCB', 'CTG', 'BID', 'TCB', 'MBB')
ORDER BY [ROE] DESC
```

### 3. Sector performance overview
```sql
SELECT s.Sector,
       COUNT(DISTINCT m.TICKER) as StockCount,
       AVG(m.CUR_MKT_CAP) as AvgMarketCap
FROM Sector_Map s
JOIN MarketCap m ON s.Ticker = m.TICKER
GROUP BY s.Sector
ORDER BY AvgMarketCap DESC
```

### 4. VN30 Index members valuation
```sql
SELECT s.Ticker, s.L1, v.[P/E], v.[P/B], v.[EV/EBITDA]
FROM Sector_Map s
JOIN Valuation v ON s.Ticker = v.TICKER
WHERE s.VNI = 'Y'
  AND v.TRADE_DATE = (SELECT MAX(TRADE_DATE) FROM Valuation)
ORDER BY v.[P/E]
```

---

## Data Quality Notes

### Common Data Patterns
- **NULL handling**: NULL values indicate data not available or not applicable
- **YoY calculations**: First year/quarter will have NULL YoY values
- **Banking aggregates**: TICKER values like 'SOCB' represent tier aggregates
- **Date formats**: Inconsistent across tables - use appropriate conversion

### Data Validation Rules
- All TICKER values should exist in Sector_Map
- Financial metrics should have consistent KEYCODEs
- Banking metrics CA.1-CA.26 follow specific calculation rules
- Ratios bounded by business logic (e.g., NPL typically < 5%)

### Known Limitations
- Historical data starts from 2016 (2017 for banking)
- Some companies may have incomplete quarterly data
- Banking metrics require auxiliary Excel files for full calculations
- Market data updated with 1-day lag

---

## Contact & Support
For questions about data definitions, calculations, or access:
- Pipeline Documentation: `.docs/` directory
- Column Mappings: `unified_pipeline/column_mappings.py`
- Banking Calculations: `unified_pipeline/banking_functions.py`

Last Updated: September 2025