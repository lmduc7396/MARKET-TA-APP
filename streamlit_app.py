"""Streamlit page for market technical analysis breadth."""

from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pymssql
import streamlit as st

st.set_page_config(page_title="Market TA Dashboard", layout="wide")

DEFAULT_EMA_OPTIONS = [5, 10, 20, 50, 100, 150, 200]
DEFAULT_EMA_SELECTION = [20, 50, 200]


def parse_odbc_connection_string(conn_str: str) -> Dict[str, str]:
    """Parse an ODBC connection string into pymssql keyword arguments."""
    parsed: Dict[str, str] = {}
    if not conn_str:
        return parsed

    for chunk in conn_str.strip().strip(";").split(";"):
        if not chunk or "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        key = key.strip().upper()
        value = value.strip().strip("'\"")
        if value.startswith("{") and value.endswith("}"):
            value = value[1:-1]

        if key == "SERVER":
            server_value = value
            if server_value.lower().startswith("tcp:"):
                server_value = server_value[4:]
            host, _, port_part = server_value.partition(",")
            parsed["server"] = host
            if port_part:
                parsed["port"] = port_part
        elif key in {"UID", "USER", "USERNAME"}:
            parsed["user"] = value
        elif key in {"PWD", "PASSWORD"}:
            parsed["password"] = value
        elif key in {"DATABASE", "DB", "INITIAL CATALOG"}:
            parsed["database"] = value
        elif key == "PORT":
            parsed["port"] = value

    return parsed


def load_db_params() -> Dict[str, str]:
    """Load database parameters from Streamlit secrets or environment variables."""
    params = {
        "server": os.getenv("DB_SERVER", ""),
        "user": os.getenv("DB_USER", ""),
        "password": os.getenv("DB_PASSWORD", ""),
        "database": os.getenv("DB_DATABASE", ""),
        "port": os.getenv("DB_PORT", ""),
    }
    try:
        secrets_container = st.secrets.get("database", st.secrets)
    except Exception:  # Secrets are optional during local development
        secrets_container = {}

    for key in params:
        if key in secrets_container and secrets_container.get(key):
            params[key] = str(secrets_container.get(key))

    connection_string_keys = (
        "TARGET_DB_CONNECTION_STRING",
        "DB_CONNECTION_STRING",
        "DATABASE_CONNECTION_STRING",
    )

    raw_conn_str = ""
    for key in connection_string_keys:
        if key in secrets_container and secrets_container.get(key):
            raw_conn_str = str(secrets_container.get(key))
            break
    if not raw_conn_str:
        for key in connection_string_keys:
            env_value = os.getenv(key, "")
            if env_value:
                raw_conn_str = env_value
                break

    if raw_conn_str:
        params.update(parse_odbc_connection_string(raw_conn_str))

    return params


def create_connection_kwargs(params: Dict[str, str]) -> Dict[str, str]:
    """Validate and normalise connection parameters for pymssql."""
    kwargs = {
        "server": params["server"],
        "user": params["user"],
        "password": params["password"],
        "database": params["database"],
    }
    port = params.get("port")
    if port:
        try:
            kwargs["port"] = int(port)
        except ValueError as exc:  # pragma: no cover - guard against invalid input
            raise ValueError("Database port must be an integer") from exc
    return kwargs


@st.cache_data(show_spinner="Loading market price data...")
def load_price_data_cached(
    conn_kwargs_items: Tuple[Tuple[str, str], ...],
    start_date_str: str,
    end_date_str: str,
) -> pd.DataFrame:
    """Fetch price history from SQL Server and cache the result."""
    conn_kwargs = dict(conn_kwargs_items)
    query = """
        SELECT TICKER, TRADE_DATE, PX_LAST
        FROM Market_Data
        WHERE TRADE_DATE BETWEEN %s AND %s
          AND PX_LAST IS NOT NULL
    """
    try:
        conn = pymssql.connect(**conn_kwargs)
    except pymssql.Error as exc:  # pragma: no cover - connection issues handled at runtime
        raise RuntimeError(f"Database connection failed: {exc}") from exc
    try:
        df = pd.read_sql(query, conn, params=(start_date_str, end_date_str))
    except pymssql.Error as exc:  # pragma: no cover - query issues handled at runtime
        raise RuntimeError(f"Database query failed: {exc}") from exc
    finally:
        conn.close()
    return df


def calculate_rsi(prices: pd.Series, window: int) -> pd.Series:
    """Calculate the smoothed RSI for a price series."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.clip(lower=0, upper=100)


def enrich_with_indicators(
    price_df: pd.DataFrame, ema_periods: Iterable[int], rsi_period: int
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    Dict[int, Dict[str, int]],
    Dict[str, int],
    pd.DataFrame,
    pd.DataFrame,
]:
    """Compute EMA and RSI metrics returning the latest snapshot, counts, and time series."""
    if price_df.empty:
        return price_df, pd.DataFrame(), {}, {}, pd.DataFrame(), pd.DataFrame()

    ema_periods = sorted({int(period) for period in ema_periods})
    df = price_df.copy()
    df["TRADE_DATE"] = pd.to_datetime(df["TRADE_DATE"])
    df.sort_values(["TICKER", "TRADE_DATE"], inplace=True)

    # Exponential moving averages per ticker
    for period in ema_periods:
        column_name = f"EMA_{period}"
        df[column_name] = df.groupby("TICKER", group_keys=False)["PX_LAST"].transform(
            lambda series, span=period: series.ewm(
                span=span, adjust=False, min_periods=span
            ).mean()
        )

    rsi_column = f"RSI_{rsi_period}"
    df[rsi_column] = df.groupby("TICKER", group_keys=False)["PX_LAST"].transform(
        lambda series: calculate_rsi(series, rsi_period)
    )

    latest = df.groupby("TICKER", as_index=False).tail(1).reset_index(drop=True)

    ema_counts: Dict[int, Dict[str, int]] = {}
    ema_timeseries_frames: list[pd.DataFrame] = []
    for period in ema_periods:
        column_name = f"EMA_{period}"
        valid_mask = latest[["PX_LAST", column_name]].notna().all(axis=1)
        under = int((latest.loc[valid_mask, "PX_LAST"] < latest.loc[valid_mask, column_name]).sum())
        above = int((latest.loc[valid_mask, "PX_LAST"] >= latest.loc[valid_mask, column_name]).sum())
        ema_counts[period] = {
            "under": under,
            "above": above,
            "coverage": int(valid_mask.sum()),
        }

        subset = df.dropna(subset=["PX_LAST", column_name]).copy()
        if subset.empty:
            continue
        subset["is_under"] = subset["PX_LAST"] < subset[column_name]
        grouped = (
            subset.groupby("TRADE_DATE")
            .agg(
                under=("is_under", "sum"),
                coverage=("TICKER", "nunique"),
            )
            .reset_index()
        )
        grouped["above"] = grouped["coverage"] - grouped["under"]
        grouped["EMA Period"] = period
        ema_timeseries_frames.append(grouped)

    rsi_series = latest[rsi_column].dropna()
    rsi_counts = {
        "over": int((rsi_series > 70).sum()),
        "under": int((rsi_series < 30).sum()),
        "coverage": int(rsi_series.size),
    }

    ema_timeseries = (
        pd.concat(ema_timeseries_frames, ignore_index=True)
        if ema_timeseries_frames
        else pd.DataFrame()
    )

    rsi_subset = df.dropna(subset=[rsi_column]).copy()
    if not rsi_subset.empty:
        rsi_subset["is_over_70"] = rsi_subset[rsi_column] > 70
        rsi_subset["is_under_30"] = rsi_subset[rsi_column] < 30
        rsi_timeseries = (
            rsi_subset.groupby("TRADE_DATE")
            .agg(
                over_70=("is_over_70", "sum"),
                under_30=("is_under_30", "sum"),
                coverage=("TICKER", "nunique"),
            )
            .reset_index()
        )
    else:
        rsi_timeseries = pd.DataFrame()

    return df, latest, ema_counts, rsi_counts, ema_timeseries, rsi_timeseries


def build_ema_timeseries_chart(ema_timeseries: pd.DataFrame) -> go.Figure | None:
    """Create a line chart showing daily EMA breadth counts."""
    if ema_timeseries.empty:
        return None

    plot_df = ema_timeseries.copy()
    plot_df.sort_values(["TRADE_DATE", "EMA Period"], inplace=True)
    plot_df["EMA Period"] = plot_df["EMA Period"].astype(str)

    fig = px.line(
        plot_df,
        x="TRADE_DATE",
        y="under",
        color="EMA Period",
        labels={
            "TRADE_DATE": "Trade Date",
            "under": "Stocks Under EMA",
            "EMA Period": "EMA",
        },
    )
    fig.update_layout(
        title="Daily Count of Stocks Under EMA",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=0, l=0, r=0),
    )
    return fig


def build_rsi_timeseries_chart(rsi_timeseries: pd.DataFrame) -> go.Figure | None:
    """Create a line chart for daily RSI extremes counts."""
    if rsi_timeseries.empty:
        return None

    plot_df = rsi_timeseries.copy()
    plot_df.sort_values("TRADE_DATE", inplace=True)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df["TRADE_DATE"],
            y=plot_df["over_70"],
            mode="lines",
            name="RSI > 70",
            line=dict(color="#AB63FA"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["TRADE_DATE"],
            y=plot_df["under_30"],
            mode="lines",
            name="RSI < 30",
            line=dict(color="#19D3F3"),
        )
    )
    fig.update_layout(
        title="Daily Count of Stocks at RSI Extremes",
        xaxis_title="Trade Date",
        yaxis_title="Count",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=0, l=0, r=0),
    )
    return fig


def build_latest_snapshot(
    latest: pd.DataFrame, ema_periods: Iterable[int], rsi_period: int
) -> pd.DataFrame:
    """Prepare a tidy snapshot table of the latest indicators per ticker."""
    if latest.empty:
        return latest

    ema_periods = sorted({int(period) for period in ema_periods})
    rsi_column = f"RSI_{rsi_period}"

    snapshot_cols = ["TICKER", "TRADE_DATE", "PX_LAST", rsi_column]
    snapshot_cols.extend(f"EMA_{period}" for period in ema_periods)
    snapshot = latest[snapshot_cols].copy()
    snapshot["TRADE_DATE"] = snapshot["TRADE_DATE"].dt.date

    for period in ema_periods:
        ema_col = f"EMA_{period}"
        snapshot[f"Under EMA {period}"] = np.where(
            snapshot[["PX_LAST", ema_col]].notna().all(axis=1)
            & (snapshot["PX_LAST"] < snapshot[ema_col]),
            "Yes",
            "No",
        )

    rename_map = {
        "TICKER": "Ticker",
        "TRADE_DATE": "Trade Date",
        "PX_LAST": "Close",
        rsi_column: f"RSI ({rsi_period})",
    }
    for period in ema_periods:
        rename_map[f"EMA_{period}"] = f"EMA {period}"

    snapshot.rename(columns=rename_map, inplace=True)
    snapshot.sort_values("Ticker", inplace=True)
    snapshot.set_index("Ticker", inplace=True)
    return snapshot


def main() -> None:
    st.title("Market Technical Analysis Overview")
    st.caption("Breadth signals based on EMA positioning and RSI levels across the market")

    params = load_db_params()

    today = date.today()
    default_start = today - timedelta(days=180)

    with st.sidebar:
        st.header("Analysis Settings")
        start_date_value = st.date_input("Start date", value=default_start)
        end_date_value = st.date_input("End date", value=today)
        ema_selection = st.multiselect(
            "EMA periods",
            options=DEFAULT_EMA_OPTIONS,
            default=DEFAULT_EMA_SELECTION,
            help="Select one or more EMA windows to evaluate",
        )
        rsi_period_value = st.slider(
            "RSI window",
            min_value=5,
            max_value=30,
            step=1,
            value=14,
            help="Number of days used in the RSI calculation",
        )
        st.caption("Database credentials are loaded securely from Streamlit secrets.")

    field_labels = {
        "server": "Server",
        "database": "Database",
        "user": "Username",
        "password": "Password",
    }
    missing_fields = [label for key, label in field_labels.items() if not params.get(key)]
    if missing_fields:
        st.warning(
            "Please provide database credentials: " + ", ".join(missing_fields),
            icon="⚠️",
        )
        st.stop()

    if start_date_value > end_date_value:
        st.error("Start date must be on or before the end date.")
        st.stop()

    if not ema_selection:
        st.warning("Select at least one EMA period to proceed.", icon="ℹ️")
        st.stop()

    try:
        connection_kwargs = create_connection_kwargs(params)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    connection_items = tuple(sorted(connection_kwargs.items()))
    start_iso = start_date_value.isoformat()
    end_iso = end_date_value.isoformat()

    try:
        price_df = load_price_data_cached(connection_items, start_iso, end_iso)
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    if price_df.empty:
        st.info("No market price data returned for the selected window.")
        st.stop()

    ema_periods = sorted({int(period) for period in ema_selection})

    with st.spinner("Calculating indicators..."):
        (
            _full_df,
            latest_snapshot_raw,
            ema_counts,
            rsi_counts,
            ema_timeseries,
            rsi_timeseries,
        ) = enrich_with_indicators(price_df, ema_periods, int(rsi_period_value))

    ema_chart = build_ema_timeseries_chart(ema_timeseries)
    rsi_chart = build_rsi_timeseries_chart(rsi_timeseries)
    snapshot_table = build_latest_snapshot(
        latest_snapshot_raw, ema_periods, int(rsi_period_value)
    )

    st.subheader("EMA Breadth (Daily)")
    if ema_chart:
        st.plotly_chart(ema_chart, use_container_width=True)
    else:
        st.write("EMA statistics unavailable for the selected inputs.")

    if ema_counts:
        ema_summary = pd.DataFrame(
            {
                "EMA Period": [str(period) for period in ema_periods],
                "Under EMA": [
                    ema_counts.get(period, {}).get("under", 0) for period in ema_periods
                ],
                "Above EMA": [
                    ema_counts.get(period, {}).get("above", 0) for period in ema_periods
                ],
                "Coverage": [
                    ema_counts.get(period, {}).get("coverage", 0)
                    for period in ema_periods
                ],
            }
        )
        ema_summary["Under %"] = np.where(
            ema_summary["Coverage"] > 0,
            (ema_summary["Under EMA"] / ema_summary["Coverage"]) * 100,
            np.nan,
        )
        ema_summary["Under %"] = ema_summary["Under %"].map(
            lambda value: f"{value:.1f}%" if pd.notna(value) else ""
        )
        st.dataframe(
            ema_summary.set_index("EMA Period"),
            use_container_width=True,
        )

    st.subheader("RSI Extremes (Daily)")
    if rsi_chart:
        st.plotly_chart(rsi_chart, use_container_width=True)
    else:
        st.write("Insufficient data to calculate RSI extremes.")

    if rsi_counts:
        rsi_summary = pd.DataFrame(
            {
                "Metric": ["RSI > 70", "RSI < 30", "Coverage"],
                "Count": [
                    rsi_counts.get("over", 0),
                    rsi_counts.get("under", 0),
                    rsi_counts.get("coverage", 0),
                ],
            }
        )
        st.dataframe(rsi_summary.set_index("Metric"), use_container_width=True)

    st.subheader("Latest Indicator Snapshot")
    if not snapshot_table.empty:
        st.dataframe(snapshot_table, use_container_width=True)
    else:
        st.write("Snapshot unavailable.")

    st.caption(
        "Data source: Market_Data table (close price only). Refresh the page after the daily update to pick up new data."
    )


if __name__ == "__main__":
    main()
