import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

SALES_FILE = "slim_sales.csv"
INV_FILE = "slim_inventory.csv"
HORIZON = 30


# ---------------------------------------------------------------- data
@st.cache_data
def load_data():
    sales = pd.read_csv(SALES_FILE, parse_dates=["Transaction Date"])
    inv = pd.read_csv(INV_FILE, parse_dates=["Start Date", "End Date"])
    last_date = sales["Transaction Date"].max()
    # open-ended stock records use 9999-12-31 -> cap at last sales date
    inv["End Date"] = inv["End Date"].clip(upper=last_date)
    return sales, inv


def daily_demand(sales, category):
    """Daily units sold for one category (returns excluded = gross demand)."""
    df = sales[(sales["Product Category"] == category) & (sales["Is Return"] == 0)]
    s = df.groupby("Transaction Date")["Qty Sold"].sum()
    full = pd.date_range(sales["Transaction Date"].min(), sales["Transaction Date"].max())
    return s.reindex(full, fill_value=0).clip(lower=0).astype(float)


def current_stock(inv, category, as_of):
    df = inv[(inv["Product Category"] == category)
             & (inv["Start Date"] <= as_of) & (inv["End Date"] >= as_of)]
    return int(df["Qty on hand"].clip(lower=0).sum())


# ---------------------------------------------------------------- model
def make_features(history, date):
    """Features for `date`, using only values before `date`."""
    h = history[history.index < date]
    return [
        h.iloc[-1], h.iloc[-7], h.iloc[-14],
        h.iloc[-7:].mean(), h.iloc[-28:].mean(),
        date.dayofweek, date.month, date.day,
    ]


def build_training(series):
    X, y = [], []
    for i in range(28, len(series)):
        X.append(make_features(series.iloc[:i], series.index[i]))
        y.append(series.iloc[i])
    return np.array(X), np.array(y)


def fit_model(series):
    X, y = build_training(series)
    model = RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                  random_state=42, n_jobs=-1)
    return model.fit(X, y)


def recursive_forecast(model, series, horizon=HORIZON):
    """Each predicted day is appended to history and used for the next day."""
    hist = series.copy()
    preds = []
    for _ in range(horizon):
        d = hist.index[-1] + pd.Timedelta(days=1)
        p = max(0.0, float(model.predict([make_features(hist, d)])[0]))
        hist.loc[d] = p
        preds.append(p)
    idx = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon)
    return pd.Series(preds, index=idx)


def naive_forecast(series, horizon=HORIZON):
    """Baseline: same weekday last week."""
    last7 = series.iloc[-7:].values
    idx = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon)
    return pd.Series([last7[i % 7] for i in range(horizon)], index=idx)


def metrics(actual, pred):
    err = np.abs(actual.values - pred.values)
    return {"MAE": err.mean(), "WMAPE %": 100 * err.sum() / max(actual.sum(), 1e-9)}


def backtest(series):
    train, test = series.iloc[:-HORIZON], series.iloc[-HORIZON:]
    model = fit_model(train)
    rf = recursive_forecast(model, train)
    nv = naive_forecast(train)
    # one-step-ahead: predict each test day using real history up to the day before
    rf1 = pd.Series([model.predict([make_features(series, d)])[0] for d in test.index], index=test.index)
    nv1 = pd.Series([series[series.index < d].iloc[-7] for d in test.index], index=test.index)
    res = {
        "RF - 30-day recursive": metrics(test, rf),
        "Baseline - 30-day (last week repeated)": metrics(test, nv),
        "RF - next-day": metrics(test, rf1),
        "Baseline - next-day (same weekday last week)": metrics(test, nv1),
    }
    return test, rf, nv, res


# ---------------------------------------------------------------- inventory
Z = {"90%": 1.28, "95%": 1.65, "99%": 2.33}


def inventory_plan(series, forecast, stock, lead_time, service, review_days, on_order):
    mean_daily = forecast.mean()
    std_daily = series.iloc[-90:].std()
    lead_demand = mean_daily * lead_time
    safety = Z[service] * std_daily * np.sqrt(lead_time)
    rop = lead_demand + safety
    target = mean_daily * (lead_time + review_days) + safety
    order_qty = max(0, target - stock - on_order)
    return {"Avg forecast demand / day": mean_daily, "Lead-time demand": lead_demand,
            "Safety stock": safety, "Reorder point": rop, "Order-up-to level": target,
            "Current stock": stock, "Recommended order qty": order_qty,
            "Reorder now?": "YES" if stock + on_order <= rop else "No"}


# ---------------------------------------------------------------- UI
def main():
    st.set_page_config(page_title="Demand Forecasting & Inventory", layout="wide")
    st.title("Product Demand Forecasting and Inventory Optimization")
    sales, inv = load_data()
    last_date = sales["Transaction Date"].max()

    st.sidebar.header("Settings")
    cats = sales["Product Category"].value_counts().index.tolist()
    category = st.sidebar.selectbox("Product category", cats)
    lead_time = st.sidebar.slider("Supplier lead time (days) - ASSUMPTION", 1, 30, 7)
    service = st.sidebar.selectbox("Service level - ASSUMPTION", list(Z), index=1)
    review_days = st.sidebar.slider("Review period (days) - ASSUMPTION", 1, 30, 7)
    on_order = st.sidebar.number_input("Units already on order", 0, 100000, 0)
    st.sidebar.caption(f"Data ends {last_date.date()}. Forecast dates are a historical demonstration.")

    series = daily_demand(sales, category)
    tab1, tab2, tab3 = st.tabs(["Sales analysis", "Demand forecast", "Inventory"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Units sold (excl. returns)", f"{int(series.sum()):,}")
        c2.metric("Avg units / day", f"{series.mean():.1f}")
        c3.metric("Categories in data", len(cats))
        st.subheader(f"Weekly units sold - {category}")
        st.line_chart(series.resample("W").sum())
        cat_tot = (sales[sales["Is Return"] == 0].groupby("Product Category")["Qty Sold"]
                   .sum().sort_values(ascending=False).head(10))
        st.subheader("Top 10 categories by units sold")
        st.bar_chart(cat_tot)
        st.subheader("Units sold by Sales Type")
        st.bar_chart(sales[sales["Is Return"] == 0].groupby("Sales Type")["Qty Sold"].sum())

    with tab2:
        with st.spinner("Training Random Forest..."):
            test, rf_bt, nv_bt, res = backtest(series)
            model = fit_model(series)
            forecast = recursive_forecast(model, series)
        st.subheader("Model validation (last 30 days held out)")
        st.dataframe(pd.DataFrame(res).T.round(2))
        st.line_chart(pd.DataFrame({"Actual": test, "Random Forest": rf_bt, "Baseline": nv_bt}))
        if res["RF - 30-day recursive"]["WMAPE %"] > res["Baseline - 30-day (last week repeated)"]["WMAPE %"]:
            st.warning("Over the full 30 days, Random Forest did not beat the baseline for this category.")
        st.subheader(f"Next {HORIZON} days forecast ({forecast.index[0].date()} to {forecast.index[-1].date()})")
        st.line_chart(pd.concat([series.iloc[-60:].rename("History"), forecast.rename("Forecast")], axis=1))
        st.metric("Forecast total units", f"{forecast.sum():.0f}")

    with tab3:
        stock = current_stock(inv, category, last_date)
        plan = inventory_plan(series, forecast, stock, lead_time, service, review_days, on_order)
        cols = st.columns(4)
        for i, (k, v) in enumerate(plan.items()):
            cols[i % 4].metric(k, v if isinstance(v, str) else f"{v:,.0f}")
        st.caption("Stock = units on hand across all stores at the last data date. "
                   "Lead time, service level, review period and on-order units are assumptions.")


if __name__ == "__main__":
    main()