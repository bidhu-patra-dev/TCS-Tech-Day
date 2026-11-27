import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

st.set_page_config(page_title="InventoryAI", page_icon="📦", layout="wide")
st.title("📦 InventoryAI — Smart Demand Forecasting & Replenishment")
st.caption("Sample-data prototype. Replace the CSV files with your real store data before making operational decisions.")

@st.cache_data
def load_data():
    sales = pd.read_csv("sales_history.csv", parse_dates=["date"])
    inventory = pd.read_csv("inventory.csv")
    products = pd.read_csv("products.csv")
    return sales, inventory, products

try:
    sales, inventory, products = load_data()
except Exception as e:
    st.error("Could not load the CSV files. Keep app.py and all three CSV files in the same folder.")
    st.exception(e)
    st.stop()

required_sales = {"date","product_id","units_sold","selling_price","discount_pct"}
required_inv = {"product_id","current_stock","lead_time_days","safety_stock_units","on_order_units"}
required_products = {"product_id","product_name","category","supplier","unit_cost"}
if not required_sales.issubset(sales.columns) or not required_inv.issubset(inventory.columns) or not required_products.issubset(products.columns):
    st.error("One or more CSV files are missing required columns. Use the supplied sample files as the schema.")
    st.stop()

sales = sales.sort_values(["product_id","date"]).copy()
sales["units_sold"] = sales["units_sold"].clip(lower=0)
sales[["selling_price","discount_pct"]] = sales.groupby("product_id")[["selling_price","discount_pct"]].transform(lambda x: x.fillna(x.median()))

FEATURES = ["lag_1","lag_7","lag_14","rolling_mean_7","rolling_mean_14",
            "day_of_week","day_of_month","month","selling_price","discount_pct"]

def make_features(frame):
    parts = []
    for pid, g in frame.groupby("product_id"):
        g = g.sort_values("date").copy()
        g["lag_1"] = g["units_sold"].shift(1)
        g["lag_7"] = g["units_sold"].shift(7)
        g["lag_14"] = g["units_sold"].shift(14)
        g["rolling_mean_7"] = g["units_sold"].shift(1).rolling(7).mean()
        g["rolling_mean_14"] = g["units_sold"].shift(1).rolling(14).mean()
        g["day_of_week"] = g["date"].dt.dayofweek
        g["day_of_month"] = g["date"].dt.day
        g["month"] = g["date"].dt.month
        parts.append(g)
    return pd.concat(parts, ignore_index=True).dropna(subset=FEATURES + ["units_sold"])

data = make_features(sales)
if len(data) < 50:
    st.error("Not enough history to train. Provide more daily sales records.")
    st.stop()

# Chronological holdout: the last 14 calendar days are used for testing.
cutoff = data["date"].max() - pd.Timedelta(days=13)
train = data[data["date"] < cutoff]
test = data[data["date"] >= cutoff]
if train.empty or test.empty:
    st.error("Not enough dates for a chronological train/test split.")
    st.stop()

model_choice = st.sidebar.selectbox(
    "Forecasting model",
    ["XGBoost Regressor", "Random Forest Regressor"] if XGBOOST_AVAILABLE else ["Random Forest Regressor"]
)
if model_choice == "XGBoost Regressor":
    model = XGBRegressor(n_estimators=350, max_depth=5, learning_rate=0.04,
                         subsample=0.9, colsample_bytree=0.9,
                         objective="reg:squarederror", random_state=42)
else:
    model = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                  random_state=42, n_jobs=-1)

model.fit(train[FEATURES], train["units_sold"])
test_pred = np.maximum(0, model.predict(test[FEATURES]))
mae = mean_absolute_error(test["units_sold"], test_pred)
rmse = np.sqrt(mean_squared_error(test["units_sold"], test_pred))
wape = np.abs(test["units_sold"].to_numpy() - test_pred).sum() / max(test["units_sold"].sum(), 1) * 100

# Recursive forecasts: each predicted day becomes history for the next forecast day.
horizon = st.sidebar.selectbox("Forecast horizon (days)", [7, 15, 30], index=0)
forecast_rows = []
for pid, g in sales.groupby("product_id"):
    g = g.sort_values("date").copy()
    history = list(g["units_sold"].astype(float))
    last = g.iloc[-1]
    price = float(g["selling_price"].tail(14).median())
    discount = float(g["discount_pct"].tail(14).mean())
    last_date = g["date"].max()
    for step in range(1, horizon + 1):
        future_date = last_date + pd.Timedelta(days=step)
        features = pd.DataFrame([{
            "lag_1": history[-1],
            "lag_7": history[-7] if len(history) >= 7 else np.mean(history),
            "lag_14": history[-14] if len(history) >= 14 else np.mean(history),
            "rolling_mean_7": np.mean(history[-7:]),
            "rolling_mean_14": np.mean(history[-14:]),
            "day_of_week": future_date.dayofweek,
            "day_of_month": future_date.day,
            "month": future_date.month,
            "selling_price": price,
            "discount_pct": discount
        }])
        prediction = max(0, float(model.predict(features[FEATURES])[0]))
        forecast_rows.append((pid, future_date, prediction))
        history.append(prediction)

forecast = pd.DataFrame(forecast_rows, columns=["product_id","date","predicted_demand"])
summary = forecast.groupby("product_id", as_index=False).agg(
    forecast_demand=("predicted_demand","sum"),
    avg_daily_forecast=("predicted_demand","mean")
)

# Forecast demand during lead time separately for each product.
lead_rows = []
for row in inventory.itertuples(index=False):
    product_forecast = forecast[forecast["product_id"] == row.product_id].sort_values("date")
    lead_demand = product_forecast.head(int(row.lead_time_days))["predicted_demand"].sum()
    cover_demand = product_forecast.head(min(horizon, int(row.lead_time_days) + 14))["predicted_demand"].sum()
    available = float(row.current_stock + row.on_order_units)
    reorder_point = lead_demand + float(row.safety_stock_units)
    if available <= lead_demand:
        risk = "High"
    elif available <= reorder_point:
        risk = "Medium"
    else:
        risk = "Low"
    excess_threshold = max(1, float(product_forecast["predicted_demand"].sum()) * 1.5)
    excess = available > excess_threshold
    order_qty = max(0, int(np.ceil(cover_demand + row.safety_stock_units - available)))
    if excess:
        status = "Excess inventory"
        order_qty = 0
    elif risk == "High":
        status = "Urgent replenishment"
    elif risk == "Medium":
        status = "Replenish soon"
    else:
        status = "Sufficient stock"
    lead_rows.append((row.product_id, lead_demand, reorder_point, risk, excess, order_qty, status))
risk_df = pd.DataFrame(lead_rows, columns=["product_id","lead_time_demand","reorder_point","risk","excess","recommended_order_qty","recommendation"])
result = products.merge(inventory, on="product_id").merge(summary, on="product_id").merge(risk_df, on="product_id")
result = result.sort_values(["risk","recommended_order_qty"], ascending=[True, False])

# Sidebar controls
st.sidebar.markdown("---")
st.sidebar.write("**Data files**")
st.sidebar.caption("sales_history.csv · inventory.csv · products.csv")

# KPI cards
c1,c2,c3,c4 = st.columns(4)
c1.metric("Products monitored", len(result))
c2.metric("High stock-out risk", int((result["risk"] == "High").sum()))
c3.metric("Excess inventory", int(result["excess"].sum()))
c4.metric("Products to reorder", int((result["recommended_order_qty"] > 0).sum()))

tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Demand Forecast", "Replenishment", "Model Evaluation"])

with tab1:
    st.subheader("Inventory risk overview")
    chart_data = result[["product_name","current_stock","forecast_demand"]].melt(
        id_vars="product_name", var_name="measure", value_name="units")
    st.plotly_chart(px.bar(chart_data, x="product_name", y="units", color="measure",
                           barmode="group", title="Current stock vs forecast demand"),
                    use_container_width=True)
    st.dataframe(result[["product_name","category","current_stock","forecast_demand",
                         "risk","recommendation"]], use_container_width=True, hide_index=True)

with tab2:
    st.subheader(f"Predicted demand for the next {horizon} days")
    chosen = st.selectbox("Choose product", products["product_name"].tolist())
    chosen_id = products.loc[products["product_name"] == chosen, "product_id"].iloc[0]
    plot_df = forecast[forecast["product_id"] == chosen_id]
    st.plotly_chart(px.line(plot_df, x="date", y="predicted_demand", markers=True,
                            title=f"{chosen}: predicted daily demand"),
                    use_container_width=True)
    st.metric("Total predicted units", f"{plot_df['predicted_demand'].sum():.0f}")

with tab3:
    st.subheader("AI-assisted replenishment recommendations")
    st.caption("Order quantities are estimates based on forecast demand, lead time, safety stock and outstanding orders.")
    display = result[["product_name","supplier","current_stock","on_order_units",
                      "lead_time_days","lead_time_demand","reorder_point","risk",
                      "recommended_order_qty","recommendation"]].copy()
    st.dataframe(display, use_container_width=True, hide_index=True)
    urgent = result[result["risk"] == "High"]
    if not urgent.empty:
        st.error("Urgent attention: " + ", ".join(urgent["product_name"].tolist()))
    st.download_button("Download recommendations CSV",
                       result.to_csv(index=False).encode("utf-8"),
                       "replenishment_recommendations.csv", "text/csv")

with tab4:
    st.subheader("Chronological holdout evaluation")
    m1,m2,m3 = st.columns(3)
    m1.metric("MAE", f"{mae:.2f} units")
    m2.metric("RMSE", f"{rmse:.2f} units")
    m3.metric("WAPE", f"{wape:.1f}%")
    eval_df = test[["date","product_id","units_sold"]].copy()
    eval_df["predicted_units"] = test_pred
    st.write("Actual vs predicted demand on the final 14 days in the sample dataset.")
    st.dataframe(eval_df.merge(products[["product_id","product_name"]], on="product_id"),
                 use_container_width=True, hide_index=True)
    st.caption("This is a sample backtest. For a fair production evaluation, use multiple rolling time-based test windows and compare against simple baselines.")

st.markdown("---")
st.caption("Prototype only: sample data is synthetic. Validate forecasts, supplier constraints, pack sizes and business rules before placing real orders.")
