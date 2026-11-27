# InventoryAI starter

## Files
- sales_history.csv: 210 days of sample daily sales for 12 products.
- inventory.csv: current stock, supplier lead time, safety stock and outstanding orders.
- products.csv: product names, categories, suppliers and unit costs.
- app.py: Streamlit application with demand forecasting, risk analysis, replenishment recommendations and a chronological holdout evaluation.
- requirements.txt: Python dependencies.

## Run
1. Extract this folder.
2. Open a terminal in this folder.
3. Install dependencies: `pip install -r requirements.txt`
4. Start the app: `streamlit run app.py`

The sample data is synthetic and intended for a prototype/demo only. XGBoost is used when installed; otherwise the app falls back to Random Forest.
