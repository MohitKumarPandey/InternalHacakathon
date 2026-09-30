import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Inventory Demand Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    background: linear-gradient(135deg, #162033, #111827);
    padding: 30px;
    border-radius: 0px 0px 22px 22px;
    margin-bottom: 30px;
}

.hero h1 {
    font-size: 38px;
    color: white;
    margin-bottom: 8px;
}

.hero p {
    font-size: 17px;
    color: #d1d5db;
}

.metric-card {
    background: #171b24;
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #2b3240;
}

.metric-title {
    color: #9ca3af;
    font-size: 14px;
}

.metric-value {
    color: white;
    font-size: 30px;
    font-weight: 700;
}

.success-box {
    background: #12351f;
    border: 1px solid #2f855a;
    padding: 18px;
    border-radius: 12px;
    color: #d1fae5;
}

.warning-box {
    background: #3b3210;
    border: 1px solid #b7791f;
    padding: 18px;
    border-radius: 12px;
    color: #fef3c7;
}

.danger-box {
    background: #3b1414;
    border: 1px solid #c53030;
    padding: 18px;
    border-radius: 12px;
    color: #fee2e2;
}

.info-box {
    background: #122b45;
    border: 1px solid #2563eb;
    padding: 18px;
    border-radius: 12px;
    color: #dbeafe;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# FILE NAMES
# =========================================================

DATASET_PATH = "sales_data.csv"
LINEAR_PATH = "linearModel.pkl"
XGB_PATH = "xgboost.pkl"



@st.cache_data
def load_dataset():

    if not os.path.exists(DATASET_PATH):
        return None

    try:

        data = pd.read_csv(DATASET_PATH)

        # -------------------------------------------------
        # SAME DATE FEATURES USED DURING TRAINING
        # -------------------------------------------------
        if "Date" in data.columns:

            data["Date"] = pd.to_datetime(
                data["Date"],
                errors="coerce"
            )

            data["Year"] = data["Date"].dt.year
            data["Month"] = data["Date"].dt.month
            data["Day"] = data["Date"].dt.day
            data["DayOfWeek"] = data["Date"].dt.dayofweek

            
            data.drop(
                columns=["Date"],
                inplace=True
            )

        return data

    except Exception as e:

        st.error(f"Dataset loading error: {e}")
        return None


default_df = load_dataset()

# Uploaded CSV takes priority over the default sales_data.csv
if st.session_state.get("uploaded_df") is not None:
    df = st.session_state.uploaded_df.copy()

    # Apply the same date preprocessing used by the default dataset
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Day"] = df["Date"].dt.day
        df["DayOfWeek"] = df["Date"].dt.dayofweek

        df.drop(
            columns=["Date"],
            inplace=True
        )
else:
    df = default_df

@st.cache_resource
def load_model(path):

    if not os.path.exists(path):
        return None

    try:
        return joblib.load(path)

    except Exception:
        return None


linear_model = load_model(LINEAR_PATH)
xgb_model = load_model(XGB_PATH)


models = {
    "Linear Regression": linear_model,
    "XGBoost": xgb_model
}


available_models = {
    name: model
    for name, model in models.items()
    if model is not None
}


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📊 Dashboard")

section = st.sidebar.radio(
    "Select Section",
    [
        "🏠 Overview",
        "🔮 Demand Prediction",
        "📈 Model Comparison",
        "📊 Data Analytics"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **PS-08: Inventory Demand Prediction**

    AI-powered demand forecasting and
    inventory risk monitoring system.
    """
)

# =========================================================
# CSV UPLOAD
# =========================================================

st.sidebar.markdown("---")
st.sidebar.subheader("📂 Upload Your CSV")

uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV dataset",
    type=["csv"],
    help="Upload a CSV with columns compatible with the trained model."
)

if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None

if uploaded_file is not None:
    try:
        uploaded_data = pd.read_csv(uploaded_file)

        if uploaded_data.empty:
            st.sidebar.error("❌ The uploaded CSV is empty.")
        else:
            st.session_state.uploaded_df = uploaded_data
            st.sidebar.success(
                f"✅ CSV loaded: {len(uploaded_data):,} rows"
            )

            st.sidebar.caption(
                f"Columns detected: {len(uploaded_data.columns)}"
            )

    except Exception as e:
        st.sidebar.error(f"❌ CSV upload failed: {e}")

if st.session_state.uploaded_df is not None:
    if st.sidebar.button(
        "↩️ Use Default Dataset",
        use_container_width=True
    ):
        st.session_state.uploaded_df = None
        st.rerun()


# =========================================================
# HERO
# =========================================================

st.markdown("""
<div class="hero">

<h1>📦 Inventory Demand Intelligence</h1>

<p>
AI-powered demand prediction and inventory risk monitoring
</p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def prepare_dataset(data):

    data = data.copy()

    if "Date" in data.columns:

        data["Date"] = pd.to_datetime(
            data["Date"],
            errors="coerce"
        )

        data["Year"] = data["Date"].dt.year
        data["Month"] = data["Date"].dt.month
        data["Day"] = data["Date"].dt.day
        data["DayOfWeek"] = data["Date"].dt.dayofweek

        data.drop(
            columns=["Date"],
            inplace=True
        )

    return data


def safe_predict(model, input_data):

    try:

        prediction = model.predict(input_data)

        return float(prediction[0])

    except Exception:

        return None


def calculate_metrics(model, X_test, y_test):

    try:

        predictions = model.predict(X_test)

        r2 = r2_score(
            y_test,
            predictions
        )

        mse = mean_squared_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(mse)

        return r2, mse, rmse

    except Exception:

        return None, None, None


# =========================================================
# OVERVIEW
# =========================================================

if section == "🏠 Overview":

    st.subheader("Dashboard Overview")

    record_count = len(df) if df is not None else 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-title">
            Models Available
            </div>

            <div class="metric-value">
            {len(available_models)}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-title">
            Historical Records
            </div>

            <div class="metric-value">
            {record_count:,}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="metric-card">

            <div class="metric-title">
            Prediction Type
            </div>

            <div class="metric-value">
            Demand
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:

        st.markdown(
            """
            <div class="metric-card">

            <div class="metric-title">
            Inventory Monitoring
            </div>

            <div class="metric-value">
            Active
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # =====================================================
    # MODEL STATUS
    # =====================================================

    st.markdown("---")

    st.subheader("🤖 Model Status")

    status_rows = []

    for name, model in models.items():

        status_rows.append({
            "Model": name,
            "Status": (
                "🟢 Loaded"
                if model is not None
                else "🔴 Not Found"
            )
        })

    st.dataframe(
        pd.DataFrame(status_rows),
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # DATASET STATUS
    # =====================================================

    st.subheader("📁 Dataset Status")

    if df is not None:

        if st.session_state.get("uploaded_df") is not None:
            dataset_name = uploaded_file.name if uploaded_file is not None else "Uploaded CSV"
            dataset_message = f"✅ <b>{dataset_name} uploaded successfully</b>"
        else:
            dataset_message = "✅ <b>sales_data.csv loaded successfully</b>"

        st.markdown(
            f"""
            <div class="success-box">

            {dataset_message}

            <br><br>

            Historical Records:
            <b>{len(df):,}</b>

            <br>

            Dataset Columns:
            <b>{len(df.columns)}</b>

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="danger-box">

            ❌ <b>sales_data.csv not found</b>

            <br><br>

            Place sales_data.csv in the same folder
            as app.py.

            </div>
            """,
            unsafe_allow_html=True
        )


    # =====================================================
    # QUICK SUMMARY
    # =====================================================

    if df is not None:

        st.subheader("📌 Dataset Summary")

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            if "Demand" in df.columns:

                st.metric(
                    "Average Demand",
                    f"{df['Demand'].mean():.2f}"
                )

        with c2:

            if "Units Sold" in df.columns:

                st.metric(
                    "Total Units Sold",
                    f"{df['Units Sold'].sum():,.0f}"
                )

        with c3:

            if "Inventory Level" in df.columns:

                st.metric(
                    "Average Inventory",
                    f"{df['Inventory Level'].mean():.2f}"
                )

        with c4:

            if "Units Ordered" in df.columns:

                st.metric(
                    "Total Units Ordered",
                    f"{df['Units Ordered'].sum():,.0f}"
                )


# =========================================================
# DEMAND PREDICTION
# =========================================================

elif section == "🔮 Demand Prediction":

    st.subheader("🔮 Demand Prediction")

    if len(available_models) == 0:

        st.error(
            "No trained model found."
        )

        st.stop()


    st.write(
        "Enter product, inventory and market information "
        "to predict demand."
    )

    if st.session_state.get("uploaded_df") is not None:
        st.info(
            "📂 Using your uploaded CSV for dataset-driven dropdown values. "
            "Prediction still requires the uploaded columns to match the features "
            "used when the trained models were created."
        )


    # =====================================================
    # PRODUCT INFORMATION
    # =====================================================

    st.subheader("🏪 Product Information")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        store_id = st.text_input(
            "Store ID",
            value="S001"
        )

    with col2:

        product_id = st.text_input(
            "Product ID",
            value="P0001"
        )

    with col3:

        if df is not None and "Category" in df.columns:

            category_values = sorted(
                df["Category"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

        else:

            category_values = [
                "Electronics",
                "Clothing",
                "Groceries",
                "Toys",
                "Furniture"
            ]

        category = st.selectbox(
            "Category",
            category_values
        )

    with col4:

        if df is not None and "Region" in df.columns:

            region_values = sorted(
                df["Region"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

        else:

            region_values = [
                "North",
                "South",
                "East",
                "West"
            ]

        region = st.selectbox(
            "Region",
            region_values
        )


    # =====================================================
    # INVENTORY
    # =====================================================

    st.subheader("📦 Inventory Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        inventory = st.number_input(
            "Inventory Level",
            min_value=0.0,
            value=100.0,
            step=1.0
        )

    with col2:

        units_sold = st.number_input(
            "Units Sold",
            min_value=0.0,
            value=50.0,
            step=1.0
        )

    with col3:

        units_ordered = st.number_input(
            "Units Ordered",
            min_value=0.0,
            value=100.0,
            step=1.0
        )


    # =====================================================
    # PRICE
    # =====================================================

    st.subheader("💰 Pricing Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        price = st.number_input(
            "Price",
            min_value=0.0,
            value=50.0,
            step=0.01
        )

    with col2:

        discount = st.number_input(
            "Discount",
            min_value=0.0,
            value=5.0,
            step=1.0
        )

    with col3:

        competitor_pricing = st.number_input(
            "Competitor Pricing",
            min_value=0.0,
            value=50.0,
            step=0.01
        )


    # =====================================================
    # MARKET CONDITIONS
    # =====================================================

    st.subheader("🌦️ Market Conditions")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        if df is not None and "Weather Condition" in df.columns:

            weather_values = sorted(
                df["Weather Condition"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

        else:

            weather_values = [
                "Sunny",
                "Rainy",
                "Cloudy",
                "Snowy"
            ]

        weather = st.selectbox(
            "Weather Condition",
            weather_values
        )

    with col2:

        promotion = st.selectbox(
            "Promotion",
            [0, 1]
        )

    with col3:

        if df is not None and "Seasonality" in df.columns:

            season_values = sorted(
                df["Seasonality"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

        else:

            season_values = [
                "Spring",
                "Summer",
                "Autumn",
                "Winter"
            ]

        seasonality = st.selectbox(
            "Seasonality",
            season_values
        )

    with col4:

        epidemic = st.selectbox(
            "Epidemic",
            [0, 1]
        )


    # =====================================================
    # DATE
    # =====================================================

    st.subheader("📅 Prediction Date")

    prediction_date = st.date_input(
        "Select Date"
    )


    # =====================================================
    # PREDICT
    # =====================================================

    st.markdown("")

    predict_button = st.button(
        "🚀 Predict Demand",
        type="primary",
        use_container_width=True
    )


    if predict_button:

        date_value = pd.Timestamp(
            prediction_date
        )

        required_prediction_columns = [
            "Store ID",
            "Product ID",
            "Category",
            "Region",
            "Inventory Level",
            "Units Sold",
            "Units Ordered",
            "Price",
            "Discount",
            "Weather Condition",
            "Promotion",
            "Competitor Pricing",
            "Seasonality",
            "Epidemic"
        ]

        if st.session_state.get("uploaded_df") is not None:
            uploaded_columns = set(
                st.session_state.uploaded_df.columns
            )

            missing_columns = [
                col for col in required_prediction_columns
                if col not in uploaded_columns
            ]

            if missing_columns:
                st.warning(
                    "⚠️ The uploaded CSV is missing columns used by the "
                    "prediction form: "
                    + ", ".join(missing_columns)
                    + ". You can still use the CSV for analytics, but "
                      "prediction must use the trained model's expected features."
                )


        # -------------------------------------------------
        # EXACT DATE FEATURES USED IN TRAINING
        # -------------------------------------------------

        year = date_value.year
        month = date_value.month
        day = date_value.day
        day_of_week = date_value.dayofweek


        # -------------------------------------------------
        # INPUT DATA
        #
        # IMPORTANT:
        # No Quarter
        # No IsWeekend
        # -------------------------------------------------

        input_data = pd.DataFrame([{

            "Store ID": store_id,
            "Product ID": product_id,

            "Category": category,
            "Region": region,

            "Inventory Level": inventory,
            "Units Sold": units_sold,
            "Units Ordered": units_ordered,

            "Price": price,
            "Discount": discount,

            "Weather Condition": weather,
            "Promotion": promotion,

            "Competitor Pricing": competitor_pricing,

            "Seasonality": seasonality,
            "Epidemic": epidemic,

            "Year": year,
            "Month": month,
            "Day": day,
            "DayOfWeek": day_of_week

        }])


        # -------------------------------------------------
        # PREDICTIONS
        # -------------------------------------------------

        predictions = {}

        for name, model in available_models.items():

            prediction = safe_predict(
                model,
                input_data
            )

            if prediction is not None:

                predictions[name] = max(
                    0,
                    prediction
                )


        if len(predictions) == 0:

            st.error(
                """
                Prediction failed.

                The saved model may have been trained
                with a different feature structure.
                """
            )

        else:

            st.subheader("📊 Predicted Demand")

            result_cols = st.columns(
                len(predictions)
            )


            for index, (name, prediction) in enumerate(
                predictions.items()
            ):

                with result_cols[index]:

                    st.metric(
                        name,
                        f"{prediction:.2f} units"
                    )


            # -------------------------------------------------
            # USE XGBOOST AS PRIMARY MODEL
            # -------------------------------------------------

            if "XGBoost" in predictions:

                final_prediction = predictions[
                    "XGBoost"
                ]

                primary_model = "XGBoost"

            else:

                primary_model = list(
                    predictions.keys()
                )[0]

                final_prediction = predictions[
                    primary_model
                ]


            # -------------------------------------------------
            # INVENTORY ANALYSIS
            # -------------------------------------------------

            st.markdown("---")

            st.subheader(
                "📦 Inventory Risk Analysis"
            )


            inventory_gap = (
                inventory - final_prediction
            )


            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "Current Inventory",
                    f"{inventory:.0f}"
                )

            with c2:

                st.metric(
                    "Predicted Demand",
                    f"{final_prediction:.2f}"
                )

            with c3:

                st.metric(
                    "Inventory Gap",
                    f"{inventory_gap:.2f}"
                )


            # -------------------------------------------------
            # STOCK ALERT
            # -------------------------------------------------

            if final_prediction > inventory:

                shortage = (
                    final_prediction - inventory
                )

                st.markdown(
                    f"""
                    <div class="danger-box">

                    🚨 <b>STOCK-OUT RISK</b>

                    <br><br>

                    Predicted demand is greater than
                    available inventory.

                    <br><br>

                    Estimated shortage:
                    <b>{shortage:.2f} units</b>

                    <br><br>

                    Recommended action:
                    <b>Replenish inventory</b>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            elif final_prediction > inventory * 0.8:

                st.markdown(
                    """
                    <div class="warning-box">

                    ⚠️ <b>HIGH INVENTORY RISK</b>

                    <br><br>

                    Predicted demand is approaching
                    the available inventory.

                    Consider replenishing stock.

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            else:

                st.markdown(
                    """
                    <div class="success-box">

                    ✅ <b>INVENTORY SUFFICIENT</b>

                    <br><br>

                    Current inventory is sufficient
                    for the predicted demand.

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            st.info(
                f"Primary prediction model: {primary_model}"
            )


# =========================================================
# MODEL COMPARISON
# =========================================================

elif section == "📈 Model Comparison":

    st.subheader(
        "📈 Model Performance Comparison"
    )

    st.write(
        "Comparison of Linear Regression, "
        "Random Forest and XGBoost."
    )


    if df is None:

        st.error(
            "No dataset is available. Upload a CSV or place sales_data.csv in the app folder."
        )

        st.stop()


    if "Demand" not in df.columns:

        st.error(
            "Demand column not found in dataset."
        )

        st.stop()
    comparison_df = prepare_dataset(df)


    X = comparison_df.drop(
        columns=["Demand"],
        errors="ignore"
    )

    y = comparison_df["Demand"]


   
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )


  
    comparison_results = []


    for name, model in models.items():

        if model is None:
            continue


        r2, mse, rmse = calculate_metrics(
            model,
            X_test,
            y_test
        )


        if r2 is not None:

            comparison_results.append({

                "Model": name,
                "R² Score": r2,
                "MSE": mse,
                "RMSE": rmse

            })


    if not comparison_results:

        st.error(
            """
            Could not calculate model metrics.

            Please check that the saved models were
            trained using the same features as the dataset.
            """
        )

        st.stop()


    results_df = pd.DataFrame(
        comparison_results
    )


   

    st.dataframe(
        results_df.style.format({
            "R² Score": "{:.4f}",
            "MSE": "{:.2f}",
            "RMSE": "{:.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )


    # -----------------------------------------------------
    # R2
    # -----------------------------------------------------

    st.subheader(
        "R² Score Comparison"
    )

    fig_r2 = px.bar(
        results_df,
        x="Model",
        y="R² Score",
        text="R² Score",
        title="Model R² Performance"
    )

    fig_r2.update_traces(
        texttemplate="%{text:.4f}",
        textposition="outside"
    )

    fig_r2.update_layout(
        yaxis_title="R² Score",
        xaxis_title="Model"
    )

    st.plotly_chart(
        fig_r2,
        use_container_width=True
    )


    # -----------------------------------------------------
    # MSE
    # -----------------------------------------------------

    st.subheader(
        "Mean Squared Error"
    )

    fig_mse = px.bar(
        results_df,
        x="Model",
        y="MSE",
        text="MSE",
        title="MSE Comparison"
    )

    fig_mse.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside"
    )

    st.plotly_chart(
        fig_mse,
        use_container_width=True
    )


   

    st.subheader(
        "Root Mean Squared Error"
    )

    fig_rmse = px.bar(
        results_df,
        x="Model",
        y="RMSE",
        text="RMSE",
        title="RMSE Comparison"
    )

    fig_rmse.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside"
    )

    st.plotly_chart(
        fig_rmse,
        use_container_width=True
    )


    st.info(
        """
        R² measures how much variation in demand is
        explained by the model.

        MSE and RMSE measure prediction error.

        Higher R² and lower MSE/RMSE indicate better
        predictive performance.
        """
    )




elif section == "📊 Data Analytics":

    st.subheader(
        "📊 Inventory & Demand Analytics"
    )


    if df is None:

        st.error(
            "No dataset is available. Upload a CSV or place sales_data.csv in the app folder."
        )

        st.stop()

    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(
            "Total Records",
            f"{len(df):,}"
        )


    with c2:

        if "Demand" in df.columns:

            st.metric(
                "Average Demand",
                f"{df['Demand'].mean():.2f}"
            )


    with c3:

        if "Inventory Level" in df.columns:

            st.metric(
                "Average Inventory",
                f"{df['Inventory Level'].mean():.2f}"
            )


    with c4:

        if "Units Sold" in df.columns:

            st.metric(
                "Total Units Sold",
                f"{df['Units Sold'].sum():,.0f}"
            )




    if "Demand" in df.columns:

        st.subheader(
            "📈 Demand Distribution"
        )

        fig = px.histogram(
            df,
            x="Demand",
            nbins=40,
            title="Demand Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # =====================================================
    # CATEGORY ANALYSIS
    # =====================================================

    if (
        "Category" in df.columns
        and "Demand" in df.columns
    ):

        st.subheader(
            "🏷️ Average Demand by Category"
        )

        category_data = (
            df.groupby("Category")["Demand"]
            .mean()
            .reset_index()
            .sort_values(
                "Demand",
                ascending=False
            )
        )

        fig = px.bar(
            category_data,
            x="Category",
            y="Demand",
            text="Demand",
            title="Average Demand by Category"
        )

        fig.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


  

    if (
        "Region" in df.columns
        and "Demand" in df.columns
    ):

        st.subheader(
            "🌍 Average Demand by Region"
        )

        region_data = (
            df.groupby("Region")["Demand"]
            .mean()
            .reset_index()
            .sort_values(
                "Demand",
                ascending=False
            )
        )

        fig = px.bar(
            region_data,
            x="Region",
            y="Demand",
            text="Demand",
            title="Average Demand by Region"
        )

        fig.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


  

    if (
        "Inventory Level" in df.columns
        and "Demand" in df.columns
    ):

        st.subheader(
            "📦 Inventory Level vs Demand"
        )

        chart_data = df.copy()

        if len(chart_data) > 5000:

            chart_data = chart_data.sample(
                5000,
                random_state=42
            )

        fig = px.scatter(
            chart_data,
            x="Inventory Level",
            y="Demand",
            title="Inventory Level vs Demand",
            opacity=0.6
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )
    if (
        "Units Sold" in df.columns
        and "Demand" in df.columns
    ):

        st.subheader(
            "🛒 Units Sold vs Demand"
        )

        chart_data = df.copy()

        if len(chart_data) > 5000:

            chart_data = chart_data.sample(
                5000,
                random_state=42
            )

        fig = px.scatter(
            chart_data,
            x="Units Sold",
            y="Demand",
            title="Units Sold vs Demand",
            opacity=0.6
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # =====================================================
    # RAW DATA
    # =====================================================

    st.subheader(
        "📋 Dataset Preview"
    )

    st.dataframe(
        df.head(100),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "PS-08 • Inventory Demand Prediction • "
    "AI/ML Internal Hackathon"
)