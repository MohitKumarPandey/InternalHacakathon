# 📦 Inventory Demand Prediction

An **Machine Learning-based Inventory Demand Prediction** system that analyzes historical sales data and predicts future product demand using **Linear Regression** and **XGBoost**.

The project provides an interactive **Streamlit dashboard** where users can explore sales data, compare machine learning models, and upload their own CSV dataset for analysis and prediction.

## 🚀 Features

* 📊 Historical sales data analysis
* 🤖 Machine Learning-based demand prediction
* 📈 Linear Regression model
* ⚡ XGBoost model
* 🔄 Model comparison
* 📁 Custom CSV dataset upload
* 🔍 Product, category and location-based analysis
* 📉 Interactive charts and visualizations
* 🖥️ Streamlit-based user interface
* 💾 Pre-trained model support

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **XGBoost**
* **Joblib**
* **Streamlit**
* **Matplotlib / Plotly**

## 🧠 Machine Learning Models

### 1. Linear Regression

Linear Regression is used as a baseline regression model to estimate inventory demand based on the available input features.

### 2. XGBoost

XGBoost is used as the main machine learning model for demand prediction. It is a gradient-boosting algorithm that builds multiple decision trees sequentially to improve prediction performance.

## 📂 Project Structure

```text
InventoryDemandPrediction/
│
├── app.py
├── sales_data.csv
├── linearModel.pkl
├── xgboost.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

## 📊 Dataset

The project uses historical sales/inventory data containing information required for demand analysis and prediction.

The application also provides a **CSV upload option**, allowing users to analyze their own compatible dataset instead of using only the default dataset.

## 📁 CSV Upload

Users can upload a CSV file directly from the Streamlit sidebar.

The uploaded dataset is used for:

* Data analysis
* Product/category exploration
* Visualizations
* Demand prediction

If no custom file is uploaded, the application uses the default `sales_data.csv` dataset.

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/MohitKumarPandey/InventoryDemandPrediction.git
```

Navigate to the project directory:

```bash
cd InventoryDemandPrediction
```

Create a virtual environment:

```bash
python -m venv myenv
```

Activate the environment on Windows:

```bash
myenv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

## 🔄 Workflow

```text
Historical Sales Data
        ↓
Data Loading
        ↓
Data Preprocessing
        ↓
Feature Selection
        ↓
Machine Learning Models
        ↓
Linear Regression + XGBoost
        ↓
Demand Prediction
        ↓
Interactive Streamlit Dashboard
```

## 🎯 Project Objective

The objective of this project is to use historical sales information and machine learning techniques to estimate product demand.

Demand prediction can help businesses understand sales patterns and support inventory planning and stock management.

## 🔮 Future Scope

* Real-time inventory data integration
* Advanced time-series forecasting
* Automated stock-level recommendations
* Low-stock alerts
* Seasonal demand analysis
* Integration with business inventory systems
* More advanced forecasting models
* Cloud-based deployment

## 👨‍💻 Author

**Mohit Kumar Pandey**

GitHub:
https://github.com/MohitKumarPandey

## 📄 License

This project is developed for educational and project demonstration purposes.
