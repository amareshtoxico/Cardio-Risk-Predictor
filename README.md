# ❤️ Cardio risk Predictor – Machine Learning Web Application
[![Live Demo](https://img.shields.io/badge/Demo-Live%20App-success?style=for-the-badge&logo=render)](https://cardio-risk-predictor.onrender.com)

A complete **end-to-end Machine Learning project** that predicts the likelihood of heart disease based on patient health parameters.  
This project covers the **entire ML lifecycle** — from data preprocessing and feature engineering to model training, evaluation, and deployment using Flask.

Designed as a **portfolio-grade data science & ML application** with clean code structure and real-world workflow.

---

## 🚀 Project Highlights

- 🧠 **Cardio-Risk-Predictor** using supervised machine learning  
- 🧹 **Missing Value Handling & Data Cleaning**  
- ⚖️ **Class Imbalance Handling**  
- 🔄 **Feature Transformation & Outlier Treatment**  
- 📊 **Model Training, Evaluation & Selection**  
- 💾 **Model & Scaler Serialization (Pickle)**  
- 🌐 **Flask Web Application** for real-time predictions  
- 🎨 Simple & clean UI for user interaction  

---

## 🧠 Machine Learning Pipeline

The project follows a **structured and industry-standard ML workflow**:

1. **Data Ingestion**
   - Dataset loaded from `heart.csv`

2. **Data Preprocessing**
   - Missing value handling
   - Data type correction
   - Feature scaling using `StandardScaler`

3. **Feature Engineering**
   - Variable transformations
   - Outlier detection & handling

4. **Class Imbalance Handling**
   - Balancing techniques applied to improve model fairness

5. **Model Training**
   - Multiple ML models trained and evaluated
   - Best-performing model selected

6. **Model Persistence**
   - Final model saved as `best_model.pkl`
   - Scaler saved as `scaler.pkl`

7. **Model Deployment**
   - Flask-based web application for predictions

---

## 🛠️ Tech Stack

### Programming & Frameworks
- Python
- Flask

### Machine Learning & Data Science
- NumPy
- Pandas
- Scikit-learn

### Visualization / UI
- HTML
- CSS

---

## 📂 Project Structure

```

Cardio-Risk-Predictor/
│
├── app.py                         # Flask application
├── main.py                        # Project execution flow
│
├── heart.csv                      # Dataset
│
├── missing_val_handle.py          # Missing value handling
├── variable_transformation_outlierhandle.py
│                                  # Feature transformation & outlier treatment
├── data_balance.py                # Class imbalance handling
│
├── model_training.py              # Model training & evaluation
├── log_code.py                    # Logging utilities
│
├── best_model.pkl                 # Trained ML model
├── scaler.pkl                     # Feature scaler
│
├── templates/
│   ├── index.html                 # Input form
│   └── result.html                # Prediction result page
│
├── static/
│   └── style.css                  # UI styling
│
└── README.md

````

---

## ▶️ How to Run the Project Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/amareshtoxico/Cardio-Risk-Predictor.git
cd heart-disease-prediction-ml
````

### 2️⃣ Create & Activate Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install flask numpy pandas scikit-learn
```

### 4️⃣ Run the Flask App

```bash
python app.py
```

### 5️⃣ Open in Browser

```
http://127.0.0.1:5000
```

---

## 🧪 Input Features

The model predicts heart disease based on clinical parameters such as:

* Age
* Sex
* Chest pain type
* Resting blood pressure
* Cholesterol level
* Fasting blood sugar
* Maximum heart rate achieved
* Exercise-induced angina
* ST depression and slope

*(Exact inputs depend on dataset features)*

---

## 🎯 Use Cases

* Healthcare risk prediction systems
* Medical decision support tools
* Machine Learning portfolio project
* Data science academic projects
* Flask + ML deployment reference

---

## 🔮 Future Enhancements

* Model explainability (SHAP / feature importance)
* Improved UI/UX design
* REST API support
* Cloud deployment (Render / Railway)
* User authentication & history tracking

---

## 👨‍💻 Author

**Amaresh Virupakshi**
Machine Learning & Python Developer

---

## ✅ Why This README Is Strong

✔ Shows **real ML depth** (not toy project)  
✔ Clearly explains preprocessing & feature engineering  
✔ Recruiter-friendly structure  
✔ Demonstrates end-to-end ownership  
✔ Matches **actual files in your project**  




