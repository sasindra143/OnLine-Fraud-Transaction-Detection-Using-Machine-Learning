
# 🛡️ Online Fraud Transaction Fraud Detection using Machine Learning  

This project implements a **machine learning model** to detect fraudulent online transactions.  
It provides a **Streamlit web interface** where users can input transaction details, and the model predicts whether the transaction is **legitimate** or **fraudulent**.  

Fraud detection is essential for banks, e-commerce, and financial institutions to minimize losses and improve security.  

---

## 📊 Dataset  

The dataset used contains transaction-level information with both legitimate and fraudulent transactions.  
Key features include:  
- Transaction Amount  
- Quantity  
- Customer Age  
- Account Age (Days)  
- Transaction Hour  
- Payment Method (encoded)  
- Product Category (encoded)  
- Device Used (encoded)  

⚠️ **Note:** Due to privacy concerns, the dataset is not included here. You can use any public fraud detection dataset (e.g., [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)) or a synthetic dataset.  

---

## 🚀 Features  

- Data preprocessing & handling missing values.  
- Class imbalance handled using **SMOTE**.  
- Machine learning models implemented:  
  - Logistic Regression  
  - Random Forest  
  - XGBoost  
- Model evaluation using Accuracy, Precision, Recall, F1-score, ROC-AUC.  
- Model persistence using **Joblib**.  
- Streamlit-based **web application** for real-time fraud detection.  

---

## 🖼️ Screenshots  

### 🔹 Input Form  
![Input Form](./screenshots/input_form.png)  

### 🔹 Prediction Result  
![Prediction Output](./screenshots/output_prediction.png)  

---

## ⚙️ Installation & Setup  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/your-username/fraud-detection-ml.git
cd fraud-detection-ml
````

### 2️⃣ Install Dependencies

Make sure you have **Python 3.10.0** installed. Then run:

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Jupyter Notebooks (Optional)

For data analysis and training models:

```bash
jupyter notebook
```

### 4️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

### 5️⃣ Access the App

Open your browser at:

```
http://localhost:8501
```

---

## 📁 Repository Structure

```
├── data/                # Dataset (link or CSV files)
├── notebooks/           # Jupyter notebooks (EDA, training, evaluation)
├── src/                 # Source code for preprocessing & ML models
├── models/              # Saved models (.pkl/.joblib)
├── results/             # Model evaluation reports & graphs
├── screenshots/         # UI and output screenshots
├── app.py               # Streamlit web app
├── requirements.txt     # Dependencies
└── README.md            # Project documentation
```

---

## 🛠️ Tech Stack

* **Programming Language:** Python 3.10
* **Libraries:**

  * Numpy, Pandas
  * Matplotlib, Seaborn
  * Scikit-learn
  * XGBoost
  * Joblib
  * Streamlit

---

## 🔮 Future Scope

* Implement **Deep Learning models** (LSTM, Autoencoders) for anomaly detection.
* Deploy the app on **Heroku, AWS, or GCP** for real-world use.
* Connect with real-time transaction APIs for live fraud monitoring.
* Build a REST API with **Flask/FastAPI** for integration with banking systems.

---

## 🙌 Acknowledgments

* [Kaggle Datasets](https://www.kaggle.com/) for providing fraud detection datasets.
* Open-source contributors of **Scikit-learn, Streamlit, and XGBoost**.

---

## 📌 Author

👨‍💻 Developed by ** Madduri Sasindra**
📧 Contact: 9959732476
🔗 GitHub: sasindra143

---

```

---

✅ This README is **ready-to-use** — just replace:  
- `your-username` with your GitHub username  
- `Your Name` and contact details  
- Update the **screenshots folder** with your own images  

Do you also want me to create a **LICENSE file (MIT)** so your project looks more professional on GitHub?
```
