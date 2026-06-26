# 🌊 Rising Waters - Flood Prediction System

## 📌 Project Overview
Rising Waters is a Machine Learning + Flask web application that predicts whether a flood will occur based on environmental and rainfall parameters.

The system analyzes weather conditions like temperature, humidity, rainfall distribution, and seasonal data to classify:
- 🚨 Flood
- ✅ No Flood

---

## 🧠 Machine Learning Model
- Algorithm: Random Forest Classifier
- Preprocessing: StandardScaler
- Input Features: 10 weather parameters
- Output: Binary classification (0 = No Flood, 1 = Flood)

---

## ⚙️ Tech Stack
- Python 🐍
- Flask 🌐
- Scikit-learn 🤖
- Pandas 📊
- HTML, CSS 🎨
- Joblib (model saving)

---

## 📊 Input Features
1. Temp  
2. Humidity  
3. Cloud Cover  
4. ANNUAL Rainfall  
5. Jan-Feb Rainfall  
6. Mar-May Rainfall  
7. Jun-Sep Rainfall  
8. Oct-Dec Rainfall  
9. Avg June  
10. Sub value  

---

## 🚀 How It Works
1. User enters weather values in UI
2. Data is scaled using StandardScaler
3. Model predicts flood probability
4. Output is displayed on web page

---

## 🖥️ Run Locally
```bash
pip install -r requirements.txt
python model_training.py
python app.py