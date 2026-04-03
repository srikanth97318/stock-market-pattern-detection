# 📊 EigenStock AI

A real-time stock market analysis dashboard built using **Linear Algebra** techniques to detect correlations and dominant market trends.

---

## 🚀 Overview

EigenStock AI is a financial analytics application that uses mathematical concepts such as **matrix multiplication** and **eigen decomposition** to analyze stock price data.

It helps uncover:

* Hidden relationships between stocks
* Overall market trends
* Key stocks influencing market movement

---

## 🧠 Features

* 📈 Real-time stock data (yFinance API)
* 🔢 Matrix-based data representation
* ⚖️ Data normalization (Z-score)
* 🔗 Correlation analysis using matrix multiplication
* 🧬 Eigenvalue & eigenvector computation
* 🔥 Dominant trend detection (Principal Component)
* 📊 Interactive dashboard (Streamlit + Plotly)

---

## 🛠️ Tech Stack

* Python
* Streamlit
* NumPy
* Pandas
* Plotly
* yFinance

---

## 📂 Project Structure

```
eigenstock-ai/
│── app.py
│── requirements.txt
│── README.md
```

---

## ⚙️ Installation & Run

1. Clone the repository:

```
git clone https://github.com/your-username/eigenstock-ai.git
cd eigenstock-ai
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the app:

```
streamlit run app.py
```

---

## 📊 How It Works

1. **Data Collection**
   Fetches real-time stock prices

2. **Matrix Representation**
   Converts stock data into matrix form

3. **Normalization**
   Standardizes data for fair comparison

4. **Relationship Matrix**
   Computes:
   R = A × Aᵀ

5. **Eigen Decomposition**
   Finds dominant patterns in data

6. **Trend Analysis**
   Identifies key stocks influencing the market

---

## 📈 Output

* Stock trend visualization
* Correlation heatmap
* Eigenvalue analysis (market strength)
* Stock influence ranking

---

## 🌐 Deployment

This project can be deployed using:

* Streamlit Cloud
* Render
* Hugging Face Spaces

---

## 🎯 Use Cases

* Financial data analysis
* Portfolio trend analysis
* Academic projects (Linear Algebra / Data Science)
* Market research

---

## 📌 Future Improvements

* 📉 Stock price prediction
* 🔔 Buy/Sell signals
* ⏱️ Live auto-refresh
* 📊 Advanced financial indicators

---

## 👨‍💻 Author

Developed as part of a **Linear Algebra-based stock analysis project**.

---

## ⭐ Acknowledgements

* yFinance for stock data
* Streamlit for dashboard framework

---

## 📜 License

This project is for educational purposes.
