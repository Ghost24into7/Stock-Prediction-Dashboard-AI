# 📈 Stock Prediction Dashboard

## 📝 Overview
The **Stock Prediction Dashboard** is a web-based application that provides stock market predictions using AI models. The application uses **GRU (Gated Recurrent Unit)** neural networks for time-series forecasting, along with data fetched from **Yahoo Finance** through the `yfinance` library. The dashboard is built using **Python**, **Flask**, and **Bootstrap**, with an intuitive interface designed with **HTML** and **CSS**.

## 🚀 Features
- **Stock Data Visualization**: Historical stock price trends with interactive charts.
- **AI-Powered Predictions**: Future stock price predictions using GRU-based models.
- **Dynamic Data Retrieval**: Real-time stock data fetched via the `yfinance` library.
- **User-Friendly Interface**: Clean and responsive design using Bootstrap for seamless user interaction.
- **Custom Predictions**: Users can select stocks and time ranges for predictions.

## 🛠️ Technologies Used
- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, Bootstrap
- **AI Model**: GRU (Gated Recurrent Unit) Neural Networks
- **Data Source**: Yahoo Finance (via `yfinance`)
- **Visualization**: Matplotlib, Plotly, or similar tools

## 🛑 Prerequisites
- **Python 3.8+**
- Libraries:
  - `Flask`
  - `yfinance`
  - `numpy`
  - `pandas`
  - `matplotlib` or `plotly`
  - `tensorflow` or `pytorch` (for GRU model)
  - `Bootstrap` (included via CDN)
- Basic knowledge of machine learning and Flask.

## 🔧 Setup Instructions
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd stock-prediction-dashboard
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\\Scripts\\activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Flask app:
   ```bash
   flask run
   ```

5. Open the application in your browser:
   ```
   http://127.0.0.1:5000/
   ```

## 📖 How to Use
1. Navigate to the dashboard in your browser.
2. Enter the stock ticker symbol (e.g., `AAPL` for Apple).
3. Select the time range for predictions (e.g., last 30 days, next 7 days).
4. View historical trends and AI-generated predictions displayed graphically.

## 📊 Output
![image](https://github.com/user-attachments/assets/fa3a8090-084f-460a-a7fc-fbf8a4aba358)


### Predictions:
![image](https://github.com/user-attachments/assets/856b29aa-c4a3-456f-8031-0fd8cc4f6b3d)
![image](https://github.com/user-attachments/assets/6d855918-754b-410a-b814-d6c7e5fbe0ef)



## 🚀 Future Enhancements
- Add support for multiple AI models (e.g., LSTM, Transformer).
- Implement portfolio analysis and optimization tools.
- Enable user authentication for personalized dashboards.

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
