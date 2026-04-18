🌲 Forest Fire Prediction System

A machine learning web application that predicts the Fire Weather Index (FWI) based on environmental factors. Built with Python and deployed on AWS for real-time predictions.

🚀 Tech Stack
Backend: Python, Flask
ML: Scikit-learn, NumPy, Pandas
Frontend: HTML/CSS
Deployment: AWS (EC2)
📊 Features
Real-time forest fire risk prediction
Simple web interface for input parameters
Scalable cloud deployment
🧾 Input Parameters

Temperature, RH, Wind Speed, Rain, FFMC, DMC, ISI, Classes, Region

⚙️ Run Locally
git clone https://github.com/your-username/forest-fire-prediction.git
cd forest-fire-prediction
pip install -r requirements.txt
python application.py

Open: http://127.0.0.1:5000/

☁️ Deployment

Deployed on AWS using EC2 with Flask (optionally served via Gunicorn & Nginx).

📌 Project Structure
     application.py
     requirements.txt
     model.pkl
     templates/


📈 Model
Algorithm: (e.g., Linear Regression / Ridge / Lasso)
Metrics: R² Score

Result:
Achieved the acuuracy of 98.37