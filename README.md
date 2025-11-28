# Student Performance Predictor (Streamlit)

Simple Streamlit web app that predicts a student's exam result using a linear regression model trained on study habits, attendance, and homework completion data.

## Project Structure
```
Student_Performance_Predictor_Streamlit/
├── data/
│   └── student_data.csv
├── main.py
├── requirements.txt
└── README.md
```

## Setup
1. Create a virtual environment (optional but recommended).
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Launch the Streamlit app from the project root:
   ```
   streamlit run main.py
   ```

## Dataset
`data/student_data.csv` contains the following columns:
- `Hours_Studied` (hours per day)
- `Attendance` (percentage)
- `Previous_Score` (0-100)
- `Sleep_Hours` (hours per day)
- `Homework_Completion` (percentage)
- `Result` (target to predict, 0-100)

Replace or expand the CSV with your own data following the same headers.

## How It Works
1. The app loads the CSV into pandas and trains a `LinearRegression` model on startup.
2. Users provide input values through sidebar sliders for all features.
3. The model predicts the expected exam result (clipped between 0-100 for realism).
4. The app displays:
   - Predicted score with a live-updating visual bar chart
   - Model evaluation metrics (MAE and RMSE)
   - Actual vs. Predicted scatter plot for test set performance

