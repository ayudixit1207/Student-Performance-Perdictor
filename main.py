import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


# --- Load dataset ---
DATA_PATH = "data/student_data.csv"
data = pd.read_csv(DATA_PATH)


# --- Streamlit UI ---
st.title("Student Performance Predictor")
st.write(
    "Predict a student's performance using study habits, attendance, "
    "previous scores, sleep hours, and homework completion."
)

st.sidebar.header("Enter Student Details")
hours = st.sidebar.slider("Hours Studied (per day)", 0.0, 12.0, 5.0, 0.5)
attendance = st.sidebar.slider("Attendance %", 0, 100, 80, 1)
prev_score = st.sidebar.slider("Previous Score", 0, 100, 75, 1)
sleep = st.sidebar.slider("Sleep Hours (per day)", 0.0, 12.0, 7.0, 0.5)
homework_completion = st.sidebar.slider("Homework Completion %", 0, 100, 90, 1)

# --- Model Training ---
feature_cols = [
    "Hours_Studied",
    "Attendance",
    "Previous_Score",
    "Sleep_Hours",
    "Homework_Completion",
]
target_col = "Result"

missing_features = [col for col in feature_cols if col not in data.columns]
if missing_features:
    st.warning(
        f"Missing columns in dataset: {missing_features}. "
        "Using default value 0 for them until dataset is updated."
    )
    for col in missing_features:
        data[col] = 0

X = data[feature_cols]
y = data[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

# --- Prediction ---
new_data = [[hours, attendance, prev_score, sleep, homework_completion]]
predicted_result = model.predict(new_data)[0]
predicted_result = float(np.clip(predicted_result, 0, 100))

st.subheader("Predicted Result")
st.success(f"Estimated Score: **{predicted_result:.2f}**")

st.subheader("Predicted Score Visual")
fig_indicator, ax_indicator = plt.subplots(figsize=(6, 1.6))
ax_indicator.barh(
    ["Estimated Score"], [100], color="#e5e7eb", edgecolor="none", height=0.4
)
ax_indicator.barh(
    ["Estimated Score"],
    [predicted_result],
    color="#2563eb",
    edgecolor="none",
    height=0.4,
)
ax_indicator.set_xlim(0, 100)
ax_indicator.set_xlabel("Score")
ax_indicator.set_xticks([0, 25, 50, 75, 100])
ax_indicator.spines["top"].set_visible(False)
ax_indicator.spines["right"].set_visible(False)
ax_indicator.spines["left"].set_visible(False)
ax_indicator.spines["bottom"].set_visible(False)
ax_indicator.tick_params(left=False)
ax_indicator.set_title("Live prediction updates with slider changes")
st.pyplot(fig_indicator)

st.subheader("Model Evaluation")
st.write(f"- Mean Absolute Error (MAE): **{mae:.2f}**")
st.write(f"- Root Mean Squared Error (RMSE): **{rmse:.2f}**")

# --- Visualization ---
st.subheader("Actual vs. Predicted (Test Set)")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred_test, color="#1f77b4", alpha=0.7)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", linewidth=1.5)
ax.set_xlabel("Actual Score")
ax.set_ylabel("Predicted Score")
ax.set_title("Model Performance")
st.pyplot(fig)





