import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Define feature names
features = ['Page_total_likes', 'Post_Hour', 'Lifetime_Post_Total_Reach',
            'Lifetime_Post_Total_Impressions', 'Lifetime_Engaged_Users',
            'Lifetime_People_who_have_liked_your_Page_and_engaged_with_your_post',
            'engagement_rate', 'interactions_per_hour']

targets = ['like', 'share', 'Total_Interactions']
log_targets = True  # Set to True if your model was trained with log-transformed targets

# Streamlit App UI
st.title("📘 Facebook Post Interaction Predictor")
st.markdown("""
This app uses a machine learning model trained on Facebook post metrics to predict the number of:

- 👍 Likes
- 🔄 Shares
- 💬 Total Interactions (including likes, shares, comments, etc.)

The model is based on a **Random Forest Regressor** trained using **multi-output regression** on historical post data. It also includes:
- Feature engineering (like engagement rate & interactions per hour)
- Log-transformation and scaling for better performance
- Inverse transformation after prediction to return real values

📌 Adjust the inputs from the sidebar to see predicted engagement for a Facebook post.
""")

st.sidebar.header("📥 Enter Post Metrics")

# Input fields
page_likes = st.sidebar.number_input("Page Total Likes", value=30000)
post_hour = st.sidebar.slider("Post Hour (0–23)", 0, 23, value=14)
reach = st.sidebar.number_input("Lifetime Post Total Reach", value=18000)
impressions = st.sidebar.number_input("Lifetime Post Total Impressions", value=20000)
engaged_users = st.sidebar.number_input("Lifetime Engaged Users", value=1200)
page_likers_engaged = st.sidebar.number_input(
    "People Who Liked Page & Engaged", value=900)

# Derived features
total_interactions_est = 1500  # default estimate for computing derived metrics
engagement_rate = st.sidebar.number_input("Engagement Rate", value=total_interactions_est / impressions)
interactions_per_hour = st.sidebar.number_input("Interactions Per Hour", value=total_interactions_est / post_hour)

# Predict button
if st.button("🔍 Predict Interactions"):
    # Assemble input into DataFrame
    input_df = pd.DataFrame([{
        'Page_total_likes': page_likes,
        'Post_Hour': post_hour,
        'Lifetime_Post_Total_Reach': reach,
        'Lifetime_Post_Total_Impressions': impressions,
        'Lifetime_Engaged_Users': engaged_users,
        'Lifetime_People_who_have_liked_your_Page_and_engaged_with_your_post': page_likers_engaged,
        'engagement_rate': engagement_rate,
        'interactions_per_hour': interactions_per_hour
    }])

    # Apply log1p transformation
    input_df_log = input_df.copy()
    for col in features:
        input_df_log[col] = np.log1p(input_df_log[col])

    # Apply scaling
    input_scaled = scaler.transform(input_df_log)

    # Make prediction
    pred_log = model.predict(input_scaled)

    # Inverse log1p if needed
    if log_targets:
        prediction = np.expm1(pred_log)
    else:
        prediction = pred_log

    # Display predictions
    st.subheader("📈 Predicted Results")
    for i, col in enumerate(targets):
        st.metric(label=f"{col.title()}", value=f"{prediction[0][i]:.2f}")