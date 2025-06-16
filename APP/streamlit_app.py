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
log_targets = True

# Setup tabs
tab1, tab2 = st.tabs(["📘 Prediction", "📊 EDA"])

# -------------------- 📘 Prediction Tab --------------------
with tab1:
    st.title("📘 Facebook Post Interaction Predictor")
    st.markdown("""
    This app predicts the number of:
    - 👍 Likes
    - 🔄 Shares
    - 💬 Total Interactions
    based on your post metrics. The model is a **Random Forest Regressor** using **multi-output regression** with feature engineering and log-transformations.
    """)

    st.sidebar.header("📥 Enter Post Metrics")

    # Input fields
    page_likes = st.sidebar.number_input("Page Total Likes", value=30000)
    post_hour = st.sidebar.slider("Post Hour (0–23)", 0, 23, value=14)
    reach = st.sidebar.number_input("Lifetime Post Total Reach", value=18000)
    impressions = st.sidebar.number_input("Lifetime Post Total Impressions", value=20000)
    engaged_users = st.sidebar.number_input("Lifetime Engaged Users", value=1200)
    page_likers_engaged = st.sidebar.number_input("People Who Liked Page & Engaged", value=900)

    # Derived features
    total_interactions_est = 1500
    engagement_rate = st.sidebar.number_input("Engagement Rate", value=total_interactions_est / impressions)
    interactions_per_hour = st.sidebar.number_input("Interactions Per Hour", value=total_interactions_est / post_hour)

    # Prediction button
    if st.button("🔍 Predict Interactions"):
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

        # Log transform and scale
        input_df_log = input_df.copy()
        for col in features:
            input_df_log[col] = np.log1p(input_df_log[col])
        input_scaled = scaler.transform(input_df_log)

        # Prediction
        pred_log = model.predict(input_scaled)
        prediction = np.expm1(pred_log) if log_targets else pred_log

        # Results
        st.subheader("📈 Predicted Results")
        for i, col in enumerate(targets):
            st.metric(label=f"{col.title()}", value=f"{prediction[0][i]:.2f}")

# -------------------- 📊 EDA Tab --------------------
with tab2:
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.markdown("Click the buttons below to view pre-generated visualizations.")

    # Replace these URLs with actual Hugging Face URLs
    box_plot_url = "shares.png"
    pair_plot_url = "like_share_interactions.png"
    dist_plot_url = "interactions.png"

    # Display buttons and graphs
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📦 Box Plots"):
            st.subheader("Box Plots")
            st.image(box_plot_url, use_column_width=True)

    with col2:
        if st.button("🔗 Pair Plots"):
            st.subheader("Pair Plot of Target Columns")
            st.image(pair_plot_url, use_column_width=True)

    with col3:
        if st.button("📊 Distribution"):
            st.subheader("Distribution of Total Interactions")
            st.image(dist_plot_url, use_column_width=True)
