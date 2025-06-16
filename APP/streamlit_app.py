import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Define features and targets
features = ['Page_total_likes', 'Post_Hour', 'Lifetime_Post_Total_Reach',
            'Lifetime_Post_Total_Impressions', 'Lifetime_Engaged_Users',
            'Lifetime_People_who_have_liked_your_Page_and_engaged_with_your_post',
            'engagement_rate', 'interactions_per_hour']
targets = ['like', 'share', 'Total_Interactions']
log_targets = True

# Setup Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📖 Introduction", "📘 Prediction", "📊 EDA", "🧾 Conclusion"])

# -------------------- 📖 Introduction --------------------
with tab1:
    st.title("📖 Project Introduction")
    st.markdown("""
    This Streamlit app is designed to **predict social media post interactions** using machine learning.

    **Dataset**  
    - Facebook page post-level metrics dataset  
    - Features include reach, impressions, engaged users, etc.  
    - Targets: 👍 Likes, 🔄 Shares, and 💬 Total Interactions  

    **Machine Learning Model**  
    - Model: Random Forest Regressor wrapped in MultiOutputRegressor  
    - Feature Engineering: Includes derived features like engagement rate and interactions/hour  
    - Transformations: Log-transform applied to skewed variables  
    - Evaluation: R² and MSE for multi-target regression  

    **Goal**  
    Provide actionable insight to social media managers and marketers to optimize post performance.
    """)

# -------------------- 📘 Prediction --------------------
with tab2:
    st.title("📘 Facebook Post Interaction Predictor")
    st.markdown("""
    Predict the number of:
    - 👍 Likes
    - 🔄 Shares
    - 💬 Total Interactions
    using Random Forest-based regression.
    """)

    st.sidebar.header("📥 Enter Post Metrics")

    # Inputs
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

    # Predict Button
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

        # Log-transform and scale
        input_df_log = input_df.copy()
        for col in features:
            input_df_log[col] = np.log1p(input_df_log[col])
        input_scaled = scaler.transform(input_df_log)

        # Predict and inverse-transform
        pred_log = model.predict(input_scaled)
        prediction = np.expm1(pred_log) if log_targets else pred_log

        st.subheader("📈 Predicted Results")
        for i, col in enumerate(targets):
            st.metric(label=f"{col.title()}", value=f"{prediction[0][i]:,.2f}")

# -------------------- 📊 EDA --------------------
with tab3:
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.markdown("Use the buttons below to explore visualizations.")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("📦 Box Plots"):
            st.subheader("Box Plot # 1")
            st.image("shares.png", use_column_width=False)
            st.subheader("Box Plot # 2")
            st.image("eda_plots/box_plot_targets.png", use_column_width=False)

    with col2:
        if st.button("🔗 Pair Plots"):
            st.subheader("Pair Plot of Target Columns")
            st.image("eda_plots/pair_plot_targets.png", use_column_width=False)

    with col3:
        if st.button("📊 Distribution"):
            st.subheader("Distribution # 1")
            st.image("eda_plots/dist_Lifetime_People_who_have_liked_your_Page_and_engaged_with_your_post.png",use_column_width=False)
            st.subheader("Distribution # 2")
            st.image("eda_plots/dist_Category.png",use_column_width=False)
            st.subheader("Distribution # 3")
            st.image("eda_plots/dist_Lifetime_Engaged_Users.png",use_column_width=False)
            st.subheader("Distribution # 4")
            st.image("eda_plots/dist_comment.png",use_column_width=False)
            st.subheader("Distribution # 5")
            st.image("eda_plots/dist_Lifetime_Post_Consumers.png",use_column_width=False)
            st.subheader("Distribution # 6")
            st.image("eda_plots/dist_Lifetime_Post_Consumptions.png",use_column_width=False)
            st.subheader("Distribution # 7")
            st.image("eda_plots/dist_Lifetime_Post_Impressions_by_people_who_have_liked_your_Page.png",use_column_width=False)
            st.subheader("Distribution # 8")
            st.image("eda_plots/dist_Lifetime_Post_Total_Impressions.png",use_column_width=False)
            st.subheader("Distribution # 9")
            st.image("eda_plots/dist_Lifetime_Post_Total_Reach.png",use_column_width=False)
            st.subheader("Distribution # 10")
            st.image("eda_plots/dist_Lifetime_Post_reach_by_people_who_like_your_Page.png",use_column_width=False)
            st.subheader("Distribution # 11")
            st.image("eda_plots/dist_Page_total_likes.png",use_column_width=False)
            st.subheader("Distribution # 12")
            st.image("eda_plots/dist_Paid.png",use_column_width=False)
            st.subheader("Distribution # 13")
            st.image("eda_plots/dist_Post_Hour.png",use_column_width=False)
            st.subheader("Distribution # 14")
            st.image("eda_plots/dist_Post_Month.png",use_column_width=False)
            st.subheader("Distribution # 15")
            st.image("eda_plots/dist_Post_Weekday.png",use_column_width=False)
            st.subheader("Distribution # 16")
            st.image("eda_plots/dist_Total_Interactions.png",use_column_width=False)
            st.subheader("Distribution # 17")
            st.image("eda_plots/dist_like.png",use_column_width=False)
            st.subheader("Distribution # 18")
            st.image("eda_plots/dist_share.png",use_column_width=False)

    with col4:
        if st.button("📊 Summary"):
            st.subheader("Summary of Dataset")
            df_summary = pd.read_csv("eda_plots/summary_statistics.csv")
            st.dataframe(df_summary.style.format(precision=2), use_container_width=True)

# -------------------- 🧾 Conclusion --------------------
with tab4:
    st.title("🧾 Conclusion")
    st.markdown("""
    ### ✅ Key Takeaways:
    - **Multi-output regression** allows us to predict multiple engagement metrics at once.
    - Using engineered features like `engagement_rate` and `interactions_per_hour` improves prediction accuracy.
    - Log-transforming skewed data helps reduce model bias and variance.

    ### 💡 Model Performance:
    The Random Forest-based model performs well in capturing non-linear patterns in post-level metrics.

    ### 📌 Potential Improvements:
    - Tune hyperparameters with GridSearchCV
    - Explore deep learning or gradient boosting models
    - Integrate additional metadata (e.g., content type, hashtags, etc.)

    ### 🚀 Application Use:
    This tool helps content strategists **forecast engagement** and optimize the **timing and type** of posts to maximize reach and interaction.
    """)
