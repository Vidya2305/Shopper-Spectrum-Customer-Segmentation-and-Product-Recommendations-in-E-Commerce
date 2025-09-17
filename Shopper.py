import streamlit as st
import joblib
import pandas as pd
from streamlit_option_menu import option_menu
#📱 Streamlit App Features

with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["Home", "Recommendation", "Clustering"],
        icons=["house", "cart", "people"],
        menu_icon="cast",
        default_index=0,
    )

if selected == "Home":
    st.title("🏠 Home")
    st.write("""
    Welcome to the Shopper App!  
    Use the sidebar to navigate between:
    - 🛍️ Recommendation
    - 👥 Clustering
    """)
    with st.expander("📌 About"):
        st.markdown("""
        🛍️ About – Recommendation
        
        → This feature uses a Cosine Similarity Recommendation System to suggest products that are similar to the one you input.
        
        → Enter a product name in the text box.
        
        → The system searches the trained model and recommends the Top 5 similar products.
        
        → Recommendations are displayed in a clean, styled list.
        
        → This helps customers discover new items.

        👥 About – Clustering
        
        → This tool applies RFM Analysis (Recency, Frequency, Monetary) with KMeans Clustering to segment customers.
        
        → Recency: How recently a customer purchased.
        
        → Frequency: How often they purchase.
        
        → Monetary: How much they spend in total.
        
        → Enter Recency, Frequency, and Monetary values to predict the customer cluster.
        
        → By predicting the customer’s cluster, we can:
        
            ✏️ Identify high-value customers 🏆
            
            ✏️ Spot at-risk or inactive groups ⚠️
        
            ✏️ Personalize marketing and retention strategies 🎯
        """)

#🎯 1️ Product Recommendation Module
#Objective: When a user inputs a product name, the app recommends 5 similar products based on collaborative filtering.
#Functionality:
#● Text input box for Product Name
#● Button: Get Recommendations
#● Display 5 recommended products as a styled list or card view
elif selected == "Recommendation":
    st.title("🛒 Product Recommender")
    st.write("Enter a product name to get recommendations.")

    product_similarity_df = joblib.load("product_similarity.pkl")

    clean_columns = pd.Index([col.strip().upper() for col in product_similarity_df.columns])
    product_similarity_df.columns = clean_columns
    product_similarity_df.index = clean_columns
    product_list = clean_columns.tolist()
    product_list_display = [prod.title() for prod in product_list]

    def recommend_products(product_name, top_n=5):
        product_name = product_name.strip().upper()
        if product_name not in product_similarity_df.columns:
            return None
        recs = product_similarity_df[product_name].sort_values(ascending=False)[1:top_n + 1]
        return recs.index.tolist()

    product_input_display = st.selectbox("Enter a product name:", product_list_display)
    product_input = product_input_display.upper()

    if st.button("Get Recommendations"):
        if product_input:
            recommendations = recommend_products(product_input)
            if recommendations is None:
                st.error("❌ Product not found! Please try another.")
            else:
                st.success("✅ Top 5 similar products:")
                for i, prod in enumerate(recommendations, 1):
                    st.write(f"{i}. {prod.title()}")
        else:
            st.warning("Please enter a product name.")

#🎯 2️ Customer Segmentation Module
#🔍 Functionality:
#● 3 number inputs for:
#○ Recency (in days)
#○ Frequency (number of purchases)
#○ Monetary (total spend)
#● Button: Predict Cluster
#● Display: Cluster label (e.g., High-Value, Regular, Occasional, At-Risk)
elif selected == "Clustering":
    st.title("👥 Customer Segmentation")
    st.write("Enter **Recency, Frequency** and **Monetary** values to predict the customer cluster.")
    scaler = joblib.load("scaler.pkl")
    kmeans = joblib.load("kmeans_model.pkl")
    cluster_labels = joblib.load("cluster_labels.pkl")

    recency = st.number_input("📅 Recency (Days since last purchase):", min_value=0, max_value=1000, value=30)
    frequency = st.number_input("🔁 Frequency (Number of purchases):", min_value=0, max_value=1000, value=10)
    monetary = st.number_input("💰 Monetary (Total Money spent):", min_value=0.0, value=100.0, step=10.0)

    if st.button("🔮 Predict Cluster"):
        try:
            input_data = pd.DataFrame([[recency, frequency, monetary]],
                                  columns=["Recency", "Frequency", "Monetary"])
            input_scaled = scaler.transform(input_data)
            cluster = int(kmeans.predict(input_scaled)[0])

            cluster_name = cluster_labels.get(cluster, "Unknown")

            st.success(f"✅ Predicted Cluster: **{cluster_name}** (Cluster {cluster})")

        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")

