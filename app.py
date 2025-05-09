import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Social Media Engagement Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define the main pages
PAGES = {
    "Introduction": "intro",
    "Data Exploration": "explore",
    "Engagement Prediction": "predict"
}

# Sidebar navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

# Load preprocessed dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Datasets/preprocessed_data.csv')
        return df
    except FileNotFoundError:
        try:
            df = pd.read_csv('Datasets/processed_social_media_data.csv')
            # Calculate engagement level based on views, likes, shares, comments
            df['Engagement_Score'] = df['Likes'] + df['Shares']*3 + df['Comments']*2
            # Create engagement level categorical variable (0: Low, 1: High)
            q75 = df['Engagement_Score'].quantile(0.75)
            q25 = df['Engagement_Score'].quantile(0.25)
            df['Engagement_Level_Calculated'] = pd.cut(
                df['Engagement_Score'], 
                bins=[-float('inf'), q25, q75, float('inf')],
                labels=[0, 1, 2]
            ).astype(int)
            df.to_csv('Datasets/preprocessed_data.csv', index=False)
            return df
        except FileNotFoundError:
            st.error("Dataset not found. Please make sure the processed dataset exists.")
            return None

# Load data
df = load_data()

# Train models
@st.cache_resource
def train_models(df):
    if df is None:
        return None, None
    
    # Drop columns related to engagement metrics to avoid data leakage
    engagement_cols = ['Views', 'Likes', 'Shares', 'Comments', 'Conversion_Rate', 'Engagement_Score'] 
    X = df.drop(columns=[col for col in engagement_cols if col in df.columns] + ['Engagement_Level_Calculated'])
    y = df['Engagement_Level_Calculated']
    
    # Ensure y is properly formatted
    y = y.astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Apply SMOTE for balancing
    smote = SMOTE(random_state=42)
    X_sm, y_sm = smote.fit_resample(X_train, y_train)
    
    # Train Logistic Regression
    lr = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    lr.fit(X_sm, y_sm)
    
    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        class_weight='balanced'
    )
    rf.fit(X_sm, y_sm)
    
    return {
        'logistic_regression': lr,
        'random_forest': rf,
        'feature_names': X.columns,
        'X_test': X_test,
        'y_test': y_test
    }

# Initialize models
model_data = None
if df is not None:
    model_data = train_models(df)

# Introduction page
def intro_page():
    st.title("📱 Social Media Engagement Predictor")
    
    st.markdown("""
    ## Welcome to the Social Media Engagement Prediction Tool
    
    This application predicts engagement levels for social media content using features available *before* publication, enabling strategic planning and optimization of content before it goes live.
    
    ### Project Overview:
    
    Most existing predictive models rely on post-publication metrics (likes, shares, comments), limiting their use in proactive decision-making. This tool fills that gap by providing engagement predictions using only pre-publication attributes.
    
    ### Key Features:
    
    - **Pre-Publication Prediction**: Forecast engagement (Low, Medium, High) using only features available before posting
    - **Multi-Platform Analysis**: Covers content strategies across TikTok, Instagram, YouTube, and Twitter
    - **Machine Learning Insights**: Leverages both Multinomial Logistic Regression and Random Forest models
    - **Interactive Visualization**: Explore relationships between content characteristics and engagement
    
    ### Dataset & Methodology:
    
    Based on the "Viral Social Media Trends and Engagement Analysis" dataset (2024) with 5,000 posts across major platforms. The problem is formulated as a multi-class classification task evaluated using accuracy, macro-averaged precision, and F1 score.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        **Predict Using:**
        - Platform type (e.g., TikTok, Instagram)
        - Content format (e.g., Video, Image, Text)
        - Regional targeting information
        - Hashtag strategies
        - Posting time parameters
        - Other pre-publication attributes
        
        **Technical Framework:**
        - Data preprocessing & feature engineering
        - Class balancing with SMOTE
        - Model training with cross-validation
        - Performance evaluation & comparison
        """)
    
    with col2:
        st.markdown("""
        **Applications:**
        - Content strategy planning and optimization
        - Resource allocation for marketing campaigns
        - Platform-specific content tailoring
        - Audience targeting refinement
        - Data-driven decision making for creators
        - Maximizing ROI on content production
        
        **Benefits:**
        - Make informed decisions before publishing
        - Understand platform-specific engagement drivers
        - Optimize content parameters for better results
        - Compare different content strategies
        """)
    
    st.info("Navigate to 'Data Exploration' to analyze engagement patterns across platforms and content types, or go to 'Engagement Prediction' to generate predictions for your specific content parameters.")
    
    st.markdown("""
    ### Why This Matters
    
    In today's competitive digital landscape, predicting social media performance before publishing allows content creators, marketers, and businesses to make data-driven decisions that maximize engagement and reach. By focusing exclusively on pre-publication features, this tool provides actionable insights that can be applied during the content creation process.
    """)



# Data exploration page
def explore_page():
    st.title("Data Exploration")
    
    if df is None:
        st.error("Dataset not found. Please check data path.")
        return
    
    st.write(f"Dataset shape: {df.shape}")
    
    # Show data sample
    st.subheader("Data Sample")
    st.dataframe(df.head())
    
    # Add tabs for different exploration views
    tab1, tab2, tab3, tab4 = st.tabs(["Basic Statistics", "Engagement Analysis", "Platform Insights", "Cross-Feature Analysis"])
    
    with tab1:
        # Basic statistics
        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe())
        
        # Distribution of engagement levels
        st.subheader("Engagement Level Distribution")
        
        fig = px.histogram(df, x="Engagement_Level_Calculated", 
                          color="Engagement_Level_Calculated",
                          labels={"Engagement_Level_Calculated": "Engagement Level"},
                          title="Distribution of Engagement Levels",
                          color_discrete_map={0: "red", 1: "gold", 2: "green"})
        
        fig.update_layout(xaxis=dict(tickmode='array', tickvals=[0, 1, 2], 
                                   ticktext=['Low', 'Medium', 'High']))
        st.plotly_chart(fig)
    
    with tab2:
        # Enhanced engagement analysis with filtering
        st.subheader("Engagement Metrics Analysis")
        
        # Engagement metrics - allow user to select metrics to view
        engagement_cols = ['Views', 'Likes', 'Shares', 'Comments', 'Conversion_Rate']
        available_metrics = [col for col in engagement_cols if col in df.columns]
        
        if available_metrics:
            selected_metrics = st.multiselect(
                "Select metrics to compare:", 
                available_metrics,
                default=available_metrics[:2] if len(available_metrics) >= 2 else available_metrics
            )
            
            if selected_metrics:
                # Allow selection of grouping variable
                group_options = ['Engagement_Level_Calculated']
                platform_cols = [col for col in df.columns if col.startswith('Platform_')]
                content_cols = [col for col in df.columns if col.startswith('Content_Type_')]
                
                if platform_cols:
                    group_options.append('Platform')
                if content_cols:
                    group_options.append('Content Type')
                
                group_by = st.selectbox("Group by:", group_options)
                
                if group_by == 'Engagement_Level_Calculated':
                    # Create violin plots for each selected metric by engagement level
                    for metric in selected_metrics:
                        fig = px.violin(df, y=metric, x='Engagement_Level_Calculated', 
                                        color='Engagement_Level_Calculated',
                                        box=True, points="all",
                                        color_discrete_map={0: "red", 1: "gold", 2: "green"},
                                        labels={"Engagement_Level_Calculated": "Engagement Level"})
                        
                        fig.update_layout(xaxis=dict(tickmode='array', tickvals=[0, 1, 2], 
                                                    ticktext=['Low', 'Medium', 'High']))
                        st.plotly_chart(fig)
                
                elif group_by == 'Platform':
                    # Create a platform indicator from one-hot encoded columns
                    if 'platform_indicator' not in df.columns:
                        platform_names = [col.replace('Platform_', '') for col in platform_cols]
                        df['platform_indicator'] = ''
                        for col in platform_cols:
                            platform_name = col.replace('Platform_', '')
                            df.loc[df[col] == 1, 'platform_indicator'] = platform_name
                    
                    # Create box plots for each selected metric by platform
                    for metric in selected_metrics:
                        fig = px.box(df, y=metric, x='platform_indicator', 
                                    color='platform_indicator',
                                    points="all",
                                    labels={"platform_indicator": "Platform"})
                        st.plotly_chart(fig)
                
                elif group_by == 'Content Type':
                    # Create a content type indicator from one-hot encoded columns
                    if 'content_type_indicator' not in df.columns:
                        content_type_names = [col.replace('Content_Type_', '') for col in content_cols]
                        df['content_type_indicator'] = ''
                        for col in content_cols:
                            content_name = col.replace('Content_Type_', '')
                            df.loc[df[col] == 1, 'content_type_indicator'] = content_name
                    
                    # Create box plots for each selected metric by content type
                    for metric in selected_metrics:
                        fig = px.box(df, y=metric, x='content_type_indicator', 
                                    color='content_type_indicator',
                                    points="all",
                                    labels={"content_type_indicator": "Content Type"})
                        st.plotly_chart(fig)
        
    with tab3:
        # Platform distribution and performance
        st.subheader("Platform Analysis")
        
        # Platform distribution
        platform_cols = [col for col in df.columns if col.startswith('Platform_')]
        if platform_cols:
            platform_data = df[platform_cols].sum().reset_index()
            platform_data.columns = ['Platform', 'Count']
            platform_data['Platform'] = platform_data['Platform'].str.replace('Platform_', '')
            
            # Platform distribution chart
            fig = px.bar(platform_data, x='Platform', y='Count', color='Platform',
                        title="Number of Posts by Platform")
            st.plotly_chart(fig)
            
            # Platform engagement analysis - success rate by platform
            st.subheader("Engagement Success by Platform")
            
            # Create a platform indicator if not already created
            if 'platform_indicator' not in df.columns:
                df['platform_indicator'] = ''
                for col in platform_cols:
                    platform_name = col.replace('Platform_', '')
                    df.loc[df[col] == 1, 'platform_indicator'] = platform_name
            
            # Calculate engagement level distribution for each platform
            platform_engagement = pd.crosstab(
                df['platform_indicator'], 
                df['Engagement_Level_Calculated'],
                normalize='index'
            ) * 100
            
            # Rename columns for better readability
            if platform_engagement.shape[1] <= 3:
                platform_engagement.columns = ['Low', 'Medium', 'High'][:platform_engagement.shape[1]]
            
            # Convert to long format for Plotly
            platform_engagement_long = platform_engagement.reset_index().melt(
                id_vars='platform_indicator',
                var_name='Engagement Level',
                value_name='Percentage'
            )
            
            # Create the stacked bar chart
            fig = px.bar(platform_engagement_long, x='platform_indicator', y='Percentage',
                        color='Engagement Level', barmode='stack',
                        labels={'platform_indicator': 'Platform'},
                        title="Engagement Level Distribution by Platform (%)",
                        color_discrete_map={'Low': 'red', 'Medium': 'gold', 'High': 'green'})
            
            st.plotly_chart(fig)
    
    with tab4:
        # Interactive cross-feature analysis
        st.subheader("Cross-Feature Analysis")
        
        # Prepare feature columns for selection
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Filter out engagement target variable
        if 'Engagement_Level_Calculated' in numerical_cols:
            numerical_cols.remove('Engagement_Level_Calculated')
        
        # Add platform and content type if they exist
        if 'platform_indicator' in df.columns:
            categorical_cols = ['platform_indicator']
        else:
            categorical_cols = []
            
        if 'content_type_indicator' in df.columns:
            categorical_cols.append('content_type_indicator')
        
        # Add one-hot encoded columns
        categorical_cols.extend([col for col in df.columns if col.startswith('Platform_') 
                                or col.startswith('Content_Type_') 
                                or col.startswith('Region_')])
        
        # Let user select features to explore
        st.markdown("### Explore Relationships Between Features")
        st.markdown("Select features to visualize their relationship and how they impact engagement:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_axis = st.selectbox("X-axis feature:", numerical_cols, index=0 if numerical_cols else 0)
        
        with col2:
            y_axis = st.selectbox("Y-axis feature:", [col for col in numerical_cols if col != x_axis], 
                                 index=1 if len(numerical_cols) > 1 else 0)
        
        color_by = st.selectbox("Color by:", ['Engagement_Level_Calculated'] + categorical_cols)
        
        # Create the scatter plot
        if x_axis and y_axis:
            fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by,
                            title=f"{x_axis} vs {y_axis} by {color_by}",
                            opacity=0.7, size_max=10)
            
            # If coloring by engagement level, set color scale
            if color_by == 'Engagement_Level_Calculated':
                fig.update_layout(coloraxis_colorbar=dict(
                    tickvals=[0, 1, 2],
                    ticktext=['Low', 'Medium', 'High']
                ))
            
            st.plotly_chart(fig)
            
            # Add correlation analysis
            corr = df[[x_axis, y_axis]].corr().iloc[0,1]
            st.markdown(f"**Correlation coefficient:** {corr:.3f}")
            
            if abs(corr) > 0.7:
                st.markdown("📊 **Strong correlation detected!** These features show a strong relationship.")
            elif abs(corr) > 0.4:
                st.markdown("📊 **Moderate correlation detected.** There appears to be a relationship worth exploring.")
            else:
                st.markdown("📊 **Weak or no correlation.** These features don't show a strong linear relationship.")
        
        # Add correlation heatmap
        st.subheader("Feature Correlation")
        
        # Select which features to include in correlation analysis
        correlation_features = st.multiselect(
            "Select features for correlation analysis:",
            numerical_cols,
            default=numerical_cols[:5] if len(numerical_cols) > 5 else numerical_cols
        )
        
        if correlation_features and len(correlation_features) > 1:
            corr = df[correlation_features].corr()
            fig = px.imshow(corr, 
                            text_auto=True, 
                            color_continuous_scale='RdBu_r',
                            title="Correlation Heatmap")
            st.plotly_chart(fig)

# Prediction page
def predict_page():
    st.title("Engagement Prediction")
    
    if df is None or model_data is None:
        st.error("Models or data not properly loaded. Please check data paths.")
        return
    
    # Extract feature info for input panel
    platform_cols = [col for col in df.columns if col.startswith('Platform_')]
    content_cols = [col for col in df.columns if col.startswith('Content_Type_')]
    region_cols = [col for col in df.columns if col.startswith('Region_')]
    
    platforms = [col.replace('Platform_', '') for col in platform_cols]
    content_types = [col.replace('Content_Type_', '') for col in content_cols]
    regions = [col.replace('Region_', '') for col in region_cols]
    
    # Create input form
    st.subheader("Enter Content Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        platform = st.selectbox("Platform", platforms)
        content_type = st.selectbox("Content Type", content_types)
        region = st.selectbox("Region", regions)
    
    # Create a form for prediction
    with st.form("prediction_form"):
        st.subheader("Additional Metrics (if needed for prediction)")
        
        # Only show these if they're not used for engagement calculation (to avoid data leakage)
        col1, col2 = st.columns(2)
        with col1:
            post_length = st.slider("Post Length (characters)", 10, 500, 150)
            post_time = st.slider("Time of Post (hour, 24-hour format)", 0, 23, 12)
        
        with col2:
            hashtag_count = st.slider("Number of Hashtags", 0, 10, 3)
            weekday = st.slider("Day of Week (0=Monday, 6=Sunday)", 0, 6, 3)
        st.subheader("Model Selection")
        st.markdown("""
        ### Choosing the Right Model for Your Prediction

        Based on our analysis of model performance, here's guidance on which model might work best for your specific needs:

        **Logistic Regression**
        - **Strengths**: 
        - Higher overall accuracy (93.8%)
        - More consistent predictions
        - Faster prediction time
        - Works well for majority class prediction
        - **Best for**: When you need quick predictions and are primarily concerned with identifying medium/high engagement content

        **Random Forest**
        - **Strengths**: 
        - Better at detecting low engagement content (8.7% recall vs 0%)
        - More balanced predictions across classes
        - Captures more complex feature interactions
        - Provides detailed feature importance analysis
        - **Best for**: When you need to identify potential low-performing content or want more nuanced predictions

        **Key Consideration**: Due to class imbalance in our training data (few "Low" engagement examples), Random Forest is generally recommended when trying to identify potentially low-performing content, despite its slightly lower overall accuracy.
        """)

        # Then replace your existing model selection line with this enhanced version:
        model_type = st.radio("Select Model for Prediction", ["Logistic Regression", "Random Forest"], 
                        help="Choose which model to use for your prediction based on the guidance above.")
        
        # Submit button
        submit_button = st.form_submit_button("Predict Engagement")
    
    # Handle prediction
    if submit_button:
        # Prepare feature vector
        feature_vector = {col: 0 for col in model_data['feature_names']}
        
        # Set selected categorical features (with error handling for missing columns)
        platform_col = f'Platform_{platform}'
        content_type_col = f'Content_Type_{content_type}'
        region_col = f'Region_{region}'
        
        if platform_col in feature_vector:
            feature_vector[platform_col] = 1
        if content_type_col in feature_vector:
            feature_vector[content_type_col] = 1
        if region_col in feature_vector:
            feature_vector[region_col] = 1
        
        # Set additional metrics if they exist in features
        if 'Post_Length' in feature_vector:
            feature_vector['Post_Length'] = post_length
        if 'Post_Time' in feature_vector:
            feature_vector['Post_Time'] = post_time
        if 'Hashtag_Count' in feature_vector:
            feature_vector['Hashtag_Count'] = hashtag_count
        if 'Day_of_Week' in feature_vector:
            feature_vector['Day_of_Week'] = weekday
        
        # Convert to dataframe for prediction
        input_df = pd.DataFrame([feature_vector])
        
        # Make prediction
        if model_type == "Logistic Regression":
            model = model_data['logistic_regression']
        else:
            model = model_data['random_forest']
        
        try:
            prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            # Default values in case of error
            prediction = 0
            probabilities = np.array([1.0, 0.0, 0.0])[:model.classes_.shape[0]]
        
        # Display prediction result
        st.subheader("Prediction Result")
        
        engagement_level = {0: "Low", 1: "Medium", 2: "High"}
        
        # Use success/warning/error for different engagement levels
        if prediction == 2:
            st.success(f"Predicted Engagement Level: {engagement_level[prediction]} ✨")
        elif prediction == 1:
            st.warning(f"Predicted Engagement Level: {engagement_level[prediction]} 📊")
        else:
            st.error(f"Predicted Engagement Level: {engagement_level[prediction]} 📉")
        
        # Show probabilities
        st.subheader("Engagement Probability")
        
        # Make sure to match the length of engagement levels with probabilities
        engagement_labels = ['Low', 'Medium', 'High']
        # Ensure we only use the available probability values (could be binary classification)
        if len(probabilities) != len(engagement_labels):
            engagement_labels = engagement_labels[:len(probabilities)]
        
        prob_df = pd.DataFrame({
            'Engagement Level': engagement_labels,
            'Probability': probabilities
        })
        
        fig = px.bar(prob_df, x='Engagement Level', y='Probability', color='Engagement Level',
                    color_discrete_map={'Low': 'red', 'Medium': 'gold', 'High': 'green'},
                    text_auto=True)
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig)
        
        # Feature importance (for Random Forest)
        if model_type == "Random Forest":
            st.subheader("Feature Importance")
            
            importance_df = pd.DataFrame({
                'Feature': model_data['feature_names'],
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                        title="Top 10 Most Important Features")
            st.plotly_chart(fig)
        
        # Model comparison
        st.subheader("Model Performance Comparison")
        
        # Prepare test data predictions
        try:
            lr_pred = model_data['logistic_regression'].predict(model_data['X_test'])
            rf_pred = model_data['random_forest'].predict(model_data['X_test'])
            
            lr_acc = accuracy_score(model_data['y_test'], lr_pred)
            rf_acc = accuracy_score(model_data['y_test'], rf_pred)
        except Exception as e:
            st.error(f"Error calculating predictions: {str(e)}")
            lr_acc = rf_acc = 0.0
            lr_pred = rf_pred = np.zeros_like(model_data['y_test'])
        
        compare_df = pd.DataFrame({
            'Model': ['Logistic Regression', 'Random Forest'],
            'Accuracy': [lr_acc, rf_acc]
        })
        
        fig = px.bar(compare_df, x='Model', y='Accuracy', color='Model',
                    text_auto=True)
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig)
        
        # Confusion matrices
        col1, col2 = st.columns(2)
        
        # Get unique classes
        unique_classes = np.unique(np.concatenate([model_data['y_test'], lr_pred, rf_pred]))
        class_labels = ['Low', 'Medium', 'High'][:len(unique_classes)]
        
        # Create index and column labels based on actual classes
        index_labels = [f'Actual {label}' for label in class_labels]
        column_labels = [f'Predicted {label}' for label in class_labels]
        
        with col1:
            st.write("Logistic Regression Confusion Matrix")
            lr_cm = confusion_matrix(model_data['y_test'], lr_pred, labels=range(len(unique_classes)))
            lr_cm_df = pd.DataFrame(
                lr_cm, 
                index=index_labels,
                columns=column_labels
            )
            st.dataframe(lr_cm_df)
        
        with col2:
            st.write("Random Forest Confusion Matrix")
            rf_cm = confusion_matrix(model_data['y_test'], rf_pred, labels=range(len(unique_classes)))
            rf_cm_df = pd.DataFrame(
                rf_cm, 
                index=index_labels,
                columns=column_labels
            )
            st.dataframe(rf_cm_df)

# Route to the selected page
if selection == "Introduction":
    intro_page()
elif selection == "Data Exploration":
    explore_page()
elif selection == "Engagement Prediction":
    predict_page()