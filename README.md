# Predicting Social Media Post Engagement Levels

This project builds a machine learning pipeline to predict social media post engagement levels (High, Medium, Low) using **only pre-publication features** such as platform, content type, region, and hashtags. The final model is deployed through a **Streamlit web app** for interactive use by marketers, content creators, and analysts.

## 🗂 Dataset
We use the [Viral Social Media Trends and Engagement Analysis](https://www.kaggle.com/datasets/atharvasoundankar/viral-social-media-trends-and-engagement-analysis) dataset (2024), which includes 5,000 posts from TikTok, Instagram, YouTube, and Twitter.

- **Numerical features**: Views, Likes, Shares, Comments  
- **Categorical features**: Platform, Content Type  
- **Text features**: Region, Hashtags  
- **Target**: Engagement Level (`High`, `Medium`, `Low`)

## 🧠 Models
Two supervised ML models are implemented from scratch:

- **Multinomial Logistic Regression**  
- **Random Forest**

They are compared using:
- Accuracy
- Macro-averaged Precision
- Macro-averaged F1 Score

## 🧪 Evaluation & Experiments
- 80/20 train-test split with 5-fold cross-validation
- Hyperparameter tuning (learning rate, tree depth, etc.)
- Feature engineering (e.g., conversion rate, one-hot encoding)
- Early stopping and regularization to mitigate overfitting

## 💻 Streamlit App
The Streamlit app allows users to:
- Input post characteristics (platform, content type, etc.)
- Get real-time predictions from both models
- Visualize class probabilities and feature importance

## ⚠️ Risk Mitigation
- Applied imputation and outlier capping for data quality  
- Used cross-validation and early stopping to reduce overfitting  
- Modular design for easy front-end/backend integration

## 📈 Expected Outcomes
- A working, interpretable engagement predictor
- A real-time decision support tool for content strategy
- Comparative insights between linear and non-linear models

## 👥 Team Contributions

**Ye Wei**: Data cleaning, EDA, feature engineering  
**Yuhao Ma**: Logistic Regression, training loop, metric eval  
**Wayne Wang**: Random Forest implementation, tuning  
**Shengping Ka**: Full pipeline integration, Streamlit frontend

---

## 📜 References

- Ghauri & Zubiaga (2020), *Instagram Engagement Prediction*  
- Sun & Zhu (2019), *Cross-platform Engagement Forecasting*  
- Xiang et al. (2023), *Context-aware Prediction in Social Networks*

---