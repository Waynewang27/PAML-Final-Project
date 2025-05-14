# Social Media Engagement Predictor

This project builds a machine learning pipeline to predict social media post engagement levels (High, Medium, Low) using **only pre-publication features** such as platform, content type, region, and hashtags. The final model is deployed through a **Streamlit web app** for interactive use by marketers, content creators, and analysts.

## Project Overview

This project investigates the predictability of social media engagement using only features available before a post is published. Using a dataset of 5,000 posts from TikTok, Instagram, YouTube, and Twitter, we implemented two classification models:

- Multinomial Logistic Regression
- Random Forest

The Random Forest model proved to be more effective at classifying all engagement levels, particularly for identifying low-engagement content, which is crucial for content strategy optimization.

## Features

- **Multi-class classification**: Predicts if a post will have High, Medium, or Low engagement
- **Pre-publication prediction**: Uses only features available before posting
- **Interactive web application**: Built with Streamlit for easy use
- **Data exploration**: Visualizations of engagement patterns across platforms
- **Model comparison**: Performance metrics for both implemented algorithms

## Setup Instructions
1. Clone the repository:
git clone <repo>
cd project

2. Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate

3. Install dependencies:
pip install -r requirements.txt

4. Run the Streamlit app:
streamlit run app.py

## Application Structure

The Streamlit app consists of three main sections:
1. **Introduction**: Project overview and methodology
2. **Data Exploration**: Interactive visualizations of the dataset
3. **Engagement Prediction**: Input form to predict engagement for new content

## Model Performance

- **Logistic Regression**: 93.8% accuracy, but poor recall for low-engagement posts
- **Random Forest**: 86.1% accuracy with better class balance and feature importance insights

## Team

This project was developed by Ye Wei, Shengping Ka, Yuhao Ma, and Wayne Wang as part of the PAML final project.

## Dataset

We use the [Viral Social Media Trends and Engagement Analysis](https://www.kaggle.com/datasets/atharvasoundankar/viral-social-media-trends-and-engagement-analysis) dataset (2024), which includes 5,000 posts from TikTok, Instagram, YouTube, and Twitter.

## References

- Ghauri & Zubiaga (2020), *Instagram Engagement Prediction*  
- Sun & Zhu (2019), *Cross-platform Engagement Forecasting*  
- Xiang et al. (2023), *Context-aware Prediction in Social Networks*
