# HPV Risk Prediction and Classification Using Machine Learning

## Overview
This project leverages machine learning techniques to predict the risk of Human Papillomavirus (HPV) infection. By utilizing the **XGBoost algorithm** and an interactive chatbot interface, the system enables efficient and accurate risk assessment, making it accessible to individuals in resource-limited settings.

## Features
- **Machine Learning Model**: Implements XGBoost for high accuracy in risk prediction.
- **Conversational Interface**: User-friendly chatbot for symptom and demographic data input.
- **Real-Time Risk Assessment**: Delivers personalized HPV risk predictions with probabilities.
- **Accessible Design**: Tailored for both healthcare professionals and individuals.

## Dataset
- **Size**: 1,000 records
- **Features**: 27 attributes, including:
  - **Demographic**: Age, marital status, number of sexual partners.
  - **Clinical**: Symptoms such as menstrual irregularities and weight loss.
  - **Behavioral**: Smoking and alcohol habits, physical activity levels.
  - **Reproductive Health**: Age of first intercourse, family planning history.

## Technology Stack
- **Programming Language**: Python
- **Machine Learning Framework**: XGBoost
- **Conversational Interface**: DialogGPT & Gemini
- **Data Visualization**: Matplotlib, Seaborn
- **Backend**: Flask (or FastAPI)

## How It Works
1. **Input Data**:
   - User symptoms and demographic details are collected via a chatbot.
2. **Data Preprocessing**:
   - Feature selection and data normalization ensure quality predictions.
3. **Model Training**:
   - The XGBoost model is trained on 80% of the dataset and validated on the remaining 20%.
   - Key metrics include AUC (98%) and confusion matrix.
4. **Real-Time Interaction**:
   - User inputs are processed using the trained model.
5. **Risk Prediction**:
   - Results are displayed with risk percentages in a simple format.

## Project Workflow
1. Collect user inputs through the chatbot.
2. Preprocess data for accurate predictions.
3. Train and validate the XGBoost model.
4. Predict and display HPV risk in real-time.
5. Option to restart the assessment with new inputs.

## Results
- **Confusion Matrix**:
  - High accuracy with minimal false negatives.
- **AUC**: 0.98, indicating excellent model performance.
- **Feature Importance**: Highlights critical factors like age and symptoms.
- **User Feedback**: Simplified interface for improved accessibility.

## Future Improvements
- Expand dataset for broader demographic coverage.
- Add multilingual support for a wider audience.
- Integrate mobile support for increased accessibility.
- Explore applications for other health risk assessments.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/hpv-risk-prediction.git
   
2. Run App:
   ```bash
   python app1.py
3. Run Chat Application:
   ```bash
   python chatbot.py

### Happy Coding!
