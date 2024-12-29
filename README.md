# SMS Spam Detection Project  

## Overview  
The **SMS Spam Detection** project is a machine learning initiative designed to classify SMS messages as either **spam** or **not spam**. This solution employs advanced classification algorithms to analyze the content of text messages and make predictions. The project is developed using **Python** and deployed as an interactive web application via **Streamlit**, providing a user-friendly interface for testing the model.  

---

## Technology Stack  
This project leverages the following technologies and libraries:  
- **Python**: Core programming language for data processing and model development.  
- **Scikit-learn**: A library offering tools for model creation, evaluation, and optimization.  
- **Pandas**: Simplifies data manipulation and analysis.  
- **NumPy**: Provides efficient numerical computations.  
- **Streamlit**: Enables rapid web application deployment with minimal effort.  

---

## Features  
1. **Data Collection**:  
   - Acquiring labeled datasets for training and evaluation.  
2. **Data Cleaning and Preprocessing**:  
   - Removing inconsistencies and preparing text for analysis.  
3. **Exploratory Data Analysis (EDA)**:  
   - Visualizing and understanding patterns in the dataset.  
4. **Model Training and Optimization**:  
   - Experimenting with multiple machine learning algorithms and selecting the best-performing model.  
5. **Web Deployment**:  
   - Interactive application accessible via a web interface.  

---

## Dataset  
The dataset, sourced from **[Kaggle](https://www.kaggle.com/uciml/sms-spam-collection-dataset)**, contains over 5,500 labeled SMS messages. These labels—`spam` or `ham` (not spam)—enable the model to learn and distinguish between the two classes effectively.  

---

## Data Cleaning and Preprocessing  
### Steps Involved:  
- **Null Value Removal**: Dropped missing or duplicate entries.  
- **Label Encoding**: Converted categorical labels (`spam`, `ham`) to numeric values.  
- **Text Preprocessing**:  
  - Tokenization: Split text into words.  
  - Special Character Removal: Removed non-alphanumeric characters.  
  - Stop Word Removal: Excluded non-informative words.  
  - Punctuation Removal: Stripped unnecessary symbols.  
  - Stemming: Reduced words to their root forms (e.g., "running" → "run").  
  - Lowercasing: Standardized text by converting to lowercase.  

---

## Exploratory Data Analysis (EDA)  
Key insights and patterns were extracted using EDA techniques. Visualizations included:  
- **Bar Charts**: Displayed frequency distributions of spam vs. non-spam messages.  
- **Pie Charts**: Highlighted proportions of each category.  
- **Heatmaps**: Showed correlations among features.  
- **Word Clouds**: Identified frequently occurring words in spam and non-spam messages.  

---

## Model Development and Evaluation  
### Algorithms Evaluated:  
1. **Naive Bayes**  
2. **Random Forest**  
3. **K-Nearest Neighbors (KNN)**  
4. **Decision Tree**  
5. **Logistic Regression**  
6. **ExtraTreesClassifier**  
7. **Support Vector Classifier (SVC)**  

Each algorithm was trained and tested on the preprocessed dataset. The **Naive Bayes** classifier achieved the highest precision (100%), making it the final choice for deployment.  

---

## Web Application Deployment  
The model is deployed using **Streamlit**, enabling users to interact with it through a web interface.  
### Features:  
- **Input Box**: Users can enter or paste an SMS message.  
- **Real-Time Prediction**: The model classifies the message as **spam** or **not spam** instantly.  

To run the web app locally:  
1. Clone this repository.  
2. Install dependencies using `pip install -r requirements.txt`.  
3. Launch the app using `streamlit run app.py`.  

---

## Conclusion  
The **SMS Spam Detection Project** demonstrates the entire machine learning pipeline, from data acquisition and preprocessing to model deployment. By leveraging cutting-edge tools and methodologies, the project delivers a reliable solution for real-world spam detection scenarios, showcasing the potential of machine learning in practical applications.  
 
## Acknowledgments  
Special thanks to [Kaggle](https://www.kaggle.com) for providing the SMS Spam Collection dataset.  
