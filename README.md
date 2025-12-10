Design Approach and Details:
The Email Spam Classifier is designed using a modular and systematic approach to ensure
efficiency, scalability, and accuracy in detecting spam emails. The overall design is divided into
multiple stages from data collection to final deployment with each module performing a specific
function.
 System Architecture:
The system follows a Machine Learning Pipeline Architecture, consisting of the following major
components:
1. Data Collection Module:
This module collects email data from publicly available datasets like Enron or Spam
Assassin. The data is divided into two categories: spam and ham which serve as the
foundation for model training and evaluation.
2. Pre-processing Module:
Raw email text is cleaned by removing unwanted characters, punctuation, numbers, and
stop words. The text is then tokenized and normalized using stemming or lemmatization
to standardize the data for analysis.
3. Feature Extraction Module:
The cleaned text is converted into numerical form using methods like Bag of Words or
TF-IDF. These features help the machine learning model understand the significance of
each word in identifying spam patterns.
4. Model Training Module:
Multiple Machine Learning algorithms such as Naive Bayes, Logistic Regression, and
Support Vector Machine (SVM) are trained using the extracted features. The bestperforming
model is selected based on evaluation metrics.
5. Classification Module:
This is the core component where new email inputs are processed and classified as either
Spam or Ham. The trained model predicts the class label based on learned features.
6. Evaluation Module:
The performance of the classifier is tested using metrics like Accuracy, Precision, Recall,
F1-Score, and Confusion Matrix to ensure reliability and effectiveness.
7. User Interface Module:
A simple web interface (Flask or Streamlit) is designed for users to input email text and
view instant classification results. It improves usability and makes the system accessible
even to non-technical users.

Login page:

<img width="953" height="398" alt="image" src="https://github.com/user-attachments/assets/64f195b8-daea-4139-8e84-52257fd7bf2e" />

Sign up page:

<img width="836" height="389" alt="image" src="https://github.com/user-attachments/assets/a325cd69-313d-4d98-908d-9de8ebea1246" />

Spam detection platform:
<img width="945" height="386" alt="image" src="https://github.com/user-attachments/assets/b6c051e5-7344-4a91-9f1e-cfd235f70fa7" />


