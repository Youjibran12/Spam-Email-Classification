from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from math import pi
import numpy as np

df = pd.read_csv("spam.csv", encoding='latin1')

df = df.rename(columns={'v1': 'label', 'v2': 'message'})
df = df[['label', 'message']]

df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['message'])
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Using Mutinomial Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy of the model is:", acc)
print("\nConfusion Matrix:\n", cm)
print("\nPrecision:", prec)
print("Recall:", rec)
print("F1-Score:", f1)

pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(model, open('model.pkl', 'wb'))
print("\nModel and vectorizer saved successfully")

class_counts = df['label'].value_counts()
plt.figure(figsize=(6,4))
plt.bar(class_counts.index, class_counts.values, color=['skyblue', 'salmon'])
plt.title("Class Distribution")
plt.xlabel("Label")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [acc, prec, rec, f1]
N = len(metrics)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
values += values[:1]

plt.figure(figsize=(6,6))
ax = plt.subplot(111, polar=True)
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], metrics)
ax.plot(angles, values, linewidth=2)
ax.fill(angles, values, alpha=0.25)
plt.title("Model Performance Radar Chart")
plt.ylim(0,1)
plt.tight_layout()
plt.show()


plt.figure(figsize=(5,4))
plt.imshow(cm, cmap='Blues', interpolation='nearest')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
plt.xticks([0,1], ['Ham', 'Spam'])
plt.yticks([0,1], ['Ham', 'Spam'])
plt.tight_layout()
plt.show()

