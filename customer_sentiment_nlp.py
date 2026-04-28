import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_json("Cell_Phones_and_Accessories_5.json", lines=True)
df = df[['reviewText', 'overall']]
df = df.dropna()
df['label'] = df['overall'].apply(lambda x: 1 if x >= 4 else (0 if x <= 2 else None))
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    df['reviewText'], df['label'], test_size=0.2, random_state=42
)


vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


model = LogisticRegression()
model.fit(X_train_tfidf, y_train)


y_pred = model.predict(X_test_tfidf)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

misclassified = X_test[y_test != y_pred]

for i in range(len(y_test)):
    if y_test.iloc[i] != y_pred[i]:
        print("Review:", X_test.iloc[i])
        print("Actual:", y_test.iloc[i])
        print("Predicted:", y_pred[i])
        print()