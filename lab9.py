# ===========================
# IMPORTS
# ===========================
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

import lime.lime_text

# ===========================
# LOAD DATA
# ===========================
def load_data():
    data = pd.read_csv(r"/Users/sreehithareddy/Downloads/fake-news.csv")
    
    # 🔥 REMOVE EMPTY ROWS
    data = data.dropna()
    
    # 🔥 ENSURE LABELS ARE STRINGS
    data['label'] = data['label'].astype(str)
    
    X = data['text']
    y = data['label']
    
    return X, y
# ===========================
# SPLIT DATA
# ===========================
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

# ===========================
# STACKING MODEL
# ===========================
from sklearn.model_selection import StratifiedKFold

def create_stacking_model():
    base_models = [
        ('lr', LogisticRegression(max_iter=1000)),
        ('nb', MultinomialNB()),
        ('dt', DecisionTreeClassifier())
    ]
    
    meta_model = LogisticRegression()
    
    model = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=StratifiedKFold(n_splits=2)   
    )
    
    return model

# ===========================
# PIPELINE
# ===========================
def create_pipeline(model):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('classifier', model)
    ])
    return pipeline

# ===========================
# TRAIN
# ===========================
def train_model(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)
    return pipeline

# ===========================
# EVALUATE
# ===========================
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)


# ===========================
# LIME GENERATOR
# ===========================
def explain_with_lime(model, X_train, X_test):
    
    import lime.lime_text
    
    explainer = lime.lime_text.LimeTextExplainer(
        class_names=['FAKE','REAL']
    )
    
    exp = explainer.explain_instance(
        X_test.iloc[0],
        model.predict_proba
    )
    
    return exp

# ===========================
# LIME EXPLAINER
# ===========================
def print_lime_explanation(explanation, model, X_test):
    
    # Get prediction
    pred = model.predict([X_test.iloc[0]])[0]
    
    print("\nPrediction:", pred)
    print("\nTop words influencing prediction:")
    print("--------------------------------")
    
    for word, weight in explanation.as_list():
        influence = "strong" if abs(weight) > 0.3 else "moderate"
        
        print(f"{word:<12} → {weight:+.3f} ({influence} influence)")

# ===========================
# MAIN
# ===========================
if __name__ == "__main__":
    
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    model = create_stacking_model()
    pipeline = create_pipeline(model)
    
    trained_model = train_model(pipeline, X_train, y_train)
    
    acc = evaluate_model(trained_model, X_test, y_test)
    print("Accuracy:", acc)
    
    explanation = explain_with_lime(trained_model, X_train, X_test)
    print_lime_explanation(explanation, trained_model, X_test)