import pandas as pd
import arff
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import Normalizer
import joblib
from sklearn.pipeline import Pipeline

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the ARFF file
with open('train.arff', 'r') as f:
    data = arff.load(f)

# Extract data and attributes
data_list = data['data']
attributes = [attr[0] for attr in data['attributes']]

# Convert to a Pandas DataFrame
df = pd.DataFrame(data_list, columns=attributes)

# Display the first few rows of the DataFrame
print(df.head())

# Preprocessing function
def preprocess_text(text):
    # Decode byte string if necessary
    if isinstance(text, bytes):
        text = text.decode('utf-8')
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Lowercasing
    tokens = [token.lower() for token in tokens]
    
    # Remove punctuation
    tokens = [token for token in tokens if token not in string.punctuation]
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Return preprocessed text
    return ' '.join(tokens)

# Apply preprocessing to the text column
df['processed_text'] = df['text'].apply(preprocess_text)


# Init N gram Vectorizer
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer='char',use_idf=True)
X = tfidf_vectorizer.fit_transform(df['processed_text'])
y = df['label']


# Create a pipeline with normalization and classifier 
pipeline = Pipeline([ ('normalizer', Normalizer(norm='l2')),('classifier', MultinomialNB(alpha=3.0)) ]) 

# Initialiser StratifiedKFold
k = 5
skf = StratifiedKFold(n_splits=k)

# Initialiser les listes pour stocker les résultats
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

# Tracer la ligne de chance
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.8)

# Boucle sur chaque split pour calculer les courbes ROC
for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train the model 
    pipeline.fit(X_train, y_train) 

    # Calculer les scores de prédiction
    y_score = pipeline.predict_proba(X_test)[:, 0]

    # Calculer les ROC et AUC
    fpr, tpr, _ = roc_curve(y_test, y_score,pos_label= 'G')
    roc_auc = auc(fpr, tpr)

    # Ajouter les résultats
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=2, alpha=0.3, label=f'Fold {i + 1} (AUC = {roc_auc:.2f})')

    # Interpolation pour la courbe ROC moyenne
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0

# Calculer la moyenne et l'écart-type des TPRs
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)

# Tracer la courbe ROC moyenne
plt.plot(mean_fpr, mean_tpr, color='b', linestyle='-', label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})', lw=2)

# Calculer l'écart-type de la courbe ROC
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

# Tracer l'ombre de l'écart-type
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2, label='± 1 écart-type')

# Ajouter les légendes et titres
plt.xlabel('Taux de Faux Positifs')
plt.ylabel('Taux de Vrais Positifs')
plt.title('Courbe ROC avec k-fold Cross-Validation (Classification Binaire)')
plt.legend(loc='lower right')
plt.show()


# Save the raw data to a DataFrame
roc_data = pd.DataFrame({"FPR": mean_fpr, "TPR": mean_tpr})

# Save the DataFrame to a CSV file
roc_data.to_csv("roc_data.csv", index=False)

print("ROC data saved successfully to 'roc_data.csv'!")

# Save the model to a file
joblib.dump(pipeline, 'naive_bayes_model.joblib')

# Save the TF-IDF vectorizer
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')

print("Model and vectorizer saved successfully!")
