
#Importing the required libraries like pandas,nltk,.......
import pandas as pd
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import swifter # Import swifter to train the large data and speed up the pandas operations

# Download stopwords if not already present
try:
    # Check if stopwords are already downloaded
    nltk.data.find('corpora/stopwords')
except LookupError:
    # If not found, download stopwords
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')
except nltk.download.DownloadError:
    # Catch potential DownloadError during download
    print("Error downloading NLTK stopwords.")
    raise # Re-raise the exception after printing

# Load dataset
data = pd.read_csv("spam_Emails_data.csv", encoding="latin-1")

# Rename your actual columns here if different:
data = data[['label', 'text']]  # Update these if needed
data.columns = ['label', 'text']

# --- Add inspection here ---
print("Original label value counts:")
print(data['label'].value_counts())
print("-" * 20)
# Convert labels to binary
# Convert label column to lowercase before mapping to handle case variations
data['label'] = data['label'].str.lower()
# The .map() function will return NaN for any values not in the dictionary.
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Drop rows where the label became NaN after mapping.
# This is crucial to handle unexpected values in the original 'label' column.
# Check the number of rows before and after dropping NaNs
initial_rows = len(data)
data = data.dropna(subset=['label'])
rows_after_dropping = len(data)
print(f"Dropped {initial_rows - rows_after_dropping} rows due to unexpected labels.")
print("-" * 20)
print("Label value counts after mapping and dropping NaNs:")
print(data['label'].value_counts())
print("-" * 20)

# Check if any rows remain
if data.empty:
    raise ValueError("No rows remain in the DataFrame after cleaning labels. Check your original labels.")


# Ensure the label column is of integer type after dropping NaNs
data['label'] = data['label'].astype(int)


# NLP Preprocessing Function
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Ensure input is a string, handle potential non-string types
    if not isinstance(text, str):
        return "" # Return an empty string or handle as appropriate

    # Lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^a-z]', ' ', text)
    # Tokenize and remove stopwords
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# Apply preprocessing with swifter
# Add .astype(str) to handle potential non-string entries in the 'text' column
data['processed_text'] = data['text'].astype(str).swifter.apply(preprocess_text) # Use swifter here


# Drop any empty or null entries after preprocessing
initial_rows_processed = len(data)
data = data[data['processed_text'].notnull()]
data = data[data['processed_text'].str.strip() != '']
rows_after_processed_dropping = len(data)
print(f"Dropped {initial_rows_processed - rows_after_processed_dropping} rows due to empty processed text.")
print("-" * 20)

# Check if any rows remain after text processing
if data.empty:
     raise ValueError("No rows remain in the DataFrame after text preprocessing. Check your text cleaning steps or input data.")

# Feature extraction using CountVectorizer
vectorizer = CountVectorizer()
# Add a check to ensure there are documents to vectorize
if len(data['processed_text']) == 0:
     raise ValueError("No documents available for vectorization after all filtering steps.")
X = vectorizer.fit_transform(data['processed_text'])
y = data['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train SVM model
model_svm = svm.SVC(kernel='linear')
model_svm.fit(X_train, y_train)

# Evaluate
y_pred = model_svm.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

