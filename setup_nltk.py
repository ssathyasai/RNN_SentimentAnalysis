"""Download required NLTK data — run once before starting the app."""
import nltk
nltk.download('stopwords')
nltk.download('punkt')
print("NLTK data downloaded successfully.")
