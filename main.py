import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import nltk
import joblib
from nltk.corpus import stopwords
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm 
tqdm.pandas()

# Load dataset
df = pd.read_csv('bestsellers.csv')

# the spreadsheet
print(df.head())
print(df.shape)
print(df.columns)
print(df.describe())

#clean data
df.drop_duplicates(inplace=True)
df.rename(columns={"Name": "Title", "Year": "Publication Year", "User Rating": "Rating"}, inplace=True)
df["Price"] = df["Price"].astype(float)

# NLP Keyword Analysis
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_tokens(title):
    tokens = re.findall(r"\b\w+\b", title.lower())  # only words
    return [t for t in tokens if t not in stop_words and len(t) > 2]

df["title_tokens"] = df["Title"].progress_apply(clean_tokens)

# count most common words
all_tokens = [token for tokens in df["title_tokens"] for token in tokens]
common_words = Counter(all_tokens).most_common(20)
print("Top 20 keywords:", common_words)

#average rating & review per keyword
keyword_stats = {}

for word, _ in common_words:
    mask = df["title_tokens"].apply(lambda tokens: word in tokens)
    avg_rating = df.loc[mask, "Rating"].mean()
    avg_reviews = df.loc[mask, "Reviews"].mean()
    keyword_stats[word] = {"avg_rating": avg_rating, "avg_reviews": avg_reviews}

keyword_df = pd.DataFrame(keyword_stats).T.sort_values("avg_reviews", ascending=False)
print(keyword_df)

#correlation analysis
correlations = df[["Rating", "Reviews", "Price", "Publication Year"]].corr()  # MOVED UP
print("\nCorrelation Matrix:")
print(correlations)

# Visualization
plt.figure(figsize=(10,6))
sns.barplot(x=keyword_df.index, y="avg_reviews", data=keyword_df)
plt.xticks(rotation=45)
plt.title("Average Reviews by Keyword in Book Titles")
plt.savefig('keyword_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(correlations, annot=True, cmap='coolwarm', center=0) 
plt.title("Correlation Matrix")
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

#---------------
# --- Trends over time ---
df["Decade"] = (df["Publication Year"] // 10) * 10
avg_price_by_decade = df.groupby("Decade")["Price"].mean()
avg_rating_by_decade = df.groupby("Decade")["Rating"].mean()

print("\nAverage Price by Decade:")
print(avg_price_by_decade)
print("\nAverage Rating by Decade:")
print(avg_rating_by_decade)

avg_rating_by_genre = df.groupby("Genre")["Rating"].mean()
print("\nAverage rating by genre:")
print(avg_rating_by_genre)

# Predictive Analytics 
df["Highly_Rated"] = (df["Rating"] >= 4.5).astype(int)

# Encode genre
le_genre = LabelEncoder()
df["Genre_encoded"] = le_genre.fit_transform(df["Genre"])

X = df[["Price", "Reviews", "Publication Year", "Genre_encoded"]]
y = df["Highly_Rated"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
print("\n🔮 Predictive Analytics Results 🔮")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

joblib.dump(model, 'book_rating_predictor.joblib') 
joblib.dump(le_genre, 'genre_encoder.joblib')
print("Model saved as 'book_rating_predictor.joblib'")


summary_text = f"""
I analyzed Amazon Bestsellers data (2009–2019). 
Key correlations found:
- Reviews vs Rating correlation: {correlations.loc['Reviews','Rating']:.2f}
- Price vs Rating correlation: {correlations.loc['Price','Rating']:.2f}
- Year vs Price correlation: {correlations.loc['Publication Year','Price']:.2f}
- Year vs Rating correlation: {correlations.loc['Publication Year','Rating']:.2f}

Average Price by Decade: {avg_price_by_decade.to_dict()}
Average Rating by Decade: {avg_rating_by_decade.to_dict()}
Keyword trends show: {keyword_df.head(5).to_dict()}
"""

print("\nText to summarize:\n", summary_text)

#AI summarization 
try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    ai_summary = summarizer(summary_text, max_length=120, min_length=50, do_sample=False)[0]["summary_text"]
    print("\n📊 Final AI Report:")
    print(ai_summary)
except Exception as e:
    print(f"\n⚠️  Summarization failed: {e}")
    print("Using fallback summary...")
    ai_summary = summary_text[:200] + "..."  
    print("Fallback Summary:", ai_summary)

# Export data
keyword_df.to_csv('keyword_analysis.csv')
correlations.to_csv('correlations.csv')
pd.DataFrame(common_words, columns=["keyword", "count"]).to_csv("top_keywords.csv", index=False)
avg_price_by_decade.to_csv("avg_price_by_decade.csv")
avg_rating_by_decade.to_csv("avg_rating_by_decade.csv")
avg_rating_by_genre.to_csv("avg_rating_by_genre.csv")

print("\n✅ Analysis complete! All files have been saved.")
