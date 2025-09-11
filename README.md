# 📚 Amazon Bestsellers Analysis (2009–2019)

This project explores **550 books from Amazon's bestseller list (2009–2019)**, uncovering trends in ratings, reviews, prices, and genres.  
It combines **data cleaning, visualization, NLP keyword analysis, and predictive modeling** to generate insights about what makes a book highly rated.



## ✨ Key Features
- 🔍 **Data Cleaning & Exploration**  
  - Removed duplicates, standardized column names, and transformed price data.  
  - Produced descriptive statistics and dataset overview.  

- 📝 **NLP Keyword Analysis**  
  - Tokenized book titles and removed stopwords using NLTK.  
  - Extracted the top 20 most common keywords and analyzed their impact on ratings & reviews.  

- 📊 **Visualization**  
  - Heatmaps for correlations between ratings, reviews, price, and publication year.  
  - Bar charts of keyword popularity and trends over time.  

- ⏳ **Trends Over Time**  
  - Price and rating averages by decade.  
  - Ratings compared by genre (fiction vs. nonfiction).  

- 🤖 **Predictive Analytics**  
  - Built a Decision Tree classifier to predict whether a book will be **highly rated (≥ 4.5)**.  
  - Model saved as `book_rating_predictor.joblib`.  

- 🧠 **AI-Generated Summary**  
  - Used Hugging Face’s `bart-large-cnn` model to generate a natural language summary of insights.
 
---

## ▶️ How to Run
1. Clone the repository:
   
   ```bash
   git clone https://github.com/Fcrainicorn/amazon-bestsellers-analysis.git
   
   cd amazon-bestsellers-analysis
   
2. Install dependencies:

   ```bash

   pip install -r requirements.txt
   ```
3. Run the analysis:

   ```bash

   python main.py
   ```
## 📈 Example Insights

Top Keywords: words like girl, life, love, and story are most common in bestseller titles.

Correlation: number of reviews has a stronger relationship with ratings than price.

Trends: average book prices declined slightly after 2010, while ratings stayed consistently high.

Predictive Model: The model achieved strong baseline accuracy in predicting highly rated books.

## 🛠 Tech Stack

Python: pandas, seaborn, matplotlib, scikit-learn, joblib, tqdm 

NLP: NLTK, Hugging Face Transformers

Machine Learning: Decision Tree Classifier

Visualization: Seaborn, Matplotlib

### 🔮 Next Steps

Test additional models (Random Forest, XGBoost) for improved predictive accuracy.

Build a lightweight dashboard with Streamlit for interactive exploration.

Expand analysis to include newer Amazon bestseller datasets.

## 📜 Acknowledgements
Dataset sourced from Kaggle: [Amazon Top 50 Bestselling Books 2009–2019](https://www.kaggle.com/datasets/sootersaalu/amazon-top-50-bestselling-books-2009-2019?resource=download).



