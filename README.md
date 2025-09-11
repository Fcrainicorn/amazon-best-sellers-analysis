# 📚 Amazon Bestsellers Analysis (2009–2019)

This project analyzes **Amazon’s Top 50 Bestselling Books from 2009–2019**, uncovering insights about ratings, reviews, pricing, and keyword trends.  

It combines **data cleaning, visualization, NLP keyword extraction, and predictive modeling** to explore what makes books successful — with an **AI-powered natural language summary** at the end.  



## ✨ Key Features

- 🧹 **Data Cleaning**  
  - Removed duplicates, standardized columns, and converted price to numeric.  
  - Added new features (decade buckets, encoded genres).  

- 📝 **NLP Keyword Analysis**  
  - Tokenized book titles using regex + NLTK stopwords.  
  - Extracted the **Top 20 keywords** and analyzed their relationship with ratings & reviews.  

- 📊 **Data Visualization**  
  - Bar charts of average reviews by keyword.  
  - Correlation heatmaps between ratings, reviews, prices, and publication year.  

- ⏳ **Trends Over Time**  
  - Average **price and rating by decade**.  
  - Genre-level comparisons of average ratings.  

- 🤖 **Predictive Analytics**  
  - Built a **Decision Tree Classifier** to predict if a book is **highly rated (≥ 4.5)**.  
  - Evaluated accuracy and classification metrics.  

- 🧠 **AI-Generated Summary**  
  - Used Hugging Face’s `facebook/bart-large-cnn` model to produce a **natural language report** of findings.  

- 📂 **Exported Results**  
  - Saved analysis outputs as CSVs:  
    - `keyword_analysis.csv`  
    - `correlations.csv`  
    - `top_keywords.csv`  
    - `avg_price_by_decade.csv`  
    - `avg_rating_by_decade.csv`  
    - `avg_rating_by_genre.csv`  



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

- **Top Keywords**: Words like *girl*, *life*, and *story* appear most often in bestselling titles.  
- **Correlations**: Number of reviews is more strongly linked to rating than price.  
- **Trends**: Average book prices declined after 2010, while ratings stayed high.  
- **Predictive Model**: Achieved solid accuracy in classifying highly rated books.  
- **AI Report**: Automatically summarizes correlations, keyword trends, and price/rating shifts over time.  




## 🛠 Tech Stack  

- **Python Libraries**: pandas, matplotlib, seaborn, scikit-learn, joblib, tqdm  
- **NLP**: NLTK, Hugging Face Transformers  
- **Machine Learning**: Decision Tree Classifier  
- **Deep Learning Backend**: PyTorch (via `transformers`)  




## 📜 Dataset  

Dataset sourced from Kaggle: [Amazon Top 50 Bestselling Books 2009–2019](https://www.kaggle.com/datasets/sootersaalu/amazon-top-50-bestselling-books-2009-2019?resource=download).  





