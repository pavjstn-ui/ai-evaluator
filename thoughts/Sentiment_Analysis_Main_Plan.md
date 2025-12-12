# SLOVAK/CZECH SENTIMENT ANALYSIS PRODUCT
## 10-Week Learning Plan - Income + Job Skills

Complete roadmap to build a production sentiment analysis system for Slovak/Czech text. Learn enterprise-grade NLP skills while building a product you can sell.

---

## **PROJECT OVERVIEW**

**What you're building:**
- Sentiment analyzer for Slovak/Czech product reviews, social media, business feedback
- API that businesses can integrate
- Dashboard showing sentiment trends
- Revenue-generating SaaS product

**Learning outcomes:**
- Production Hugging Face transformers
- Fine-tuning on Czech/Slovak text
- RAG and vector databases (FAISS)
- Data pipelines (ETL)
- Cloud deployment (Azure/AWS)
- Product building and selling

**Job market fit:** 95%+ for Czech NLP positions
**Timeline:** 10 weeks intensive
**Income potential:** €500-2000/month (selling to SMEs)
**Interview ready:** After Week 8

---

## **PHASE 1: WEEKS 1-2 - HUGGING FACE FUNDAMENTALS**

Learn transformers. Build your first English sentiment classifier. Adapt to Slovak/Czech.

---

### **WEEK 1: Setup + English Sentiment Baseline**

#### 1.1 - Environment Setup (Day 1)

**Install Python and libraries:**
```bash
# Create virtual environment
python3 -m venv sentiment_env
source sentiment_env/bin/activate  # On Windows: sentiment_env\Scripts\activate

# Install core libraries
pip install transformers torch datasets evaluate scikit-learn pandas numpy jupyter

# Verify installation
python -c "import torch; print(torch.__version__)"
python -c "from transformers import pipeline; print('OK')"
```

**GPU setup (optional but recommended):**
```bash
# Check if GPU available
python -c "import torch; print(torch.cuda.is_available())"

# If no GPU: CPU is fine for learning, GPU for production
# If you have NVIDIA GPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Resources:**
- PyTorch installation: https://pytorch.org/get-started/locally/
- Transformers docs: https://huggingface.co/docs/transformers/

#### 1.2 - Learn Transformer Basics (Days 1-2)

**Read (3 hours):**
- Hugging Face course Chapter 1: https://huggingface.co/course/chapter1
- Focus on: "Transformers, what can they do?" and "How do transformers work?"

**Key concepts to understand:**
- What is a transformer?
- Tokenization (converting text to numbers)
- Attention mechanisms (why transformers work)
- Pre-trained vs fine-tuned models

**Quick intro video:**
- https://www.youtube.com/watch?v=BqH2rD2jP5U (10 min overview)

#### 1.3 - First Sentiment Classifier (Days 2-3)

**Create notebook:** `sentiment_analysis.ipynb`

**Code to run:**
```python
# Step 1: Load pre-trained model
from transformers import pipeline

# Simple English sentiment (baseline)
classifier = pipeline("sentiment-analysis", 
                     model="distilbert-base-uncased-finetuned-sst-2-english")

# Test it
test_sentences = [
    "This product is amazing!",
    "Terrible quality, waste of money",
    "It's okay, nothing special",
    "Best purchase ever",
    "Broken after one day"
]

for sentence in test_sentences:
    result = classifier(sentence)
    print(f"Text: {sentence}")
    print(f"Sentiment: {result[0]['label']}, Score: {result[0]['score']:.4f}")
    print()
```

**Expected output:**
```
Text: This product is amazing!
Sentiment: POSITIVE, Score: 0.9998

Text: Terrible quality, waste of money
Sentiment: NEGATIVE, Score: 0.9994
```

**Understand the output:**
- `label`: POSITIVE or NEGATIVE
- `score`: Confidence (0-1), how sure the model is

**Task 1:** Run the above. Test with 10 of your own sentences.

**Resources:**
- Hugging Face pipeline API: https://huggingface.co/docs/transformers/main/en/task_summary#sentiment-analysis
- Pre-trained English models: https://huggingface.co/models?pipeline_tag=text-classification&sort=likes

#### 1.4 - Understand Different Models (Day 3)

Different models, different trade-offs:

```python
# Model 1: Fast, small, good for real-time
model1 = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
# Size: 268MB, Speed: Fast, Accuracy: 90%

# Model 2: Slower, larger, better accuracy
model2 = pipeline("sentiment-analysis", model="roberta-large-mnli")
# Size: 1.6GB, Speed: Slow, Accuracy: 95%

# Model 3: Multilingual (works in many languages including Czech)
model3 = pipeline("sentiment-analysis", model="xlm-roberta-base")
# Size: 558MB, Speed: Medium, Accuracy: 88%

# Test all three
text = "Product is excellent"
for model_name, model in [("DistilBERT", model1), ("RoBERTa", model2), ("XLM-R", model3)]:
    result = model(text)
    print(f"{model_name}: {result[0]['label']} ({result[0]['score']:.4f})")
```

**For this project:** Start with `distilbert-base-uncased-finetuned-sst-2-english` (fast, good accuracy).

**Task 2:** Test 3 different models. Record speed and accuracy differences. Save in notebook.

---

### **WEEK 2: Czech/Slovak Sentiment + Fine-Tuning**

#### 2.1 - Multilingual Models (Days 1-2)

Pre-trained English won't work well for Czech/Slovak. Use multilingual models.

**Available options:**

```python
# Option 1: XLM-RoBERTa (good multilingual)
model = pipeline("sentiment-analysis", model="xlm-roberta-base")

# Test Czech
czech_texts = [
    "Výborný produkt, velmi spokojený",  # Excellent product, very happy
    "Hrozná kvalita, otrava peněz",  # Terrible quality, waste of money
    "Průměrné, nic moc zvláštního",  # Average, nothing special
]

for text in czech_texts:
    result = model(text)
    print(f"Czech: {text} → {result[0]['label']}")
```

**Problem:** XLM-RoBERTa is trained on 100+ languages, so it's okay but not great for Czech specifically.

**Better approach:** Find Czech-specific model or fine-tune on Czech data.

**Czech/Slovak models available:**
- https://huggingface.co/models?language=cs (Czech)
- https://huggingface.co/models?language=sk (Slovak)

Look for:
- `sentiment-analysis` tag
- Published by Czech/Slovak researchers
- Recent upload (2023+)

**Example Czech model:**
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "bert-base-multilingual-cased"  # Generic multilingual
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Or search HF Hub for Czech-specific
# https://huggingface.co/models?language=cs&pipeline_tag=text-classification
```

**Task 1:** Find one Czech and one Slovak sentiment model on Hugging Face Hub. Test them. Compare accuracy.

**Resources:**
- Czech models: https://huggingface.co/models?language=cs
- Slovak models: https://huggingface.co/models?language=sk
- XLM-RoBERTa: https://huggingface.co/xlm-roberta-base

#### 2.2 - Fine-tune on Czech Product Reviews (Days 3-5)

Pre-trained models are okay. Fine-tuning makes them great.

**Step 1: Create training data (1 day)**

Collect or create 200-500 Czech product review snippets with sentiment labels.

```python
# Create training data
czech_training_data = [
    ("Výborná kvalita, doporučuji", 1),  # 1 = positive
    ("Špatně funguje, vrátil jsem", 0),  # 0 = negative
    ("Průměrný produkt", 0),  # 0 = neutral/negative
    ("Přesně jak jsem čekal, spokojený", 1),
    ("Hned se rozbilo", 0),
    ("Skvělá cena za kvalitu", 1),
    ("Nesouhlasí s popisem", 0),
    ("Rychlý handling, dobrý produkt", 1),
]

# Save as CSV
import pandas as pd
df = pd.DataFrame(czech_training_data, columns=['text', 'label'])
df.to_csv('czech_reviews.csv', index=False)
print(f"Created {len(df)} training examples")
```

**Where to get data:**
- Heureka.cz comments (scrape with BeautifulSoup)
- Amazon.cz reviews
- Mall.cz reviews
- Local Czech e-commerce sites
- Create manually (200 reviews takes 1-2 hours)

**Data format needed:**
```
text,label
"Výborný produkt",1
"Hrozná kvalita",0
```

**Task 1:** Create or collect 300+ Czech product reviews with sentiment labels. Save to CSV.

**Step 2: Fine-tune model (Days 3-5)**

```python
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Load data
df = pd.read_csv('czech_reviews.csv')
dataset = Dataset.from_pandas(df)

# Split train/test
dataset = dataset.train_test_split(test_size=0.2)

# Load pre-trained model
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir='./czech_sentiment_model',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
)

trainer.train()

# Save model
trainer.save_model('./czech_sentiment_model')
print("Model saved!")
```

**What's happening:**
1. Load pre-trained XLM-RoBERTa
2. Add a classification head (2 labels: positive/negative)
3. Train on your Czech data
4. Model learns Czech-specific sentiment patterns

**Time to train:** 5-15 minutes on CPU, 1-2 minutes on GPU

**Task 2:** Run fine-tuning. Save model. Test on unseen Czech sentences.

**Resources:**
- Fine-tuning guide: https://huggingface.co/docs/transformers/training
- Training arguments: https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments

#### 2.3 - Test Fine-tuned Model (Day 5)

```python
from transformers import pipeline

# Load your fine-tuned model
classifier = pipeline(
    "sentiment-analysis",
    model="./czech_sentiment_model",
    device=0  # GPU if available, -1 for CPU
)

# Test on new Czech text
test_czech = [
    "Nejlepší nákup roku!",  # Best purchase of the year!
    "Zcela zklamaný, nefunguje",  # Completely disappointed, doesn't work
    "Dobrý poměr ceny a kvality",  # Good price-to-quality ratio
    "Přesně co jsem očekával",  # Exactly what I expected
]

for text in test_czech:
    result = classifier(text)
    print(f"Text: {text}")
    print(f"Sentiment: {result[0]['label']}, Confidence: {result[0]['score']:.4f}\n")
```

**Expected:** Your fine-tuned model should be more accurate on Czech text than the generic multilingual model.

**Task 3:** Compare:
- Generic XLM-RoBERTa vs your fine-tuned model
- Test on 20 Czech sentences
- Record accuracy for both
- Save comparison in notebook

**Milestone:** Week 2 complete - You have a working Czech sentiment classifier.

---

## **PHASE 2: WEEKS 3-4 - RETRIEVAL-AUGMENTED GENERATION (RAG)**

Add memory. Instead of classifying sentiment alone, retrieve similar past reviews for context.

---

### **WEEK 3: Vector Embeddings & FAISS**

#### 3.1 - Understand Embeddings (Days 1-2)

Embeddings = converting text to vectors (lists of numbers) that capture meaning.

**Concept:**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

text1 = "Výborný produkt, muito happy"  # Great product
text2 = "Skvělý nákup"  # Excellent purchase
text3 = "Hrozná kvalita"  # Terrible quality

# Convert to vectors
emb1 = model.encode(text1)  # 768-dim vector
emb2 = model.encode(text2)  # 768-dim vector
emb3 = model.encode(text3)  # 768-dim vector

print(f"Embedding shape: {emb1.shape}")  # (768,)

# Similar texts = similar vectors
from scipy.spatial.distance import cosine
dist_1_2 = cosine(emb1, emb2)  # ~0.1 (close, both positive)
dist_1_3 = cosine(emb1, emb3)  # ~0.7 (far, opposite sentiment)

print(f"Distance 1-2: {dist_1_2:.3f}")
print(f"Distance 1-3: {dist_1_3:.3f}")
```

**Install:**
```bash
pip install sentence-transformers scipy
```

**Task 1:** Create embeddings for 20 Czech product reviews. Verify: positive reviews have low distance to each other, far from negative reviews.

**Resources:**
- Sentence Transformers: https://www.sbert.net/
- How embeddings work: https://www.youtube.com/watch?v=iY2AZYdZAVc (10 min video)

#### 3.2 - FAISS Vector Database (Days 2-4)

FAISS = store embeddings, search them fast.

**Install:**
```bash
pip install faiss-cpu  # or faiss-gpu if GPU available
```

**Basic usage:**
```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# Your Czech reviews
reviews = [
    {"id": 1, "text": "Výborný produkt", "sentiment": "positive"},
    {"id": 2, "text": "Hrozná kvalita", "sentiment": "negative"},
    {"id": 3, "text": "Dobrý nákup", "sentiment": "positive"},
    {"id": 4, "text": "Nefunguje", "sentiment": "negative"},
]

# Create embeddings
embeddings = embedding_model.encode([r['text'] for r in reviews]).astype('float32')
print(f"Embeddings shape: {embeddings.shape}")  # (4, 768)

# Create FAISS index
dimension = 768
index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean)
index.add(embeddings)

print(f"Index trained with {index.ntotal} vectors")

# Search: find similar reviews
query = "Výborný, doporučuji"  # Great, recommend
query_emb = embedding_model.encode([query]).astype('float32')

distances, indices = index.search(query_emb, k=2)  # Find top 2

print(f"\nQuery: {query}")
print(f"Top 2 similar reviews:")
for i, idx in enumerate(indices[0]):
    print(f"  {i+1}. {reviews[idx]['text']} (distance: {distances[0][i]:.3f})")
```

**Output:**
```
Query: Výborný, doporučuji
Top 2 similar reviews:
  1. Výborný produkt (distance: 0.245)
  2. Dobrý nákup (distance: 1.123)
```

**Task 1:** Create FAISS index with 100 Czech reviews. Test queries. Verify similarity makes sense.

**Save/Load index:**
```python
# Save
faiss.write_index(index, 'czech_reviews.index')

# Load later
index = faiss.read_index('czech_reviews.index')
```

**Resources:**
- FAISS tutorial: https://github.com/facebookresearch/faiss/wiki/Tutorials
- Vector search explained: https://www.pinecone.io/learn/vector-search/

#### 3.3 - RAG for Sentiment (Day 4-5)

Use FAISS to find similar past reviews, use them to inform current sentiment.

```python
class SentimentAnalyzerWithRAG:
    def __init__(self, model_path, faiss_index_path):
        # Load fine-tuned sentiment model
        self.classifier = pipeline(
            "sentiment-analysis",
            model=model_path,
            device=0
        )
        # Load embeddings model
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        # Load FAISS index
        self.index = faiss.read_index(faiss_index_path)
        self.reviews = self.load_reviews()  # Metadata
    
    def analyze_with_context(self, new_review):
        """Analyze sentiment AND retrieve similar reviews"""
        
        # 1. Get sentiment of new review
        sentiment_result = self.classifier(new_review)
        base_sentiment = sentiment_result[0]['label']
        base_confidence = sentiment_result[0]['score']
        
        # 2. Find similar reviews
        query_emb = self.embedding_model.encode([new_review]).astype('float32')
        distances, indices = self.index.search(query_emb, k=3)
        
        similar_reviews = [self.reviews[idx] for idx in indices[0]]
        
        # 3. Check if context agrees
        context_sentiments = [r['sentiment'] for r in similar_reviews]
        context_agreement = context_sentiments.count(base_sentiment) / len(context_sentiments)
        
        # 4. Return with context
        return {
            'review': new_review,
            'sentiment': base_sentiment,
            'confidence': base_confidence,
            'context_agreement': context_agreement,
            'similar_reviews': similar_reviews[:2],  # Top 2
            'final_confidence': base_confidence * 0.7 + context_agreement * 0.3
        }

# Usage
analyzer = SentimentAnalyzerWithRAG(
    model_path='./czech_sentiment_model',
    faiss_index_path='czech_reviews.index'
)

new_review = "Skvělý nákup, velmi spokojen"
result = analyzer.analyze_with_context(new_review)
print(f"Review: {result['review']}")
print(f"Sentiment: {result['sentiment']}")
print(f"Base confidence: {result['confidence']:.3f}")
print(f"Context agreement: {result['context_agreement']:.1%}")
print(f"Final confidence: {result['final_confidence']:.3f}")
print(f"Similar reviews: {result['similar_reviews']}")
```

**What's happening:**
1. Classify new review with fine-tuned model
2. Find 3 similar past reviews via FAISS
3. Check if similar reviews agree with classification
4. Boost confidence if context supports it

**Task 2:** Implement RAG-enhanced sentiment analyzer. Test on 10 new reviews. Record how often context helps.

**Milestone:** Week 3 - RAG sentiment analysis working.

---

### **WEEK 4: LangChain Integration**

Wire RAG components together with LangChain.

#### 4.1 - LangChain Basics (Days 1-2)

LangChain = framework for building LLM applications.

**Install:**
```bash
pip install langchain langchain-community langchain-openai
```

**Simple example:**
```python
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# 1. Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name='paraphrase-multilingual-mpnet-base-v2'
)

# 2. Create vector store from your reviews
# (You'd load existing FAISS index or create from documents)
documents = [
    {'page_content': review['text'], 'metadata': {'sentiment': review['sentiment']}}
    for review in reviews
]

vectorstore = FAISS.from_documents(documents, embeddings)

# 3. Create retriever
retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

# 4. Use it
query = "Výborný produkt"
similar = retriever.get_relevant_documents(query)
print(f"Retrieved {len(similar)} similar reviews:")
for doc in similar:
    print(f"  - {doc.page_content}")
```

**Task 1:** Build LangChain retrieval setup. Test with 5 queries.

**Resources:**
- LangChain docs: https://python.langchain.com/docs/get_started
- Retrieval tutorial: https://python.langchain.com/docs/use_cases/question_answering/

#### 4.2 - Full Pipeline with LLM (Days 3-5)

Add LLM to generate explanations alongside sentiment.

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Setup (you'll need OpenAI API key)
llm = ChatOpenAI(
    model_name='gpt-3.5-turbo',
    temperature=0.7,
    api_key='sk-...'  # Your OpenAI key
)

# Create prompt
prompt_template = """Based on the review and similar past reviews, analyze sentiment.

REVIEW: {question}

CONTEXT (Similar past reviews):
{context}

Provide:
1. Sentiment (positive/negative/neutral)
2. Confidence (0-100%)
3. Key reasons
4. Comparison to similar reviews"""

prompt = PromptTemplate(
    input_variables=['question', 'context'],
    template=prompt_template
)

# Create chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=retriever,
    return_source_documents=True
)

# Use it
review = "Výborný produkt, doporučuji všem"
result = qa_chain.invoke({'query': review})

print("Sentiment Analysis with Context:")
print(result['result'])
print("\nContext sources:")
for doc in result['source_documents']:
    print(f"  - {doc.page_content}")
```

**Task 2:** Setup OpenAI API (free trial includes $5 credit). Run LLM-enhanced sentiment analysis.

**Note:** This costs money (GPT-3.5 is cheap, ~$0.001 per query).

**Milestone:** Week 4 - Full RAG + LLM pipeline working.

---

## **PHASE 3: WEEKS 5-6 - DATA PIPELINES & AUTOMATION**

Get real data. Build pipelines to process it continuously.

---

### **WEEK 5: Data Scraping**

#### 5.1 - Identify Data Sources (Day 1)

Czech/Slovak review sources:
- **Heureka.cz** - Largest Czech e-commerce reviews (900k+ products)
- **Mall.cz** - Product reviews
- **Alza.cz** - Comments
- **Amazon.cz** - Product reviews
- **TripAdvisor.cz** - Business/restaurant reviews
- **Google Reviews** - Local business

**Focus:** Heureka (easiest to scrape, most data)

#### 5.2 - Build Heureka Scraper (Days 2-4)

```bash
pip install beautifulsoup4 requests
```

```python
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime

class HeurekaScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def scrape_product_reviews(self, product_url, max_pages=3):
        """Scrape reviews from single product"""
        reviews = []
        
        for page in range(1, max_pages + 1):
            url = f"{product_url}?page={page}"
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find review containers (adjust selector based on current HTML)
                review_items = soup.find_all('div', class_='review')
                
                for item in review_items:
                    # Extract review text
                    review_text = item.find('p', class_='review-text')
                    rating = item.find('span', class_='rating')
                    author = item.find('span', class_='author')
                    
                    if review_text:
                        reviews.append({
                            'text': review_text.text.strip(),
                            'rating': int(rating.text) if rating else None,
                            'author': author.text if author else 'Unknown',
                            'date': datetime.now().isoformat(),
                            'source': 'Heureka'
                        })
                
                time.sleep(1)  # Be polite to server
                
            except Exception as e:
                print(f"Error on page {page}: {e}")
                continue
        
        return reviews

scraper = HeurekaScraper()

# Example: scrape reviews for a product
product_urls = [
    'https://www.heureka.cz/apple-iphone-14/',
    'https://www.heureka.cz/samsung-galaxy-a14/',
]

all_reviews = []
for url in product_urls:
    print(f"Scraping {url}...")
    reviews = scraper.scrape_product_reviews(url, max_pages=5)
    all_reviews.extend(reviews)
    print(f"  Got {len(reviews)} reviews")

# Save
df = pd.DataFrame(all_reviews)
df.to_csv('heureka_reviews.csv', index=False)
print(f"Total: {len(df)} reviews saved")
```

**Note:** Check Heureka's robots.txt and terms of service. Use respectful scraping (delays, user agent, etc.).

**Task 1:** Scrape 500+ Czech product reviews from one source. Save to CSV.

**Resources:**
- BeautifulSoup tutorial: https://www.crummy.com/software/BeautifulSoup/bs4/doc/
- Web scraping ethics: https://blog.apify.com/web-scraping-ethics/

#### 5.3 - ETL Pipeline (Days 4-5)

Extract → Transform → Load. Automate the process.

```python
import pandas as pd
from datetime import datetime
import os

class ReviewETLPipeline:
    def __init__(self, output_file='processed_reviews.csv'):
        self.output_file = output_file
        self.processed_reviews = []
    
    def extract(self, csv_file):
        """Load raw data"""
        df = pd.read_csv(csv_file)
        print(f"Extracted {len(df)} reviews")
        return df
    
    def transform(self, df):
        """Clean and process"""
        # Remove duplicates
        df = df.drop_duplicates(subset=['text'])
        
        # Remove short reviews
        df = df[df['text'].str.len() > 20]
        
        # Remove special characters
        df['text'] = df['text'].str.replace(r'[^\w\s]', '', regex=True)
        
        # Normalize whitespace
        df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True).str.strip()
        
        # Convert rating to sentiment label
        df['sentiment'] = df['rating'].apply(lambda x: 
            'positive' if x >= 4 else 'negative' if x <= 2 else 'neutral'
        )
        
        print(f"After cleaning: {len(df)} reviews")
        print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
        
        return df
    
    def load(self, df):
        """Save processed data"""
        df.to_csv(self.output_file, index=False)
        print(f"Saved to {self.output_file}")

# Run pipeline
pipeline = ReviewETLPipeline()
raw_df = pipeline.extract('heureka_reviews.csv')
clean_df = pipeline.transform(raw_df)
pipeline.load(clean_df)
```

**Automate daily (using APScheduler):**

```bash
pip install schedule
```

```python
import schedule
import time

def run_pipeline():
    print(f"[{datetime.now()}] Running ETL pipeline...")
    pipeline = ReviewETLPipeline()
    # ... run extraction, transformation, loading
    print("Pipeline complete")

# Schedule daily at 9 AM
schedule.every().day.at("09:00").do(run_pipeline)

while True:
    schedule.run_pending()
    time.sleep(60)
```

**Task 2:** Build full ETL pipeline. Run daily. Verify data quality.

**Milestone:** Week 5 - Automated data collection.

---

### **WEEK 6: Cloud Deployment & API**

#### 6.1 - Choose Cloud Provider (Day 1)

**For Czech jobs:**
- **Azure** (Microsoft, popular in Czech Republic)
- **AWS** (most common globally)
- **Google Cloud** (good for data/ML)

Start with Azure (local advantage for Czech employers).

**Create free account:**
- Azure: https://azure.microsoft.com/en-us/free/
  - Free: €170 credit for 30 days
- AWS: https://aws.amazon.com/free/
  - Free: 12 months free tier

#### 6.2 - Build Flask API (Days 2-3)

```bash
pip install flask gunicorn
```

```python
# app.py
from flask import Flask, request, jsonify
import json
from transformers import pipeline

app = Flask(__name__)

# Load model
classifier = pipeline(
    "sentiment-analysis",
    model="./czech_sentiment_model",
    device=0
)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze sentiment of provided text"""
    try:
        data = request.json
        text = data.get('text')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Analyze
        result = classifier(text)
        
        return jsonify({
            'text': text,
            'sentiment': result[0]['label'],
            'confidence': float(result[0]['score']),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch', methods=['POST'])
def batch_analyze():
    """Analyze multiple reviews"""
    try:
        data = request.json
        texts = data.get('texts', [])
        
        results = []
        for text in texts:
            result = classifier(text)
            results.append({
                'text': text,
                'sentiment': result[0]['label'],
                'confidence': float(result[0]['score'])
            })
        
        return jsonify({'results': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
```

**Test locally:**
```bash
python app.py
# In another terminal:
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Výborný produkt!"}'
```

**Expected response:**
```json
{
  "text": "Výborný produkt!",
  "sentiment": "POSITIVE",
  "confidence": 0.998,
  "timestamp": "2024-01-15T10:30:45.123456"
}
```

**Task 1:** Build API. Test locally. Verify all endpoints work.

**Resources:**
- Flask tutorial: https://flask.palletsprojects.com/
- RESTful API design: https://restfulapi.net/

#### 6.3 - Deploy to Azure/AWS (Days 4-5)

**Option A: Azure App Service (easiest)**

```bash
# Install Azure CLI
# Follow: https://learn.microsoft.com/en-us/cli/azure/install-azure-cli

# Login
az login

# Create resource group
az group create --name sentiment-rg --location eastus

# Create App Service plan
az appservice plan create --name sentiment-plan --resource-group sentiment-rg --sku B1

# Create app
az webapp create --resource-group sentiment-rg --plan sentiment-plan --name sentiment-analyzer-YOURNAME

# Deploy
az webapp up --resource-group sentiment-rg --name sentiment-analyzer-YOURNAME --runtime "PYTHON|3.11"
```

Your API is now live at: `https://sentiment-analyzer-YOURNAME.azurewebsites.net/analyze`

**Option B: Heroku (deprecated, use alternatives)**

**Option C: Docker + AWS/Azure**

Create Dockerfile:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

Build and push to cloud registry (AWS ECR or Azure Container Registry).

**Task 2:** Deploy API to cloud. Test with curl from public internet.

**Milestone:** Week 6 - Live API accessible globally.

---

## **PHASE 4: WEEKS 7-8 - DASHBOARDS & PRODUCT**

---

### **WEEK 7: Streamlit Dashboard**

Build interactive dashboard for demos and selling.

```bash
pip install streamlit streamlit-extras
```

```python
# dashboard.py
import streamlit as st
import pandas as pd
from datetime import datetime
import requests

st.set_page_config(page_title="Sentiment Analyzer", layout="wide")

st.title("🇨🇿 Czech/Slovak Sentiment Analysis")
st.write("Real-time sentiment analysis for your reviews and feedback")

# Sidebar
with st.sidebar:
    st.header("Settings")
    language = st.selectbox("Language", ["Czech", "Slovak", "Mixed"])
    batch_mode = st.checkbox("Batch mode (upload CSV)")

# Main content
if batch_mode:
    st.subheader("📁 Batch Analysis")
    uploaded_file = st.file_uploader("Upload CSV with reviews", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(df)} reviews")
        
        if st.button("Analyze"):
            with st.spinner("Analyzing..."):
                # Call API
                results = []
                for text in df['text']:
                    response = requests.post(
                        'http://localhost:5000/analyze',
                        json={'text': text}
                    )
                    results.append(response.json())
                
                results_df = pd.DataFrame(results)
                
                # Show results
                st.dataframe(results_df)
                
                # Summary stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    pos = (results_df['sentiment'] == 'POSITIVE').sum()
                    st.metric("Positive", f"{pos} ({pos/len(results_df):.0%})")
                with col2:
                    neg = (results_df['sentiment'] == 'NEGATIVE').sum()
                    st.metric("Negative", f"{neg} ({neg/len(results_df):.0%})")
                with col3:
                    avg_conf = results_df['confidence'].mean()
                    st.metric("Avg Confidence", f"{avg_conf:.1%}")

else:
    st.subheader("✍️ Single Review Analysis")
    review = st.text_area("Enter a review:", height=100)
    
    if st.button("Analyze Review"):
        if review:
            with st.spinner("Analyzing..."):
                response = requests.post(
                    'http://localhost:5000/analyze',
                    json={'text': review}
                )
                result = response.json()
                
                # Display result
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Review:** {result['text']}")
                
                with col2:
                    sentiment = result['sentiment']
                    confidence = result['confidence']
                    
                    if sentiment == 'POSITIVE':
                        st.success(f"**{sentiment}**\n{confidence:.1%}")
                    else:
                        st.error(f"**{sentiment}**\n{confidence:.1%}")

# Statistics tab
st.subheader("📊 Historical Statistics")
# Load from your database
stats_df = pd.DataFrame({
    'Date': pd.date_range('2024-01-01', periods=30),
    'Positive': [60 + i for i in range(30)],
    'Negative': [30 - i for i in range(30)],
})

st.line_chart(stats_df.set_index('Date'))
```

Run:
```bash
streamlit run dashboard.py
# Opens at http://localhost:8501
```

**Task 1:** Build dashboard. Test batch and single review modes.

**Deploy Streamlit (free):**
```bash
# Push to GitHub
git push

# Deploy via Streamlit Cloud: https://streamlit.io/cloud
# Connect your GitHub repo
# Done - live at: app.streamlit.app
```

**Resources:**
- Streamlit docs: https://docs.streamlit.io/

### **WEEK 8: Packaging for Sale**

#### 8.1 - Create SaaS Landing Page

Simple Streamlit page for customers:

```python
# landing_page.py
import streamlit as st

st.set_page_config(page_title="Sentiment Analysis for Czech Business")

st.title("Sentiment Analysis für Your Czech Business")

st.write("""
Analyze customer feedback, reviews, and feedback in Czech/Slovak.
Get instant insights into what your customers think.
""")

st.subheader("Features")
st.markdown("""
- **Real-time analysis** of Czech/Slovak text
- **REST API** for integration
- **Batch processing** for large datasets
- **Dashboard** for tracking trends
- **Accuracy >95%** on Czech reviews
""")

st.subheader("Pricing")
col1, col2, col3 = st.columns(3)

with col1:
    st.write("**Free**")
    st.write("- 100 analyses/month")
    st.write("- Dashboard view")
with col2:
    st.write("**Pro - €50/month**")
    st.write("- 10,000 analyses/month")
    st.write("- API access")
    st.write("- Email support")
with col3:
    st.write("**Enterprise**")
    st.write("- Unlimited")
    st.write("- Priority support")
    st.write("- Custom features")

if st.button("Get Started"):
    st.write("Coming soon! Contact: your_email@gmail.com")
```

#### 8.2 - Create README & Documentation

```markdown
# Czech Sentiment Analysis API

Production-grade sentiment analysis for Czech and Slovak text.

## Quick Start

```python
import requests

response = requests.post(
    'https://your-api.azurewebsites.net/analyze',
    json={'text': 'Výborný produkt!'}
)
print(response.json())
# {'sentiment': 'POSITIVE', 'confidence': 0.998}
```

## Features

- Real-time analysis
- >95% accuracy on Czech reviews
- Supports Czech and Slovak
- REST API
- Python SDK available

## Pricing

See [pricing page](https://your-website.com/pricing)
```

**Task 2:** Package product. Create landing page. Write documentation.

**Milestone:** Week 8 - Product ready to sell.

---

## **PHASE 5: WEEKS 9-10 - SELLING & OPTIMIZATION**

---

### **WEEK 9: Market Launch**

#### 9.1 - Find First Customers (3 days)

Target Czech/Slovak businesses that need sentiment analysis:

**E-commerce:**
- Shop owners on Heureka (1000+)
- Mall.cz sellers
- Alza.cz merchants

**Approach:**
```
Email template:

Subject: Automatic sentiment analysis for your reviews

Hi [Business Owner],

I built an AI system that analyzes Czech customer reviews in real-time.
It identifies what customers like/dislike about your products.

You can:
- See sentiment trends in your reviews
- Identify problems faster
- Respond to feedback with data

Try free demo: [Link to your Streamlit app]

Pricing: €50/month for full access

Interested? Reply to this email.

Best,
[Your name]
```

**Send to:** 50 shop owners. Expect 5-10% response rate (2-5 responses).

**Channels:**
- Cold email (most effective)
- LinkedIn outreach
- Czech startup communities
- Facebook business groups
- Slack communities

**Task 1:** Send 50 cold emails. Track responses. Follow up.

#### 9.2 - Optimize Based on Feedback (2 days)

Listen to customers:
- What features do they want?
- What's the pain point they're solving?
- What's hard about your product?

Iterate. Make it easier to use.

**Task 2:** Get 5+ customer conversations. Record feedback. Prioritize improvements.

### **WEEK 10: Scale & Optimize**

#### 10.1 - Performance Optimization

Model serving:
```python
# Use ONNX for faster inference
from optimum.onnxruntime import ORTModelForSequenceClassification

model = ORTModelForSequenceClassification.from_pretrained(
    "./czech_sentiment_model",
    from_transformers=True
)

# 2-3x faster than PyTorch
```

API improvements:
```python
# Add caching
from flask_caching import Cache

cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@app.route('/analyze', methods=['POST'])
@cache.cached(timeout=3600, query_string=True)
def analyze():
    # ... same code
```

#### 10.2 - Expand Features

- Multi-language support (add English, German)
- Aspect-based sentiment (what specifically is positive/negative?)
- Trend detection (is sentiment improving over time?)
- Integration: Zapier, Make.com, direct ecommerce platform APIs

#### 10.3 - Monetization

**Active customers:** 5-10
**Revenue:** €250-500/month
**Goal for next month:** 20+ customers → €1000+/month

**Milestone:** Week 10 - Product generating income, customers happy.

---

## **PROJECT STRUCTURE**

```
sentiment-analysis/
├── data/
│   ├── raw/  (scraped reviews)
│   ├── processed/  (cleaned, labeled)
│   └── test.csv  (validation data)
├── models/
│   ├── czech_sentiment_model/  (fine-tuned weights)
│   ├── embeddings/  (sentence-transformer weights)
│   └── faiss_index/  (FAISS index files)
├── src/
│   ├── scraper.py  (web scraping)
│   ├── sentiment_model.py  (HF classifier)
│   ├── rag.py  (RAG pipeline)
│   ├── api.py  (Flask API)
│   ├── pipeline.py  (ETL orchestration)
│   └── utils.py  (helpers)
├── app/
│   ├── dashboard.py  (Streamlit)
│   ├── landing_page.py  (sales page)
│   └── api.py  (production API)
├── notebooks/
│   ├── 01_explore_data.ipynb
│   ├── 02_fine_tuning.ipynb
│   ├── 03_rag_pipeline.ipynb
│   └── 04_evaluation.ipynb
├── tests/
│   ├── test_model.py
│   ├── test_api.py
│   └── test_pipeline.py
├── requirements.txt
├── README.md
├── Dockerfile
└── deploy/
    ├── streamlit_config.toml
    └── azure_deploy.yml
```

---

## **WEEKLY CHECKLIST**

### Week 1
- [ ] Environment setup complete
- [ ] HF course Chapters 1-3 read
- [ ] First English sentiment classifier working
- [ ] Understand different model trade-offs

### Week 2
- [ ] Fine-tuned Czech sentiment model created
- [ ] 300+ Czech training reviews collected
- [ ] Model tested on unseen Czech text
- [ ] Accuracy measured vs pre-trained model

### Week 3
- [ ] Embeddings concept understood
- [ ] FAISS index created with 100+ reviews
- [ ] Similarity search working
- [ ] Vector database persisted to disk

### Week 4
- [ ] LangChain retrieval setup complete
- [ ] RAG sentiment analyzer implemented
- [ ] LLM integration tested (optional)
- [ ] End-to-end pipeline working

### Week 5
- [ ] Data sources identified
- [ ] Scraper built and tested
- [ ] 500+ reviews scraped
- [ ] Raw data saved

### Week 6
- [ ] ETL pipeline built
- [ ] Data cleaning automated
- [ ] Cloud account created
- [ ] API deployed and tested publicly

### Week 7
- [ ] Streamlit dashboard created
- [ ] Single review analysis works
- [ ] Batch processing works
- [ ] Dashboard deployed

### Week 8
- [ ] Landing page created
- [ ] README documentation written
- [ ] Pricing tiers defined
- [ ] Product packaged

### Week 9
- [ ] 50+ cold emails sent
- [ ] 5+ customer conversations
- [ ] Feedback documented
- [ ] First customer acquired

### Week 10
- [ ] Performance optimized
- [ ] Features added based on feedback
- [ ] 10+ customers acquired
- [ ] €500+/month revenue

---

## **KEY RESOURCES**

| Topic | Link |
|-------|------|
| Hugging Face Course | https://huggingface.co/course |
| Transformers Docs | https://huggingface.co/docs/transformers |
| Sentence Transformers | https://www.sbert.net/ |
| FAISS | https://github.com/facebookresearch/faiss |
| LangChain | https://python.langchain.com/ |
| Streamlit | https://streamlit.io/ |
| Flask | https://flask.palletsprojects.com/ |
| Azure | https://azure.microsoft.com/en-us/free/ |

---

## **ESTIMATED TIMELINE**

| Phase | Weeks | Hours | Cumulative |
|-------|-------|-------|-----------|
| Foundations | 1-2 | 40 | 40 |
| RAG | 3-4 | 35 | 75 |
| Data + Pipelines | 5-6 | 35 | 110 |
| Product | 7-8 | 30 | 140 |
| Launch | 9-10 | 30 | 170 |

**Total: ~170 hours of focused work over 10 weeks**
**= 17 hours/week = 2-3 hours/day**

---

## **SUCCESS METRICS**

By end of Week 10, you should have:

✓ Production sentiment model (>95% accurate on Czech)
✓ Live API serving requests
✓ Streamlit dashboard deployed
✓ 10+ paying customers
✓ €500+/month recurring revenue
✓ Full understanding of: transformers, fine-tuning, RAG, deployment
✓ Impressive portfolio project for Czech tech jobs
✓ Ready for job interviews with real product to show

---

## **NEXT STEP**

**Today:** Start Week 1.1

Go to: https://huggingface.co/course/chapter1

Read Chapters 1-3 (2-3 hours). Come back when done.

Then we move to Week 1.2 and you build your first classifier.

**Ready?**
