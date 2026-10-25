# AI in Education: YouTube Comment Text Mining

An Indonesian-language text-mining project that explores public opinions about artificial intelligence (AI) in education using comments from selected YouTube videos. The workflow compares TF-IDF and Word2Vec representations, applies K-Means clustering, and uses Latent Dirichlet Allocation (LDA) to extract themes from the comments.

## Project Overview

AI is increasingly discussed as a tool for learning, teaching, and completing academic work. This project analyzes YouTube comments to explore how people discuss AI and its potential effects on education.

Rather than using predefined sentiment labels, the project applies **unsupervised learning and topic modeling** to discover recurring patterns and themes in the collected comments.

## Objectives

- Collect comments from selected YouTube videos discussing AI and education.
- Clean and preprocess Indonesian text.
- Compare TF-IDF and Word2Vec as text representations.
- Explore comment groupings using K-Means clustering.
- Extract and interpret recurring discussion themes using LDA.
- Compare clustering and topic-modeling results for qualitative insight.

## Data Collection

Comments are collected through the **YouTube Data API** using the Google API Python client.

- **Source:** Seven selected YouTube videos.
- **Collection limit:** Up to 500 top-level comments per video.
- **Collected dataset used in the analysis:** 1,169 comments.
- **Fields collected:** Video ID, author, comment text, like count, and publication timestamp.
- **Saved file:** `comments_yt.csv`.

The per-video collection limit helps reduce the risk of one video dominating the collected sample. The comments are still a sample from selected videos, not a representative survey of all public opinion.

The notebook uses a YouTube API key stored in Google Colab Secrets under `YT_API_KEY`. 

## Workflow

### 1. Text preprocessing

The notebook performs the following steps:

1. Convert text to lowercase.
2. Remove URLs, numbers, punctuation, and excess whitespace.
3. Tokenize the comments.
4. Remove Indonesian stopwords and tokens shorter than three characters.
5. Apply Indonesian stemming using Sastrawi.
6. Remove additional noise words.
7. Join the remaining tokens into a final text representation.

### 2. Text representation

**TF-IDF**
- Uses `TfidfVectorizer` with a maximum of 5,000 features.
- Produces a sparse representation based on term importance across comments.
- The notebook reports a matrix of 1,169 comments and 4,415 features after preprocessing.

**Word2Vec**
- Trains 100-dimensional word embeddings.
- Uses a context window of 5 and a minimum word count of 2.
- Represents each comment by averaging the embeddings of words found in the trained vocabulary.

### 3. K-Means clustering

K-Means is evaluated for cluster counts from 2 through 10 using silhouette scores.

- TF-IDF clustering produces low silhouette scores, suggesting that the comments are not clearly separated using this representation.
- Word2Vec produces comparatively higher silhouette scores in the notebook.
- The analysis examines both two-cluster and three-cluster Word2Vec solutions.
- t-SNE is used to visualize high-dimensional comment representations in two dimensions.

### 4. Topic modeling with LDA

Latent Dirichlet Allocation is applied to a count-vector representation of the processed comments.

- Uses a maximum vocabulary size of 1,000 features.
- Compares two-topic and three-topic solutions.
- Interprets the top words in each topic to understand recurring discussions about AI and education.

## Key Findings

### Word2Vec clustering

The notebook interprets the two-cluster solution as separating:
- General reactions to and discussion about the videos.
- Discussions focused on AI, technology, and its possible effects.

The three-cluster solution provides more detail by distinguishing:
- Reactions and interactions related to the videos.
- AI technology and its real-world development and applications.
- AI's benefits and drawbacks in education, including teaching and student work.

Although Word2Vec appears more useful than TF-IDF for this clustering task, some overlap remains between clusters.

### LDA topic modeling

The notebook finds that the **three-topic LDA solution provides more useful thematic separation** than the two-topic solution. The interpreted themes include:

1. **Concerns and risks:** possible harms or misuse of AI in education.
2. **Benefits of AI:** making tasks easier, supporting teaching, and helping people complete work.
3. **Human cognition and AI development:** discussion about the human brain, limitations, and future AI development in Indonesia.

The two-topic solution more broadly separates general discussion of AI technology from discussion of its benefits in education.

### Overall conclusion

For the purpose of interpreting opinions about AI in education, the notebook concludes that **LDA provides more useful themes than K-Means clustering**. K-Means groups comments by representation similarity, but some clusters contain reactions to the videos rather than substantive opinions about AI. LDA produces themes that align more directly with the project's research focus.

These findings are qualitative interpretations of unsupervised-model outputs, not verified labels or causal conclusions.

## Tools and Libraries

- Python
- pandas
- Google API Python Client / YouTube Data API
- NLTK
- Sastrawi
- scikit-learn
- Gensim (Word2Vec)
- Matplotlib

## How to Run

1. Open `ai-in-education-text-mining.ipynb` in Google Colab or Jupyter Notebook.
2. Enable the YouTube Data API for a Google Cloud project and obtain an API key if you want to rerun the scraping step.
3. In Google Colab, add the key to **Secrets** with the name `YT_API_KEY` and allow notebook access. Do not hard-code or commit the key.
4. Run the cells in order to collect comments, preprocess text, create text representations, run clustering, and fit LDA.
5. If using the saved `comments_yt.csv`, load that file instead of scraping comments again.

Install the main packages if needed:

```bash
pip install pandas google-api-python-client nltk Sastrawi scikit-learn gensim matplotlib
```

The notebook downloads NLTK tokenization and stopword resources during execution. The scraping step requires a valid API key and may be subject to YouTube Data API quota limits.

## Limitations

- The data comes from seven selected videos and may not reflect the views of all Indonesian students, educators, or the general public.
- YouTube comments can include slang, spelling variations, sarcasm, and mixed-language text that automated preprocessing may not fully capture.
- Removing frequent words and applying stemming can also remove context or nuance.
- K-Means and LDA are unsupervised methods; the clusters and topics require human interpretation.
- t-SNE is a visualization technique and should not be treated as proof that clusters are objectively distinct.
- The project explores themes in comments; it does not establish whether AI improves or harms educational outcomes.

## Project Type

Academic project focused on Indonesian NLP, text mining, unsupervised clustering, topic modeling, and qualitative interpretation of online discussions.
