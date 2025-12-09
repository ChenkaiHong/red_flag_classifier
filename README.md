# Movie Review Sentiment Classifier

A Streamlit web application for sentiment classification of movie reviews as "Positive" or "Negative" using Large Language Models (LLMs) with few-shot prompting.

## Features

- **LLM-Powered Classification**: Uses OpenAI GPT models with few-shot prompting
- **Few-Shot Learning**: Automatically creates balanced few-shot examples from training data
- **Configurable Prompting**: Adjustable number of examples per class in the prompt
- **Multiple Model Support**: Choose between different OpenAI models
- **Interactive Web Interface**: User-friendly Streamlit app for real-time classification
- **Batch Processing**: Support for multiple upload methods with cost estimation:
  - Text files (one review per line)
  - CSV files (with 'sentence', 'review', 'text', or 'content' column) 
  - Manual text input
- **LLM Evaluation**: Performance testing on sample test set with API cost tracking
- **Downloadable Results**: Export batch classification results as CSV

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Get OpenAI API Key:**
   - Sign up at https://platform.openai.com
   - Create an API key in your dashboard
   - Add credits to your account

3. **Run the app:**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** to `http://localhost:8501`

## Data

The app uses a trimmed version of the IMDB Movie Review Dataset:
- `data/imdb_reviews_sentiment.json` - 200 reviews (100 positive, 100 negative)
- Original dataset: `data/IMDB Dataset.csv` (50,000 reviews from Kaggle)

The JSON file contains an array with objects having:
- `item_id`: Unique identifier
- `sentence`: Review text content (truncated to 500 characters)
- `gold_label`: Either "Positive" or "Negative"

## Usage

1. **API Setup**: Enter your OpenAI API key in the sidebar and select a model
2. **Prepare Few-Shot Examples**: Click "Prepare Few-Shot Examples" to create balanced training examples
3. **Single Classification**: Enter a movie review and click "Classify Review" for individual predictions
4. **Batch Classification**: Upload files or enter multiple reviews (with cost estimates)
5. **LLM Evaluation**: Test performance on a sample of the test set

## Model Details

- **Algorithm**: Few-shot prompting with OpenAI GPT models
- **Prompt Engineering**: Structured prompts with clear sentiment definitions and balanced examples
- **Few-Shot Examples**: Configurable number of examples per class (0-50)
- **Temperature**: Low temperature (0.1) for consistent predictions
- **Evaluation**: Accuracy, precision, recall, F1-score, and confusion matrix on test samples

## Cost Considerations

- **Single Classification**: ~$0.001 per review
- **Batch Processing**: Cost scales linearly with number of reviews
- **Model Evaluation**: ~$0.05-0.20 for 50-item test sample

## Example Usage

### Single Review Classification
```
Input: "This movie was absolutely fantastic! The acting was superb."
Output: 👍 Positive - This review expresses a favorable opinion!
```

### Few-Shot Prompt Structure
```
You are a movie review sentiment classifier that categorizes reviews as either "Positive" or "Negative".

Your task is to analyze the sentiment expressed in movie reviews:
- Positive: Reviews that express enjoyment, praise, satisfaction, or recommendation.
- Negative: Reviews that express disappointment, criticism, or dissatisfaction.

Examples:
Review: "A masterpiece of modern cinema. The director's vision shines through every frame."
Sentiment: Positive

Review: "What a waste of time. The story made no sense and the ending was unsatisfying."
Sentiment: Negative

Review: "[Your input review here]"
Sentiment:
```

### Batch Classification
Upload files with cost estimates shown before processing. Results are downloadable as CSV.

## Acknowledgments

Documentation for this project was generated with Claude Code.