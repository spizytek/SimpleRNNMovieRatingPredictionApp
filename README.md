# Simple RNN Sentiment Analysis on IMDB Dataset

This project demonstrates how to use a pre-trained Simple RNN model to perform sentiment analysis on movie reviews from the IMDB dataset. The model predicts whether a given review has a positive or negative sentiment.

## Project Overview

The project includes the following steps:
1. **Loading the IMDB Dataset**: The dataset is preprocessed, and word indices are mapped to their corresponding words.
2. **Loading the Pre-trained Model**: A pre-trained Simple RNN model is loaded to perform sentiment analysis.
3. **Text Preprocessing**: Helper functions are provided to preprocess the input text and decode reviews.
4. **Prediction**: The model predicts the sentiment of a given review and provides a confidence score.

## Files in the Project

- `simplernn_imdb_model.h5`: The pre-trained Simple RNN model file.
- Jupyter Notebook: Contains the code for loading the model, preprocessing text, and making predictions.

## Key Functions

- `decode_review(text)`: Decodes a sequence of integers back into a human-readable review.
- `preprocess_text(text)`: Preprocesses the input review text for the model.
- `predict_sentiment(review)`: Predicts the sentiment of a given review and returns the sentiment (Positive/Negative) along with the confidence score.

## Example Usage

1. Provide a movie review as input:
    ```python
    example_review = "The movie was fantastic! I really loved it and would watch it again."
    sentiment, confidence = predict_sentiment(example_review)
    print(f"Predicted Sentiment: {sentiment} (Confidence: {confidence:.4f})")
    ```

2. Output:
    ```
    Predicted Sentiment: Positive (Confidence: 0.7887)
    ```

## Requirements

- Python 3.x
- TensorFlow
- NumPy

## How to Run

1. Clone the repository and open the Jupyter Notebook.
2. Ensure the pre-trained model file `simplernn_imdb_model.h5` is in the same directory.
3. Run the cells in the notebook sequentially to load the model, preprocess the text, and make predictions.

## Acknowledgments

- IMDB Dataset: [IMDB Movie Reviews Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- TensorFlow/Keras: For building and training the RNN model.
