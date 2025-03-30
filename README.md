# IMDB Sentiment Analysis with LSTM

## Project Overview
This project implements a sentiment analysis model using deep learning to classify IMDB movie reviews as positive or negative. The model uses a bidirectional LSTM architecture with dropout layers to prevent overfitting.

## Dataset
The project uses the IMDB dataset, which contains 50,000 movie reviews labeled as positive or negative sentiment. Each review is preprocessed to improve model performance.

## Preprocessing Pipeline
1. **Text Cleaning**:
   - Converting text to lowercase
   - Removing HTML tags
   - Removing special characters
   - Normalizing whitespace

2. **Tokenization**:
   - Breaking text into individual words using NLTK's word_tokenize

3. **Stopword Removal**:
   - Filtering out common English stopwords that don't contribute to sentiment

4. **Lemmatization**:
   - Reducing words to their base forms using WordNet lemmatizer
   - Example: "running" → "run", "better" → "good"

## Model Architecture
- **Embedding Layer**: Converts words to dense vectors of fixed size (128 dimensions)
- **LSTM Layers**: Two LSTM layers (64 and 32 units) for sequence processing
- **Dropout Layers**: Multiple dropout layers (0.5, 0.5, 0.3) to prevent overfitting
- **Dense Layers**: Fully connected layers with ReLU and sigmoid activations

## Training Details
- **Training Strategy**: Early stopping with patience of 3 epochs
- **Batch Size**: 64
- **Optimizer**: Adam
- **Loss Function**: Binary Cross-Entropy

## Results
- **Test Accuracy**: 86.95%
- **Training Time**: ~100 seconds per epoch

### Confusion Matrix Analysis
- **True Negatives**: 4271
- **False Positives**: 688
- **False Negatives**: 702
- **True Positives**: 4337

This shows the model performs equally well on both positive and negative reviews, with a balanced error rate.

## Requirements
- Python 3.6+
- TensorFlow 2.x
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Matplotlib
- Seaborn

## Running the Code
1. Install the required packages:
   ```
   pip install tensorflow pandas numpy nltk scikit-learn matplotlib seaborn
   ```
2. Download the IMDB dataset and place it in the project directory
3. Run the script:
   ```
   python sentiment_analysis.py
   ```

## Future Improvements
- Experiment with different preprocessing techniques
- Try bidirectional LSTM or transformer-based models
- Implement cross-validation
- Add attention mechanisms
- Fine-tune hyperparameters
