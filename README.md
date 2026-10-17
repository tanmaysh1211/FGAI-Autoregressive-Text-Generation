# Autoregressive Text Generation Using RNN, BiRNN, LSTM, and GRU

## 1. Project Overview

This project implements and compares four recurrent neural network architectures for autoregressive text generation using PyTorch:

- Simple Recurrent Neural Network (RNN)
- Bidirectional Recurrent Neural Network (BiRNN)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)

The models are trained on *Frankenstein; or, The Modern Prometheus* by Mary Wollstonecraft Shelley. Each model learns to predict the next word from a sequence of preceding words and generates text sequentially from a given seed sequence.

The project evaluates predictive performance, training time, text diversity, repetition, and the influence of sampling temperature.

## 2. Objectives

- Select and preprocess a textual dataset suitable for language modelling.
- Construct input–target sequences for next-word prediction.
- Implement and train four recurrent neural network architectures under comparable experimental conditions.
- Evaluate the models using test loss, perplexity, prediction accuracy, and training time.
- Generate text using multiple seed sequences and sampling temperatures.
- Compare generated text for coherence, contextual consistency, diversity, and repetition.
- Identify the most suitable model based on quantitative and qualitative results.

## 3. Dataset Description

**Dataset:** Frankenstein; or, The Modern Prometheus  
**Author:** Mary Wollstonecraft Shelley  
**Source:** Project Gutenberg  
**Dataset link:** https://www.gutenberg.org/ebooks/84

The dataset is an English-language literary novel containing approximately 438,841 characters in the downloaded plain-text file. The original file contains Gutenberg introductory and licensing material in addition to the novel.

After preprocessing, the dataset contains approximately 86,013 tokens and 7,021 unique tokens.

### Preprocessing

The preprocessing pipeline includes:

1. Loading the downloaded text file using UTF-8 encoding.
2. Converting text to lowercase.
3. Normalizing whitespace.
4. Tokenizing words and punctuation.
5. Constructing a vocabulary with word-to-index and index-to-word mappings.
6. Splitting the token sequence into training, validation, and test sets.
7. Converting tokens into numerical IDs.
8. Creating fixed-length input sequences and their corresponding next-word targets.

The displayed split contains 68,810 training tokens, 8,601 validation tokens, and 8,602 test tokens. The vocabulary contains 6,414 entries after the vocabulary is constructed for modelling.

Each input sequence contains 12 tokens, and its target is the next token.

## 4. Models Implemented

### Simple RNN

The Simple RNN processes sequential inputs using recurrent hidden states. It provides a baseline for next-word prediction but can struggle to retain long-range information.

### Bidirectional RNN (BiRNN)

The BiRNN uses forward and backward recurrent processing to learn representations from both directions within the input window. It is included for architectural comparison. During generation, predictions must depend only on the supplied seed and previously generated tokens, not on future ground-truth text.

### Long Short-Term Memory (LSTM)

The LSTM uses memory cells and gates to regulate information flow, helping it learn longer dependencies in sequential text.

### Gated Recurrent Unit (GRU)

The GRU uses update and reset gates to control information retention and replacement. It offers an alternative recurrent architecture with fewer gates than an LSTM.

## 5. Technology Stack

- Python 3.12
- PyTorch
- NumPy
- pandas
- Matplotlib
- Jupyter Notebook


The experiments were run on a CPU using a PyTorch CPU build. CUDA acceleration was unavailable.

## 6. Experimental Setup

All four architectures were trained on the same preprocessed dataset using the same next-word prediction objective.

The notebook uses five training epochs and compares validation performance during training. The evaluation includes:

- Test loss: cross-entropy loss on the test set.
- Perplexity: exponential of the average test cross-entropy loss.
- Test accuracy: percentage of next-word predictions that match the target.
- Training time: elapsed training time for each model.
- Generation metrics: total generated words, distinct words, unique-word ratio, and repeated-word rate.

Generated text was evaluated using two seed sequences:

- `it was on a dreary night of november`
- `i had worked hard for nearly two years`

The sampling temperatures were 0.5, 1.0, and 1.5.

## 7. Quantitative Results

| Model | Test Loss | Test Perplexity | Test Accuracy (%) | Training Time (minutes) |
|---|---:|---:|---:|---:|
| LSTM | 5.6176 | 275.2373 | 14.7846 | 5.4703 |
| GRU | 5.6298 | 278.6080 | 14.9709 | 7.8190 |
| RNN | 5.6866 | 294.8746 | 13.3760 | 5.8039 |
| BiRNN | 5.7289 | 307.6274 | 13.7951 | 9.7087 |

### Results Interpretation

- **LSTM:** Achieved the lowest test loss and perplexity, indicating the strongest predictive performance according to these metrics.
- **GRU:** Achieved the highest test accuracy, although its test loss and perplexity were slightly higher than LSTM's.
- **Simple RNN:** Provided a baseline with lower predictive performance than LSTM and GRU.
- **BiRNN:** Had the highest test loss and perplexity and the longest measured training time in this experiment.

The GRU and LSTM results are close, so the preferred model depends on whether predictive loss, accuracy, speed, or generated-text quality is prioritized.

## 8. Sampling Temperature Analysis

Temperature controls how concentrated or distributed the next-word sampling probabilities are.

- **Temperature 0.5:** Produces more conservative samples by favouring high-probability words. This can improve local predictability but may increase repetitive phrasing.
- **Temperature 1.0:** Uses the model's unscaled probability distribution and provides a useful baseline for comparison.
- **Temperature 1.5:** Makes the distribution flatter, allowing less probable words to be selected. This generally increases lexical diversity but can reduce grammatical and contextual consistency.

The experiment measures diversity using the unique-word ratio and repetition using the repeated-word rate. These metrics do not directly measure grammatical correctness or semantic coherence, so generated passages must also be inspected manually.

In the observed samples, increasing temperature generally increased lexical diversity. However, the higher-temperature outputs frequently contained unrelated words and incoherent phrases. The best temperature is therefore a trade-off between diversity and readability rather than simply the temperature with the highest unique-word ratio.

## 9. Generated Text and Qualitative Evaluation

The notebook generates samples from the same seed sequences for all four models at each temperature.

The samples show that the models can continue a literary-style prompt and sometimes produce locally plausible phrases. However, they also exhibit grammatical errors, inconsistent context, and occasionally repeated expressions.

The GRU shows repetition in some low-temperature samples, while higher-temperature outputs across the models often contain unusual word combinations. These observations indicate that next-word prediction alone does not guarantee coherent long-form text.

The generated passages are saved in `generated_text/temperature_samples.json`, and quantitative generation metrics are saved in `results/generation_metrics.csv`.

## 10. Results and Supporting Files

- `assignment2.ipynb` — complete implementation, training, evaluation, visualizations, and generation.
- `data/frankenstein.txt` — downloaded original dataset.
- `data/frankenstein_cleaned.txt` — preprocessed dataset.
- `generated_text/temperature_samples.json` — generated samples for the four models, two seeds, and three temperatures.
- `results/model_comparison.csv` — test metrics and training times.
- `results/generation_metrics.csv` — diversity and repetition metrics.
- `results/training_history.json` — training and validation history.
- `plots/validation_loss.png` — validation loss comparison.
- `models/` — saved model checkpoints, if included in the repository.

## 11. How to Run the Project

### Prerequisites

Install Python 3.12 and Git. A CPU is sufficient, although training may take several minutes.

### Setup

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

Create and activate a virtual environment on Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open `assignment2.ipynb` and execute the notebook cells in order. Ensure that the dataset is located at the path expected by the notebook.

Training the four models can take several minutes on a CPU. Re-running the complete notebook will retrain the models.

## 12. Limitations

- The dataset is a single literary work, limiting the range of language and writing styles learned.
- The vocabulary-based approach may generate awkward word combinations and does not guarantee grammatical correctness.
- The experiments use a limited number of epochs and a relatively small model-training budget.
- The bidirectional architecture requires careful handling during autoregressive generation to avoid information leakage.
- Diversity and repetition metrics provide only partial evidence of text quality.
- Results can vary with random seeds, hardware, and software versions.

## 13. Conclusion

This project compares Simple RNN, BiRNN, LSTM, and GRU models for autoregressive next-word prediction and text generation using Frankenstein.

LSTM achieved the lowest test loss (5.6176) and perplexity (275.2373), while GRU achieved the highest test accuracy (14.9709%). LSTM also had the shortest measured training time in the reported experiment.

Based on the quantitative predictive metrics, LSTM is the leading candidate. Nevertheless, the generated samples show that all four models have limitations in coherence and contextual consistency. The final model choice should therefore consider generated-text quality alongside predictive metrics and computational cost.

## 14. References

1. Shelley, M. W. (1818). *Frankenstein; or, The Modern Prometheus*. Project Gutenberg. https://www.gutenberg.org/ebooks/84
2. PyTorch Documentation. https://pytorch.org/docs/stable/index.html
