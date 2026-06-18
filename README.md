# Neural Machine Translation using PyTorch Transformer

This project implements an English-to-German Neural Machine Translation (NMT) system using PyTorch's built-in `nn.Transformer` module. The trained model is deployed as a REST API using FastAPI for real-time translation.

## Features

* PyTorch `nn.Transformer`
* English → German translation
* Tokenization and vocabulary preprocessing
* Training and validation pipeline
* Greedy decoding for inference
* FastAPI deployment
* Interactive API documentation via Swagger UI

## Tech Stack

* Python
* PyTorch
* FastAPI
* Uvicorn
* Scacy
* dataset

## Model

The model uses PyTorch's Transformer architecture consisting of:

* Token Embeddings
* Positional Encoding
* Transformer Encoder-Decoder
* Linear Output Layer

## Training

Training was performed using:

* CrossEntropyLoss
* Adam Optimizer
* Padding Masks
* Causal Masks

The best model is saved as:

```text
best_model.pt
```

## Running the API

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

to access the Swagger UI.

## API Example

Request:

```json
{
  "text": "How are you?"
}
```

Response:

```json
{
  "translation": "Wie geht es dir?"
}
```

## Future Improvements

* BLEU Score Evaluation
* Larger Training Dataset
* Better Tokenization
* Docker Deployment

## Author

Khailash S
