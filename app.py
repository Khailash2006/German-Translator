"""
FastAPI deployment for the English -> German Transformer translator.

Run with:
    uvicorn app:app --host 0.0.0.0 --port 8000

Then call:
    POST http://localhost:8000/translate
    body: {"text": "how are you"}
"""

import math
import pickle

import spacy
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

PAD, UNK, BOS, EOS = 0, 1, 2, 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TranslatorModel(nn.Module):
    def __init__(
        self,
        src_vocab,
        tgt_vocab,
        d_model=256,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        dropout=0.1,
    ):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab, d_model, padding_idx=0)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model,
            nhead,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(d_model, tgt_vocab)
        self.d_model = d_model

    def forward(self, src, tgt, src_key_padding_mask, tgt_key_padding_mask, tgt_mask):
        src = self.src_emb(src) * math.sqrt(self.d_model)
        tgt = self.tgt_emb(tgt) * math.sqrt(self.d_model)
        out = self.transformer(
            src,
            tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return self.fc(out)

def encode(sentence, vocab, tokenizer):
    tokens = [BOS] + [vocab.get(tok, UNK) for tok in tokenizer(sentence)] + [EOS]
    return torch.tensor(tokens, dtype=torch.long)


def decode(tensor, inv_vocab):
    return " ".join(
        inv_vocab.get(x, "UNK") for x in tensor.tolist() if x not in (PAD, BOS, EOS)
    )


def make_masks(src, tgt):
    src_pad_mask = (src == PAD).to(torch.bool)
    tgt_pad_mask = (tgt == PAD).to(torch.bool)

    tgt_len = tgt.size(1)
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=tgt.device)

    return src_pad_mask, tgt_pad_mask, tgt_mask

nlp_en = None
model = None
vocab_en = None
vocab_de = None
inv_vocab_de = None


def tokenize_en(text):
    return [t.text.lower() for t in nlp_en.tokenizer(text)]


def translate(sentence: str, max_len: int = 50, beam_size: int = 5, repetition_penalty: float = 1.3) -> str:
    src = encode(sentence, vocab_en, tokenize_en).unsqueeze(0).to(DEVICE)

    beams = [([BOS], 0.0)]

    with torch.no_grad():
        for _ in range(max_len):
            new_beams = []
            for tokens, score in beams:
                if tokens[-1] == EOS:
                    new_beams.append((tokens, score))
                    continue

                tgt = torch.tensor(tokens, dtype=torch.long, device=DEVICE).unsqueeze(0)
                src_pad_mask, tgt_pad_mask, tgt_mask = make_masks(src, tgt)
                output = model(src, tgt, src_pad_mask, tgt_pad_mask, tgt_mask)
                logits = output[:, -1, :].squeeze(0)

                for t in set(tokens):
                    logits[t] /= repetition_penalty

                log_probs = torch.log_softmax(logits, dim=-1)
                topk_log_probs, topk_idx = log_probs.topk(beam_size)

                for lp, idx in zip(topk_log_probs.tolist(), topk_idx.tolist()):
                    new_beams.append((tokens + [idx], score + lp))

            new_beams.sort(key=lambda x: x[1] / len(x[0]), reverse=True)
            beams = new_beams[:beam_size]

            if all(tokens[-1] == EOS for tokens, _ in beams):
                break

    best_tokens = beams[0][0]
    return decode(torch.tensor(best_tokens), inv_vocab_de)


app = FastAPI(title="EN-DE Translator", version="1.0")

app = FastAPI(title="EN-DE Translator", version="1.0")


class TranslateRequest(BaseModel):
    text: str
    beam_size: int = 5
    max_len: int = 50


class TranslateResponse(BaseModel):
    source: str
    translation: str


@app.on_event("startup")
def load_artifacts():
    global nlp_en, model, vocab_en, vocab_de, inv_vocab_de

    nlp_en = spacy.load("en_core_web_sm")

    with open("vocab_en.pkl", "rb") as f:
        vocab_en = pickle.load(f)
    with open("vocab_de.pkl", "rb") as f:
        vocab_de = pickle.load(f)
    inv_vocab_de = {v: k for k, v in vocab_de.items()}

    model_ = TranslatorModel(len(vocab_en), len(vocab_de)).to(DEVICE)
    state_dict = torch.load("best_model.pt", map_location=DEVICE)
    model_.load_state_dict(state_dict)
    model_.eval()

    globals()["model"] = model_
    globals()["inv_vocab_de"] = inv_vocab_de

    print(f"Model loaded on {DEVICE}. EN vocab: {len(vocab_en)}, DE vocab: {len(vocab_de)}")


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@app.post("/translate", response_model=TranslateResponse)
def translate_endpoint(req: TranslateRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    result = translate(req.text, max_len=req.max_len, beam_size=req.beam_size)
    return TranslateResponse(source=req.text, translation=result)
