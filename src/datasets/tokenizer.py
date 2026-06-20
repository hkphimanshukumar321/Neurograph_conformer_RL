import json
from pathlib import Path

class Tokenizer:
    """A simple Character or Word-level Tokenizer for EEG-to-Text generation.
    
    Reserves tokens:
    0: <PAD>
    1: <SOS> (Start of Sequence)
    2: <EOS> (End of Sequence)
    3: <UNK> (Unknown)
    """
    
    PAD_ID = 0
    SOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3

    def __init__(self, mode="char"):
        self.mode = mode
        self.vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.inv_vocab = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        
    def fit(self, texts: list[str]):
        """Build vocabulary from a list of strings."""
        for text in texts:
            if not isinstance(text, str):
                continue
            tokens = list(text) if self.mode == "char" else text.split()
            for token in tokens:
                if token not in self.vocab:
                    idx = len(self.vocab)
                    self.vocab[token] = idx
                    self.inv_vocab[idx] = token
                    
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        if not isinstance(text, str):
            text = str(text)
        tokens = list(text) if self.mode == "char" else text.split()
        
        ids = []
        if add_special_tokens:
            ids.append(self.SOS_ID)
            
        for token in tokens:
            ids.append(self.vocab.get(token, self.UNK_ID))
            
        if add_special_tokens:
            ids.append(self.EOS_ID)
            
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        tokens = []
        for idx in ids:
            if skip_special_tokens and idx in (self.PAD_ID, self.SOS_ID, self.EOS_ID):
                continue
            tokens.append(self.inv_vocab.get(idx, "<UNK>"))
            
        return "".join(tokens) if self.mode == "char" else " ".join(tokens)
        
    @property
    def vocab_size(self):
        return len(self.vocab)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"mode": self.mode, "vocab": self.vocab}, f, ensure_ascii=False, indent=2)
            
    @classmethod
    def load(cls, path: str | Path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tokenizer = cls(mode=data["mode"])
        tokenizer.vocab = data["vocab"]
        tokenizer.inv_vocab = {int(v): k for k, v in data["vocab"].items()}
        return tokenizer
