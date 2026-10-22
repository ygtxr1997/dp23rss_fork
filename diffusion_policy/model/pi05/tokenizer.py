# Adapted from OpenPI (rlinf-openpi 0.1.1); see licenses/OpenPI-LICENSE.
"""Local SentencePiece tokenizer for the fixed non-discrete-state LIBERO recipe."""
import logging

import numpy as np
import sentencepiece


class PaligemmaTokenizer:
    """Tokenize language using embedded model bytes, without downloads or JAX."""

    def __init__(self, model_proto: bytes, max_len: int = 200):
        self.model_proto = model_proto
        self.max_len = max_len
        self.processor = sentencepiece.SentencePieceProcessor(model_proto=model_proto)

    def tokenize(self, prompt: str):
        """Return padded int64 token IDs and a boolean validity mask."""
        cleaned = prompt.strip().replace('_', ' ').replace('\n', ' ')
        tokens = self.processor.encode(cleaned, add_bos=True) + self.processor.encode('\n')
        if len(tokens) > self.max_len:
            logging.warning('Prompt has %d tokens; truncating to %d', len(tokens), self.max_len)
        length = min(len(tokens), self.max_len)
        tokens_L = np.zeros(self.max_len, dtype=np.int64)
        mask_L = np.arange(self.max_len) < length
        tokens_L[:length] = tokens[:length]
        return tokens_L, mask_L
