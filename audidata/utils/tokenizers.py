import bisect
import itertools
import re
from typing import List

import numpy as np


class BaseTokenizer:
    r"""Base class for all tokenizers.
    """

    def __init__(self, words: str):
        self.words = words
        self.vocab_size = len(self.words)

        self.token_to_word = {token: word for token, word in enumerate(self.words)}
        self.word_to_token = {word: token for token, word in enumerate(self.words)}

    def stoi(self, word: str) -> int:
        r"""String (word) to index.
        """
        if word in self.words:
            return self.word_to_token[word]
        
    def itos(self, token: int) -> str:
        r"""Index to string (word).
        """
        assert 0 <= token < self.vocab_size
        return self.token_to_word[token]


class SpecialTokenizer(BaseTokenizer):
    def __init__(self):
        words = ["<pad>", "<sos>", "<eos>", "<unk>"]
        super().__init__(words=words)
        

class NameTokenizer(BaseTokenizer):
    def __init__(self):
        words = [
            "note_on", "note_off", "note_sustain",
            "pedal_on", "pedal_off", "pedal_sustain",
            "beat", "downbeat",
        ]
        words = ["name={}".format(w) for w in words]

        vocab_size = 16  # Reserve for future
        words = pad_list(words, vocab_size)

        super().__init__(words=words)


def pad_list(x: List, max_len: int) -> List:
    assert len(x) <= max_len
    while len(x) < max_len:
        x.append("blank_{}".format(len(x)))
    return x


class TimeTokenizer(BaseTokenizer):
    def __init__(self, max_duration=60., fps=100):

        self.fps = fps
        self.delta_t = 1. / fps

        words = ["time={:.4f}".format(time) for time in \
            np.arange(0, max_duration + self.delta_t, self.delta_t)]

        super().__init__(words=words) 


    def stoi(self, word: str) -> int:

        if "time=" in word:
            time = float(re.search('time=(.*)', word).group(1))
            time = round(time * self.fps) / self.fps
            word = "time={:.4f}".format(time)
            token = super().stoi(word)
            return token
    

class PitchTokenizer(BaseTokenizer):
    def __init__(self, classes_num=128):

        words = ["pitch={}".format(pitch) for pitch in range(classes_num)]

        super().__init__(words=words)

    
class VelocityTokenizer(BaseTokenizer):
    def __init__(self, classes_num=128):
        
        words = ["velocity={}".format(pitch) for pitch in range(classes_num)]

        super().__init__(words=words)


class ConcatTokenizer:
    def __init__(self, tokenizers, verbose=False):

        self.tokenizers = tokenizers

        self.words = list(itertools.chain(*[tk.words for tk in tokenizers]))
        self.vocab_sizes = [tk.vocab_size for tk in self.tokenizers]
        self.vocab_size = np.sum(self.vocab_sizes)

        self.cumulative_sizes = np.cumsum(self.vocab_sizes)

        if verbose:
            print("Vocab size: {}".format(self.vocab_size))
            for tk in self.tokenizers:
                print(tk.vocab_size)

    def stoi(self, word: str) -> int:
        
        start_token = 0

        for tk in self.tokenizers:
            
            token = tk.stoi(word)
            
            if token is not None:
                return start_token + token
            else:
                start_token += tk.vocab_size

        raise NotImplementedError("{} is not in the vocabulary!".format(word))

    def itos(self, token: int) -> str:
        
        assert 0 <= token < self.vocab_size

        tokenizer_idx = bisect.bisect_right(self.cumulative_sizes, token)
        tokenizer = self.tokenizers[tokenizer_idx]
        rel_token = token - self.cumulative_sizes[tokenizer_idx - 1]
        word = tokenizer.itos(rel_token)

        return word


if __name__ == '__main__':
    r"""Example.
    """

    tokenizer = ConcatTokenizer([
        SpecialTokenizer(),
        NameTokenizer(),
        TimeTokenizer(),
        PitchTokenizer(),
        VelocityTokenizer()
    ])

    token = tokenizer.stoi("name=note_on")
    word = tokenizer.itos(token)
    print(token, word)

    token = tokenizer.stoi("time=26.789")
    word = tokenizer.itos(token)
    print(token, word)

    token = tokenizer.stoi("pitch=34")
    word = tokenizer.itos(token)
    print(token, word)

    token = tokenizer.stoi("velocity=34")
    word = tokenizer.itos(token)
    print(token, word)
