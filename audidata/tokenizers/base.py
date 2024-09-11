import bisect
import itertools
import re
from typing import Any, Union

import numpy as np


class BaseTokenizer:
    r"""Base class for all tokenizers.
    """

    def __init__(self, words: list[Any]):
        self.words = words
        self.vocab_size = len(self.words)

        self.token_to_word = {token: word for token, word in enumerate(self.words)}
        self.word_to_token = {word: token for token, word in enumerate(self.words)}

    def stoi(self, word: Any) -> int:
        r"""String (word) to index.
        """
        return self.word_to_token.get(word)
        
    def itos(self, token: int) -> Any:
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


def pad_list(x: list, max_len: int) -> list:
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
        
class DrumTokenizer(BaseTokenizer):
    def __init__(self, classes_num=128):
        
        words = ["drum_pitch={}".format(pitch) for pitch in range(classes_num)]

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
        
        if tokenizer_idx == 0:
            rel_token = token
        else:
            rel_token = token - self.cumulative_sizes[tokenizer_idx - 1]
        
        word = tokenizer.itos(rel_token)

        return word
    
class ProgramTokenizer(BaseTokenizer):
    def __init__(self):
        words = [f"program={i}" for i in range(128)]
        super().__init__(words=words)

    def stoi(self, word: Union[str, int]) -> int:
        if isinstance(word, int):
            if 0 <= word < 128:
                return word
        elif isinstance(word, str) and word.startswith("program="):
            try:
                program = int(word.split("=")[1])
                if 0 <= program < 128:
                    return program
            except ValueError:
                pass
        return self.word_to_token.get("<unk>", 0)  # Default to 0 (Piano). TODO: discuss

    def itos(self, token: int) -> str:
        if 0 <= token < 128:
            return f"program={token}"
        return "program=0"
    
class DictTokenizer(BaseTokenizer):
    def __init__(self, key_tokenizer_pairs):
        self.key_tokenizer_pairs = key_tokenizer_pairs
        self.keys = list(key_tokenizer_pairs.keys())
        self.tokenizers = list(key_tokenizer_pairs.values())
        
        words = ["<bot>", "<eot>"]  # begin of token, end of token
        for key in self.keys:
            words.append(f"<{key}>")  # Add a token for each key
        for tokenizer in self.tokenizers:
            for word in tokenizer.words:
                words.append(f"<{word}>")
        
        super().__init__(words=words)
        
        self.bot_token = self.word_to_token["<bot>"]
        self.eot_token = self.word_to_token["<eot>"]
        self.key_tokens = {key: self.word_to_token[f"<{key}>"] for key in self.keys}
        self.vocab_size = len(self.words)
    
    def tokenize(self, sequence):
        tokens = []
        current_dict = None
        for item in sequence:
            if item == "<bot>":
                if current_dict is not None:
                    tokens.append(current_dict)
                current_dict = {key: self.key_tokens[key] for key in self.keys}  
                # Initialize with key tokens, therefore if a key has not been used, it will be initialized with the key token.
            elif item == "<eot>":
                if current_dict is not None:
                    tokens.append(current_dict)
                    current_dict = None
            else:
                key, value = item[1:-1].split(":", 1)
                tokenizer = self.key_tokenizer_pairs[key]
                token = tokenizer.stoi(value)
                if token is not None and current_dict is not None:
                    current_dict[key] = token
        
        if current_dict is not None:
            tokens.append(current_dict)
        
        return tokens
    
    def detokenize(self, tokens):
        sequence = []
        for token_dict in tokens:
            sequence.append("<bot>")
            for key, value in token_dict.items():
                if value == self.key_tokens[key]:
                    continue  # Skip if it's just the key token
                tokenizer = self.key_tokenizer_pairs[key]
                word = tokenizer.itos(value)
                sequence.append(f"<{key}:{word}>")
            sequence.append("<eot>")
        
        return sequence
    
    def get_vocab_sizes(self):
        vocab_dict = {}
        for key, tokenizer in self.key_tokenizer_pairs.items():
            vocab_dict[key] = tokenizer.vocab_size + 1 # Add 1 for the key token
        return vocab_dict


if __name__ == '__main__':
    r"""Example.
    """
    
    print("Testing ConcatTokenizer")

    tokenizer = ConcatTokenizer([
        SpecialTokenizer(),
        NameTokenizer(),
        TimeTokenizer(),
        PitchTokenizer(),
        DrumTokenizer(),
        VelocityTokenizer(),
        ProgramTokenizer(),
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

    token = tokenizer.stoi("program=34")
    word = tokenizer.itos(token)
    print(token, word)
    
    token = tokenizer.stoi("drum_pitch=34")
    word = tokenizer.itos(token)
    print(token, word)
    
    print("Testing DictTokenizer")
    
    tokenizer = DictTokenizer({
        "special": SpecialTokenizer(),
        "onset": ConcatTokenizer([
            TimeTokenizer(),
            NameTokenizer(),
        ]),
        "pitch": PitchTokenizer(),
        "velocity": VelocityTokenizer(),
        "offset": ConcatTokenizer([
            TimeTokenizer(),
            NameTokenizer(),
        ]),
    })
    
    print("Vocab sizes:", tokenizer.get_vocab_sizes())

    new_sequence = ["<bot>", "<special:<sos>>", "<eot>", "<bot>", "<onset:time=1.9000>", "<pitch:pitch=34>", "<velocity:velocity=34>", "<offset:time=1.9700>", "<eot>"]
    new_tokens = tokenizer.tokenize(new_sequence)
    print("New sequence tokens:", len(new_tokens), "tokens, ", len(new_sequence), "words, ", new_tokens)
    new_text = tokenizer.detokenize(new_tokens)
    print("New sequence detokenized:", new_text)