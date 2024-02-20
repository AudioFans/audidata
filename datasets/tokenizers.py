import re
import numpy as np
from .constants import *


class BaseTokenizer:
    def __init__(self):
        
        self.strings = ["<pad>", "<sos>", "<eos>", "<unk>"]
        self.vocab_size = len(self.strings)

        self.token_to_string = {token: string for token, string in enumerate(self.strings)}
        self.string_to_token = {string: token for token, string in enumerate(self.strings)}
        
    def itos(self, token):
        assert 0 <= token < self.vocab_size
        return self.token_to_string[token]

    def stoi(self, string):
        if string in self.strings:
            return self.string_to_token[string]


class TimeTokenizer:
    def __init__(self):
        self.vocab_size = SEGMENT_SECONDS * TARGET_FRAMES_PER_SECONDS + 1
        self.frames_per_second = TARGET_FRAMES_PER_SECONDS

    def itos(self, token):
        assert 0 <= token < self.vocab_size

        time = token / self.frames_per_second
        string = "<time>={}".format(time)
        return string

    def stoi(self, string):
        if "<time>" in string:
            time = float(re.search('<time>=(.*)', string).group(1))
            token = int(time)
            # token = int(time * self.frames_per_second)
            return token


class InstrumentTokenizer:
    def __init__(self):
        self.vocab_size = 128

    def itos(self, token):
        assert 0 <= token < self.vocab_size

        string = "<inst>={}".format(token)
        return string

    def stoi(self, string):
        if "<inst>" in string:
            token = int(re.search('<inst>=(.*)', string).group(1))
            return token


class InstrumentTokenizer129:
    def __init__(self):
        self.vocab_size = 129

    def itos(self, token):
        assert 0 <= token < self.vocab_size

        string = "<inst>={}".format(token)
        return string

    def stoi(self, string):
        if "<inst>" in string:
            token = int(re.search('<inst>=(.*)', string).group(1))
            return token


class SoundEventTokenizer:
    def __init__(self):
        self.vocab_size = 1024

    def itos(self, token):
        assert 0 <= token < self.vocab_size

        string = "<sound_event>={}".format(token)
        return string

    def stoi(self, string):
        if "<sound_event>" in string:
            token = int(re.search('<sound_event>=(.*)', string).group(1))
            return token


class PitchTokenizer:
    def __init__(self):
        self.vocab_size = 128

    def itos(self, token):
        assert 0 <= token < self.vocab_size

        string = "<pitch>={}".format(token)
        return string

    def stoi(self, string):
        if "<pitch>" in string:
            token = int(re.search('<pitch>=(.*)', string).group(1))
            return token


class DrumsTokenizer:
    def __init__(self):
        self.vocab_size = 128

    def itos(self, token):
        assert 0 <= token < self.vocab_size

        string = "<drum>={}".format(token)
        return string

    def stoi(self, string):
        if "<drum>" in string:
            token = int(re.search('<drum>=(.*)', string).group(1))
            return token


class VelocityTokenizer:
    def __init__(self):
        self.vocab_size = 128

    def itos(self, token):
        assert 0 <= token < self.vocab_size

        string = "<velocity>={}".format(token)
        return string

    def stoi(self, string):
        if "<velocity>" in string:
            token = int(re.search('<velocity>=(.*)', string).group(1))
            return token


class TieTokenizer:
    def __init__(self):
        self.vocab_size = 4
        self.strings = ["<tie>=on", "<tie>=off"]

        self.token_to_string = {token: string for token, string in enumerate(self.strings)}
        self.string_to_token = {string: token for token, string in enumerate(self.strings)}
        
    def itos(self, token):
        assert 0 <= token < self.vocab_size
        return self.token_to_string[token]

    def stoi(self, string):
        if string in self.strings:
            return self.string_to_token[string]


class MidiControllerTokenizer:
    def __init__(self):
        self.vocab_size = 128

    def itos(self, token):
        assert 0 <= token < self.vocab_size

        string = "<midi_controller>={}".format(token)
        return string

    def stoi(self, string):
        if "<midi_controller>" in string:
            token = int(re.search('<midi_controller>=(.*)', string).group(1))
            return token


class KeyTokenizer:
    def __init__(self):
        self.vocab_size = 32

    def itos(self, token):
        assert 0 <= token < self.vocab_size

        string = "<key>={}".format(token)
        return string

    def stoi(self, string):
        if "<key>" in string:
            token = int(re.search('<key>=(.*)', string).group(1))
            return token


class ChordRootTokenizer:
    def __init__(self):
        self.vocab_size = 32

    def itos(self, token):
        assert 0 <= token < self.vocab_size

        string = "<chord_root>={}".format(token)
        return string

    def stoi(self, string):
        if "<chord_root>" in string:
            token = int(re.search('<chord_root>=(.*)', string).group(1))
            return token


class ChordPlusTokenizer:
    def __init__(self):
        self.vocab_size = 32

    def itos(self, token):
        assert 0 <= token < self.vocab_size

        string = "<chord_plus>={}".format(token)
        return string

    def stoi(self, string):
        if "<chord_plus>" in string:
            token = int(re.search('<chord_plus>=(.*)', string).group(1))
            return token


class TimeSignatureTokenizer:
    def __init__(self):
        self.vocab_size = 32

    def itos(self, token):
        assert 0 <= token < self.vocab_size

        string = "<time_signature>={}".format(token)
        return string

    def stoi(self, string):
        if "<time_signature>" in string:
            token = int(re.search('<time_signature>=(.*)', string).group(1))
            return token


class BarTokenizer:
    def __init__(self):
        self.vocab_size = 1001

    def itos(self, token):
        assert 0 <= token < self.vocab_size

        string = "<bar>={}".format(token)
        return string

    def stoi(self, string):
        if "<bar>" in string:
            token = int(re.search('<bar>=(.*)', string).group(1))
            return token


class BeatTokenizerDownUp:
    def __init__(self):

        self.vocab_size = 4
        self.strings = ["<beat_type>=down", "<beat_type>=up"]

        self.token_to_string = {token: string for token, string in enumerate(self.strings)}
        self.string_to_token = {string: token for token, string in enumerate(self.strings)}

    def itos(self, token):
        assert 0 <= token < self.vocab_size
        return self.token_to_string[token]

    def stoi(self, string):
        if string in self.strings:
            return self.string_to_token[string]

    # def itos(self, token):
    #     assert 0 <= token < self.vocab_size

    #     if token == 0:
    #         beat_type = "down"
    #     elif token == 1:
    #         beat_type = "up"
    #     else:
    #         raise NotImplementedError

    #     string = "<beat_type>={}".format(beat_type)
    #     return string

    # def stoi(self, string):
    #     if "<beat_type>" in string:
    #         token = int(re.search('<beat_type>=(.*)', string).group(1))
    #         return token


class BeatTokenizer:
    def __init__(self):
        self.vocab_size = 16

    def itos(self, token):
        assert 0 <= token < self.vocab_size

        string = "<beat>={}".format(token)
        return string

    def stoi(self, string):
        if "<beat>" in string:
            token = int(re.search('<beat>=(.*)', string).group(1))
            return token


class SubBeatTokenizer:
    def __init__(self):
        self.vocab_size = 32

    def itos(self, token):
        assert 0 <= token < self.vocab_size

        string = "<subbeat>={}".format(token)
        return string

    def stoi(self, string):
        if "<subbeat>" in string:
            token = int(re.search('<subbeat>=(.*)', string).group(1))
            return token


class StructureTokenizer:
    def __init__(self):
        self.vocab_size = 32

    def itos(self, token):
        assert 0 <= token < self.vocab_size

        string = "<structure>={}".format(token)
        return string

    def stoi(self, string):
        if "<structure>" in string:
            token = int(re.search('<structure>=(.*)', string).group(1))
            return token


class TextTokenizer:
    def __init__(self):
        self.vocab_size = 20000

    def itos(self, token):
        assert 0 <= token < self.vocab_size

        string = "<text>={}".format(token)
        return string

    def stoi(self, string):
        if "<text>" in string:
            token = int(re.search('<text>=(.*)', string).group(1))
            return token


class ImageTokenizer:
    def __init__(self):
        self.vocab_size = 20000

    def itos(self, token):
        assert 0 <= token < self.vocab_size

        string = "<image>={}".format(token)
        return string

    def stoi(self, string):
        if "<image>" in string:
            token = int(re.search('<image>=(.*)', string).group(1))
            return token


class GtzanGenreTokenizer:
    def __init__(self):
        self.vocab_size = 10

        genres = ["blues", "classical", "country", "disco", "hiphop", 
            "jazz", "metal", "pop", "reggae", "rock"]

        self.idx_to_genre = {idx: genre for idx, genre in enumerate(genres)}
        self.genre_to_idx = {genre: idx for idx, genre in enumerate(genres)}

    def itos(self, token):
        assert 0 <= token < self.vocab_size

        string = "<genre>={}".format(self.idx_to_genre[token])
        return string

    def stoi(self, string):
        if "<genre>" in string:
            genre = re.search('<genre>=(.*)', string).group(1)
            token = self.genre_to_idx[genre]
            return token


class Tokenizer:
    def __init__(self):
        self.tokenizers = [
            BaseTokenizer(), 
            TimeTokenizer(), 
            InstrumentTokenizer(),
            SoundEventTokenizer(),
            PitchTokenizer(),
            DrumsTokenizer(),
            VelocityTokenizer(),
            MidiControllerTokenizer(),
            KeyTokenizer(),
            ChordRootTokenizer(),
            ChordPlusTokenizer(),
            BarTokenizer(),
            BeatTokenizerDownUp(),
            BeatTokenizer(),
            SubBeatTokenizer(),
            StructureTokenizer(),
        ]

        self.vocab_size = np.sum([tokenizer.vocab_size for tokenizer in self.tokenizers])

        # for tokenizer in self.tokenizers:
        #     print(tokenizer.vocab_size)

    def itos(self, token):
        assert 0 <= token < self.vocab_size

        for tokenizer in self.tokenizers:
            if token >= tokenizer.vocab_size:
                token -= tokenizer.vocab_size
            else:
                break
            
        return tokenizer.itos(token)

    def stoi(self, string):
        
        start_token = 0

        for tokenizer in self.tokenizers:
            
            token = tokenizer.stoi(string)
            
            if token is not None:
                return start_token + token
            else:
                start_token += tokenizer.vocab_size


class Tokenizer2:
    def __init__(self):
        self.tokenizers = [
            BaseTokenizer(), 
            TimeTokenizer(), 
            InstrumentTokenizer(),
            SoundEventTokenizer(),
            PitchTokenizer(),
            DrumsTokenizer(),
            VelocityTokenizer(),
            TieTokenizer(),
            MidiControllerTokenizer(),
            KeyTokenizer(),
            ChordRootTokenizer(),
            ChordPlusTokenizer(),
            BarTokenizer(),
            BeatTokenizerDownUp(),
            BeatTokenizer(),
            SubBeatTokenizer(),
            StructureTokenizer(),
        ]

        self.vocab_size = np.sum([tokenizer.vocab_size for tokenizer in self.tokenizers])

        # for tokenizer in self.tokenizers:
        #     print(tokenizer.vocab_size)

    def itos(self, token):
        assert 0 <= token < self.vocab_size

        for tokenizer in self.tokenizers:
            if token >= tokenizer.vocab_size:
                token -= tokenizer.vocab_size
            else:
                break
            
        return tokenizer.itos(token)

    def stoi(self, string):
        
        start_token = 0

        for tokenizer in self.tokenizers:
            
            token = tokenizer.stoi(string)
            
            if token is not None:
                return start_token + token
            else:
                start_token += tokenizer.vocab_size

        raise NotImplementedError("{} is not supported!".format(string))


# class Tokenizer3:
#     def __init__(self):
#         self.tokenizers = [
#             BaseTokenizer(),
#             TimeTokenizer(),
#             InstrumentTokenizer129(),
#             SoundEventTokenizer(),
#             PitchTokenizer(),
#             DrumsTokenizer(),
#             VelocityTokenizer(),
#             TieTokenizer(),
#             MidiControllerTokenizer(),
#             KeyTokenizer(),
#             ChordRootTokenizer(),
#             ChordPlusTokenizer(),
#             BarTokenizer(),
#             BeatTokenizerDownUp(),
#             BeatTokenizer(),
#             SubBeatTokenizer(),
#             StructureTokenizer(),
#         ]
#
#         self.vocab_size = np.sum([tokenizer.vocab_size for tokenizer in self.tokenizers])
#
#         # for tokenizer in self.tokenizers:
#         #     print(tokenizer.vocab_size)
#
#     def itos(self, token):
#         assert 0 <= token < self.vocab_size
#
#         for tokenizer in self.tokenizers:
#             if token >= tokenizer.vocab_size:
#                 token -= tokenizer.vocab_size
#             else:
#                 break
#
#         return tokenizer.itos(token)
#
#     def stoi(self, string):
#
#         start_token = 0
#
#         for tokenizer in self.tokenizers:
#
#             token = tokenizer.stoi(string)
#
#             if token is not None:
#                 return start_token + token
#             else:
#                 start_token += tokenizer.vocab_size
#
#         raise NotImplementedError("{} is not supported!".format(string))

class Tokenizer3:
    def __init__(self):
        self.tokenizers = [
            BaseTokenizer(),
            TimeTokenizer(),
            PitchTokenizer(),
            VelocityTokenizer(),
        ]

        self.vocab_size = np.sum([tokenizer.vocab_size for tokenizer in self.tokenizers])

        # for tokenizer in self.tokenizers:
        #     print(tokenizer.vocab_size)

    def itos(self, token):
        assert 0 <= token < self.vocab_size

        for tokenizer in self.tokenizers:
            if token >= tokenizer.vocab_size:
                token -= tokenizer.vocab_size
            else:
                break

        return tokenizer.itos(token)

    def stoi(self, string):

        start_token = 0

        for tokenizer in self.tokenizers:

            token = tokenizer.stoi(string)

            if token is not None:
                return start_token + token
            else:
                start_token += tokenizer.vocab_size

        raise NotImplementedError("{} is not supported!".format(string))

class Tokenizer3Gtzan:
    def __init__(self):
        self.tokenizers = [
            BaseTokenizer(), 
            TimeTokenizer(), 
            InstrumentTokenizer129(),
            SoundEventTokenizer(),
            PitchTokenizer(),
            DrumsTokenizer(),
            VelocityTokenizer(),
            TieTokenizer(),
            MidiControllerTokenizer(),
            KeyTokenizer(),
            ChordRootTokenizer(),
            ChordPlusTokenizer(),
            BarTokenizer(),
            BeatTokenizerDownUp(),
            BeatTokenizer(),
            SubBeatTokenizer(),
            StructureTokenizer(),
            GtzanGenreTokenizer(),
        ]

        self.vocab_size = np.sum([tokenizer.vocab_size for tokenizer in self.tokenizers])

        # for tokenizer in self.tokenizers:
        #     print(tokenizer.vocab_size)

    def itos(self, token):
        assert 0 <= token < self.vocab_size

        for tokenizer in self.tokenizers:
            if token >= tokenizer.vocab_size:
                token -= tokenizer.vocab_size
            else:
                break
            
        return tokenizer.itos(token)

    def stoi(self, string):
        
        start_token = 0

        for tokenizer in self.tokenizers:
            
            token = tokenizer.stoi(string)
            
            if token is not None:
                return start_token + token
            else:
                start_token += tokenizer.vocab_size

        raise NotImplementedError("{} is not supported!".format(string))


def add():

    base_tokenizer = BaseTokenizer()
    
    if True:
        time_tokenizer = TimeTokenizer()
        string = time_tokenizer.itos(123)
        token = time_tokenizer.stoi(string)

        inst_tokenizer = InstrumentTokenizer()
        string = inst_tokenizer.itos(123)
        token = inst_tokenizer.stoi(string)

        tokenizer = SoundEventTokenizer()
        string = tokenizer.itos(123)
        token = tokenizer.stoi(string)

    tokenizer = Tokenizer3Gtzan()
    tokenizer.itos(6100)

    from IPython import embed; embed(using=False); os._exit(0)



if __name__ == '__main__':

    add()