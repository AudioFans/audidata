import re


class TextNormalization:
    def __init__(self):
        pass

    def __call__(self, data: dict) -> dict:

        sentence = data["caption"]

        # Transform to lower case
        sentence = sentence.lower()

        # Remove any forgotten space before punctuation and double space
        sentence = re.sub(r'\s([,.!?;:"](?:\s|$))', r'\1', sentence).replace('  ', ' ')

        # Remove punctuations
        sentence = re.sub('[(,.!?;:|*\")]', ' ', sentence).replace('  ', ' ')

        data["caption"] = sentence
        
        return data