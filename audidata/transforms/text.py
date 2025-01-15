from __future__ import annotations

import re


class TextNormalization:
    def __init__(self) -> None:
        pass

    def __call__(self, x: str) -> str:
        r"""Normalize a string. 

		E.g., "How are you? Fine, thank you!" => "how are you fine thank you"

        From: https://github.com/XinhaoMei/WavCaps/blob/master/captioning/data_handling/text_transform.py
        """

        # Transform to lower case
        x = x.lower()

        # Remove any forgotten space before punctuation and double space
        x = re.sub(r'\s([,.!?;:"](?:\s|$))', r'\1', x).replace('  ', ' ')

        # Remove punctuations
        x = re.sub('[(,.!?;:|*\")]', ' ', x).replace('  ', ' ')

        return x