# import re


# class TextNormalization:
#     def __init__(self):
#         pass

#     def __call__(self, x: str) -> str:
#         r"""Normalize a string. 
#         Ref: https://github.com/XinhaoMei/WavCaps/blob/master/captioning/data_handling/text_transform.py

#         E.g., "How are you? Fine, thank you!" => "how are you fine thank you"
#         """

#         # Transform to lower case
#         x = x.lower()

#         # Remove any forgotten space before punctuation and double space
#         x = re.sub(r'\s([,.!?;:"](?:\s|$))', r'\1', x).replace('  ', ' ')

#         # Remove punctuations
#         x = re.sub('[(,.!?;:|*\")]', ' ', x).replace('  ', ' ')

#         return x