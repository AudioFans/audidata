from audidata.transforms.text import TextNormalization


if __name__ == '__main__':

    transform = TextNormalization()

    # input_txt = "How are you? Fine, thank you!"
    output_txt = transform(input_txt)

    print("Input: {}".format(input_txt))
    print("Output: {}".format(output_txt))