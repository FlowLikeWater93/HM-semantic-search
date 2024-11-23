import re
import nltk
# nltk.download('punkt_tab')

def generate_tokens(text):
    raw_vocab = []
    vocab = {'<UNK>': 0}
    for i in range(len(text)):
        for token in nltk.word_tokenize(text[i]) :
            raw_vocab.append(token)
    vocab_set = set(raw_vocab)
    i = 1
    for item in vocab_set:
        vocab[item] = i
        i += 1
    return vocab


def clean_text(text):
    # convert to lower case
    input = text.lower()
    # remove punctuations
    input = re.sub('-', ' ', input)
    input = re.sub('[\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\]\^\_\`\{\|\}\~]', '', input)
    # only select words with lower case letters
    # ignore numbers
    output = ''
    broken_input = re.findall('[a-z]+', input)
    for i in range(len(broken_input)):
        output = output + ' ' + broken_input[i]
    # return final clean text
    return output[1:]


def tokenize(text, vocab, max_seq_len=50):
    tokens = []
    for i in range(len(text)):
        # enforce max sequence length
        tokns = []
        for tok in nltk.word_tokenize(text[i]) :
            try :
                tokns.append(vocab[tok])
            except :
                tokns.append(0)
        # Padding
        if len(tokns)<max_seq_len:
            tokns = tokns+([0]* (max_seq_len-len(tokns)))
        else :
            tokns = tokns[:max_seq_len]
        # append to full list
        tokens.append(tokns)
    # return tokens
    return tokens
