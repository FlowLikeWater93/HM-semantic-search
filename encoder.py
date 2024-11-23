import torch
import math
import numpy as np

class Encoder(torch.nn.Module):
    def __init__(self, vocab_len):
        super().__init__()
        self.embed_len = 128
        # Embedding layer : input = vocab length & output = embedding length
        self.embedding_layer = torch.nn.Embedding(num_embeddings=vocab_len, embedding_dim=self.embed_len)
        # Encoder
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.embed_len, nhead=4, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)
        # Linear layer
        self.linear = torch.nn.Linear(self.embed_len, 5)
        # Softmax activation layer
        self.sm = torch.nn.Softmax(dim=1)
        # Initialize weights
        self.init_weights()



    def positional_encoding(self, batch_size, max_seq_length, d_model, n=10000):
        # generate an empty matrix for the positional encodings (pe)
        pe = np.zeros(max_seq_length*d_model).reshape(max_seq_length, d_model)
        # for each position
        # for b in np.arange(batch_size) :
        for k in np.arange(max_seq_length):
            # for each dimension
            for i in np.arange(d_model//2):
                # calculate the internal value for sin and cos
                theta = k / (n ** ((2*i)/d_model))
                # even dims: sin
                pe[k, 2*i] = math.sin(theta)
                # odd dims: cos
                pe[k, 2*i+1] = math.cos(theta)
        return torch.from_numpy(pe)



    def init_weights(self):
        initrange = 0.1
        self.embedding_layer.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)



    def sentence_embedding(self, X):
        # Pass to embedding layer
        embeddings = self.embedding_layer(X)
        # Generate positional encoding tensor
        pe = self.positional_encoding(embeddings.shape[0], embeddings.shape[1], embeddings.shape[2])
        # add to embeddings
        embeddings_pe = embeddings + pe
        # Pass to transfomer encoder
        encoder_output = self.transformer_encoder(embeddings)
        # average word embeddings to generate sentence embedding
        return encoder_output.mean(dim=1)



    def forward(self, X):
        # Pass to embedding layer
        embeddings = self.embedding_layer(X)
        # Generate positional encoding tensor
        pe = self.positional_encoding(embeddings.shape[0], embeddings.shape[1], embeddings.shape[2])
        # add to embeddings
        embeddings_pe = embeddings + pe
        # Pass to transfomer encoder
        encoder_output = self.transformer_encoder(embeddings)
        # Find word vector with max value in axis 1
        classification_head = encoder_output.max(dim=1)[0]
        # pass the output of the encoder to the linear layer
        output = self.linear(classification_head)
        # softmax for multiclass classification
        return self.sm(output)
