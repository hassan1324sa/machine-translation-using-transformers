import torch 
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):
    def __init__(self, vocabSize, dModel):
        super().__init__()
        self.dModel = dModel
        self.embedding = nn.Embedding(vocabSize, dModel)
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.dModel) # 512


class PositionalEncoding(nn.Module):
    def __init__(self, dModel, dropout, seqLen):
        super().__init__()
        self.dModel = dModel
        
        self.seqLen = seqLen
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seqLen,dModel)
        #[0,...,350] 
        pos = torch.arange(0,seqLen,dtype=torch.float).unsqueeze(1)

        divTerm = torch.exp(torch.arange(0,dModel,2,dtype=torch.float) * (-math.log(10000.0) / dModel))
        #[0,...,512] ODD numbers *  -4 / 512
        # divTerm = -2048
        # for even data
        pe[:, 0::2] = torch.sin(pos * divTerm)
        # for odd data
        pe[:, 1::2] = torch.cos(pos * divTerm)
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
        self.register_buffer('pe', pe) # why we dont use nn.Parameter here?
        # we don't need that in the model.state_dict()
        # because we dont want to train this buffer
        # and the Positional Encoding is not a learnable part
    def forward(self,x):
        x = x+(self.pe[:,:x.shape[1],:]).requires_grad_(False) # because that parameter dont need to be updated
        return self.dropout(x)


# the difference between Layer Normalization and Batch Normalization 
# is that Layer Normalization normalizes the input of each layer
# while Batch Normalization normalizes the input of each batch
class LayerNormalization(nn.Module):
    def __init__(self, eps=10**-6): #the mean of 10**-6 0.000001
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForward(nn.Module):
    def __init__(self, dModel, dFF=2048, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dModel, dFF)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dFF, dModel)
    def forward(self, x):
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))

class MultiHeadAttention(nn.Module):
    def __init__(self, dModel, nHead, dropout):
        super().__init__()
        self.dModel = dModel
        self.nHead = nHead
        assert dModel % nHead == 0, "dModel should be divisible by nHead"
        self.depth = dModel // nHead

        self.dropout = nn.Dropout(dropout)
        self.Wq = nn.Linear(dModel, dModel) 
        self.Wk = nn.Linear(dModel, dModel) 
        self.Wv = nn.Linear(dModel, dModel) 
        self.Wo = nn.Linear(dModel, dModel)

    @staticmethod
    def attention(q, k, v, mask, dropout):
        depth = q.shape[-1]
        # Transpose only the last two dimensions of k
        attentionScores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(depth)
        if mask is not None:
            attentionScores = attentionScores.masked_fill(mask == 0, -1e9)
        attentionWeights = torch.softmax(attentionScores, dim=-1)
        attentionWeights = dropout(attentionWeights)
        return torch.matmul(attentionWeights, v), attentionScores

    def forward(self, Q, K, V, mask=None):
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)
        # Reshape Q, K, V to (batch, seq_len, n_head, d_head) then transpose to (batch, n_head, seq_len, d_head)
        Q = Q.view(Q.shape[0], Q.shape[1], self.nHead, self.depth).transpose(1, 2)
        K = K.view(K.shape[0], K.shape[1], self.nHead, self.depth).transpose(1, 2)
        V = V.view(V.shape[0], V.shape[1], self.nHead, self.depth).transpose(1, 2)
        x, self.attentionScores = MultiHeadAttention.attention(Q, K, V, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.nHead * self.depth)
        x = self.Wo(x)
        return x



class ResidualConnection(nn.Module):
    def __init__(self,dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))


class EncoderBlock(nn.Module):
    def __init__(self, AttentionBlock, FFN, dropout):
        super().__init__()
        self.Attention = AttentionBlock
        self.feedForward = FFN
        self.residualConnection = nn.ModuleList([ ResidualConnection(dropout) for _ in range(2)])
    def forward(self, x, mask):
        x = self.residualConnection[0](x,lambda x : self.Attention(x,x,x,mask))
        x = self.residualConnection[1](x, self.feedForward)
        return x


class Encoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x,mask)
            return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, selfAttentionBlock, crossAttentionBlock, FFN, dropout):
        super().__init__()
        self.selfAttention = selfAttentionBlock
        self.crossAttention = crossAttentionBlock
        self.feedForward = FFN
        self.residualConnection = nn.ModuleList([ ResidualConnection(dropout) for _ in range(3)])
    # x : input of the decoder 
    # encoderOutput : output of the encoder
    # mask : mask for the encoder
    # tgtMask : target y 
    def forward(self, x, encoderOutput , mask,tgtMash):
        x = self.residualConnection[0](x,lambda x : self.selfAttention(x,x,x,tgtMash))
        x = self.residualConnection[1](x,lambda x : self.crossAttention(x,encoderOutput,encoderOutput,mask))
        x = self.residualConnection[2](x, self.feedForward)
        return x

class Decoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    def forward(self,x,encoderOutput,mask,tgtMask):
        for layer in self.layers:
            x = layer(x,encoderOutput,mask,tgtMask)
            return self.norm(x)


class LinearLayer(nn.Module):  
    def __init__(self, dModel, vocabSize): 
        super().__init__()    
        self.fc = nn.Linear(dModel, vocabSize) 
    def forward(self, x):  
        return torch.log_softmax(self.fc(x), dim = -1)



class Transformer(nn.Module):
    # for translation
    # src = x 
    # tgt = y
    def __init__(self, encoder, decoder, srcEmbedding,tgtEmbedding, srcPos, tgtPos, linearLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.srcEmbedding = srcEmbedding
        self.tgtEmbedding = tgtEmbedding
        self.srcPos = srcPos
        self.tgtPos = tgtPos
        self.linearLayer = linearLayer

    def encode(self,src,mask):
        src = self.srcEmbedding(src)
        src = self.srcPos(src)
        encoderOutput = self.encoder(src,mask)
        return encoderOutput
    
    def decode(self,tgt,mask,encoderOutput,srcMask,tgtMask):
        tgt = self.tgtEmbedding(tgt)    
        tgt = self.tgtPos(tgt)
        decoderOutput = self.decoder(tgt,encoderOutput,srcMask,tgtMask)
        return decoderOutput 
    def forward(self, src, tgt, srcMask, tgtMask):
        encoderOutput = self.encode(src,srcMask)
        decoderOutput = self.decode(tgt,tgtMask,encoderOutput,srcMask,tgtMask)
        return self.linearLayer(decoderOutput)



def buildTransformers(srcVocabSize,tgtVocabSize,srcSeqLen,tgtSeqLen,dModel=512,n=6,h=8,dropout=0.1,dFF=2048):
    # embedding layers
    srcEmbedding = InputEmbeddings(dModel=dModel,vocabSize=srcVocabSize)
    tgtEmbedding = InputEmbeddings(dModel=dModel,vocabSize=tgtVocabSize)
    
    # positional embedding layers
    srcPos = PositionalEncoding(dModel=dModel,dropout=dropout,seqLen=srcSeqLen)
    tgtPos = PositionalEncoding(dModel=dModel,dropout=dropout,seqLen=tgtSeqLen)

    # encoder Blocks
    EncoderBlocks = []
    for _ in range(n):
        encoderAttention = MultiHeadAttention(dModel=dModel, nHead=h, dropout=dropout)
        ffn = FeedForward(dModel=dModel,dFF=dFF,dropout=dropout)
        encoderBlock = EncoderBlock(encoderAttention,FFN=ffn,dropout=dropout)
        EncoderBlocks.append(encoderBlock)
    # decoder Blocks
    decoderBlocks = []
    for _ in range(n):
        decoderAttention = MultiHeadAttention(dModel=dModel, nHead=h, dropout=dropout)
        decoderCrossAttention = MultiHeadAttention(dModel=dModel, nHead=h, dropout=dropout)
        ffn = FeedForward(dModel=dModel,dFF=dFF,dropout=dropout)
        decoderBlock = DecoderBlock(decoderAttention,decoderCrossAttention,FFN=ffn, dropout=dropout)
        decoderBlocks.append(decoderBlock)
    
    # encoder
    encoder = Encoder(nn.ModuleList(EncoderBlocks))
    # decoder
    decoder = Decoder(nn.ModuleList(decoderBlocks))
    # linear layer
    linearLayer = LinearLayer(dModel=dModel,vocabSize=tgtVocabSize)
    # transformer
    transformers = Transformer(encoder, decoder, srcEmbedding, tgtEmbedding, srcPos, tgtPos, linearLayer)
    for param in transformers.parameters():
        if param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)
    return transformers


