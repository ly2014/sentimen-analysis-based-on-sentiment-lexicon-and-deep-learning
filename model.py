import torch
import torch.nn as nn
import torch.nn.functional as F


class SLCABG(nn.Module):
    def __init__(self, n_dim, sentence_length, word_vectors):
        super(SLCABG, self).__init__()
        self.word_embeddings = nn.Embedding.from_pretrained(word_vectors)
        self.word_embeddings.weight.requires_grad = False
        self.convs = nn.ModuleList(
            [nn.Sequential(
                nn.Conv1d(n_dim, 128, h),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(sentence_length - h + 1)
                           ) for h in [3, 4, 5]]
        )
        self.gru = nn.GRU(128*3, 64, batch_first=True, bidirectional=True, dropout=0.4)
        self.weight_W = nn.Parameter(torch.Tensor(128, 128))
        self.weight_proj = nn.Parameter(torch.Tensor(128, 1))
        self.fc = nn.Linear(128, 2)
        nn.init.uniform_(self.weight_W, -0.1, 0.1)
        nn.init.uniform_(self.weight_proj, -0.1, 0.1)

    def forward(self, x):
        embed_x = self.word_embeddings(x)
        embed_x = embed_x.permute(0, 2, 1)
        out = [conv(embed_x) for conv in self.convs]
        out = torch.cat(out, 1)
        out = out.permute(0, 2, 1)
        out, _ = self.gru(out)
        u = torch.tanh(torch.matmul(out, self.weight_W))
        att = torch.matmul(u, self.weight_proj)
        att_score = F.softmax(att, dim=1)
        scored_x = out * att_score
        feat = torch.sum(scored_x, dim=1)
        out = self.fc(feat)
        return out
