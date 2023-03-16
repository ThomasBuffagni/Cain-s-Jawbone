import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from itertools import accumulate


class SentencePairsDataset(Dataset):
    def __init__(self, embeddings: dict) -> None:
        super().__init__()
        self.embeddings = embeddings

        # For each sentence we provide one positive and one negative example except for the last one
        counts = [(len(page) - 1) for page in self.embeddings.values()]
        self.accum_counts = list(accumulate(counts))
        self.accum_counts.append(self.accum_counts[-1] + 1)
        self.len_ = 2 * sum(counts)

    def __len__(self) -> int:
        return self.len_

    def __getitem__(self, index) -> np.array:
        page_number, sentence_index = self.get_tuple_index(index=index // 2)

        embed = self.embeddings[page_number][sentence_index]

        if index < self.len_ // 2:
            other_embed = self.get_next_sentence_embed(page_number, sentence_index)
            label = 1.
        else:
            other_embed = self.get_not_next_sentence_embed(page_number, sentence_index)
            label = 0.
        
        return np.concatenate([embed, other_embed]), np.array(label, dtype=np.float32)

    def get_next_sentence_embed(self, page_number, sentence_index):
        return self.embeddings[page_number][sentence_index + 1]

    def get_not_next_sentence_embed(self, page_number, sentence_index):
        all_indices = list(range(len(self.embeddings[page_number])))
        all_indices.remove(sentence_index)
        all_indices.remove(sentence_index + 1)
        other_index = np.random.choice(all_indices, 1)[0]

        return self.embeddings[page_number][other_index]

    def get_tuple_index(self, index):
        for i, count in enumerate(self.accum_counts):
            if index < count:
                page_number = i + 1
                break
        
        sentence_index = index
        if i > 0:
            sentence_index -= self.accum_counts[i-1]

        return page_number, sentence_index


class SentencePairsClassifier(nn.Module):
    def __init__(self, layer_sizes) -> None:
        super().__init__()

        self.model = nn.Sequential(
            *[
                self.block(
                    in_features=layer_sizes[i-1],
                    out_features=layer_sizes[i],
                    activation='sigmoid' if i == len(layer_sizes) - 1 else 'relu',
                )
                for i in range(1, len(layer_sizes))
            ]
        )

    def block(self, in_features: int, out_features: int, activation: str):
        if activation == 'relu':
            activation_fn = nn.ReLU()
        elif activation == 'sigmoid':
            activation_fn = nn.Sigmoid()

        return nn.Sequential(
                nn.Linear(in_features, out_features),
                activation_fn,
        )
    
    def forward(self, x):
        return self.model(x)