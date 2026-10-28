from math import gamma
from narwhals.selectors import enum
import matplotlib.pyplot as plt

from gensim.models import Word2Vec
from collections import defaultdict
import torch
import torch.nn as nn

from torchtext.vocab import GloVe, vocab

from torchtext.datasets import AG_NEWS

train_iter_ag_news = AG_NEWS(split="train")
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import find_similar_words

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device:{device}")


GloVe.url["6B"] = "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip"

# creating an instance of the 6B version of Glove() model
glove_vectors_6B = GloVe(
    name="6B"
)  # you can specify the model with the following format: GloVe(name='840B', dim=300)
# You must continue with the 6B model as it is lighter. You can load different pretrained GloVe models from torch() using ```torch.nn.Embedding.from_pretrained```.
# creating another instance of a bigger Glove() model
# glove_vectors_840B = GloVe()


# load the glove model pretrained weights into a PyTorch embedding layer
embeddings_Glove6B = torch.nn.Embedding.from_pretrained(
    glove_vectors_6B.vectors, freeze=True
)


word_to_index = glove_vectors_6B.stoi
print(f"sample index check: {word_to_index["team"]}")

embeddings_Glove6B.weight[word_to_index["team"]]

# an array of example words
words = [
    "taller",
    "short",
    "black",
    "white",
    "dress",
    "pants",
    "big",
    "small",
    "red",
    "blue",
    "smile",
    "frown",
    "race",
    "stroll",
    "tiny",
    "huge",
    "soft",
    "rough",
    "team",
    "individual",
]

embedding_dict_Glove6B = {}
for word in words:
    # Get the index of the word from the vocab to access its embedding
    embedding_vector = embeddings_Glove6B.weight[word_to_index[word]]
    if embedding_vector is not None:
        # words not found in the embedding index will be skipped.
        # add the embedding vector of word to the embedding_dict_Glove6b
        embedding_dict_Glove6B[word] = embedding_vector


# Call the function to find similar words
target_word = "small"
top_k = 2
similar_words = find_similar_words(target_word, embedding_dict_Glove6B, top_k)

# Print the similar words
print("{} most similar words to {}:".format(top_k, target_word), similar_words)

# Train a word2vec model from gensim
sentences = [
    ["I", "like", "to", "eat", "pizza"],
    ["Pizza", "is", "my", "favorite", "food"],
    ["I", "enjoy", "eating", "pasta"],
]
sentences = [[word.lower() for word in sentence] for sentence in sentences]

# Create an instance of Word2Vec model
# The `size` parameter specifies the dimensionality of the word embeddings (in this case, 100).
# The `window` parameter determines the size of the context window.
# The `min_count` parameter sets the minimum frequency of a word to be included in the training process.
# Finally, the `workers` parameter controls the number of threads used for training.
w2v_model = Word2Vec(sentences, vector_size=100, window=3, min_count=1, workers=4)

# build vocab using the training data
w2v_model.build_vocab(sentences, progress_per=10000)

# tain the model on your training data
w2v_model.train(
    sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1
)

# That's it! You've trained a word2vec model using the `gensim` library. You can now access the word embeddings using `model.wv` and explore various operations such as finding similar words, calculating word similarities, and more.

# Use the trained model to find similar words to "pizza" and calculate the similarity between "pizza" and "pasta".

# Finding similar words
similar_words = w2v_model.wv.most_similar("pizza")
print(f"similar words to 'pizza': {similar_words}")

# calculate the word similarity
similarity = w2v_model.wv.similarity("pizza", "pasta")
print(f"similarity between 'pizza' and 'pasta': {similarity}")


# The word embeddings obtained from the model would be more meaningful and informative with larger and more diverse training data.
# Use the trained model to create a PyTorch embedding layer (just like what you did with the pretrained GloVe model) and use it in any task as an embedding layer.
word_vectors = w2v_model.wv
word_to_index = {word: index for index, word in enumerate(word_vectors.index_to_key)}

embedding_dim = w2v_model.vector_size
embedding = torch.nn.Embedding(len(word_vectors.index_to_key), embedding_dim)

# SAFE loading (no NumPy -> Torch conversion)
embedding.weight.data = torch.tensor(word_vectors.vectors.tolist(), dtype=torch.float32)

# example usage
word = "pizza"
word_index = word_to_index[word]
word_embedding = embedding(torch.LongTensor([word_index]))

print(f"word: {word}, embedding shape: {word_embedding.shape}")


# Text classification using pretrained word embeddings
# You are ready to use the embeddings in a task, then.
# Let's use the pretrained embeddings to classify text data into topics:

# build vocab form glove_vectors
# vocab(ordered_dict: Dict, min_freq,: int = 1, specials: Optional[List[str]] = None)
vocab = vocab(glove_vectors_6B.stoi, 0, specials=("<unk>", "<pad>"))
vocab.set_default_index(vocab["<unk>"])

print(f"vocabs: {vocab(["<unk>", "Hello", "hello"])}")

# define tokenizer
tokenizer = get_tokenizer("basic_english")

# define functions to process text and labels
# split the dataset into training and testing iterators
train_iter, test_iter = AG_NEWS()

# convert the training and testing iterators to map-style datasets.
train_dataset = to_map_style_dataset(train_iter)
test_dataset = to_map_style_dataset(test_iter)

# determine the number of samples to be used for training and validation (5% for validation)
num_train = int(len(train_dataset) * 0.85)

# randomly spliting the training datset into training and validation datasets using 'random_split'.
# the training dataset will contain 95% of the samples, and the validation dataset will contain the remaining 5%.

print(f"num_train:{num_train}")

split_train_, split_valid_ = random_split(
    train_dataset, [num_train, len(train_dataset) - num_train]
)

# define class labels
ag_news_label = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tec"}
"""ag_news_label[y]"""
num_class = len(set([label for (label, text) in train_iter]))

# collate data in batches


def text_pipeline(x):
    x = x.lower()  # you need this as your vocab is in lowercase
    return vocab(tokenizer(x))


def label_pipeline(x):
    return int(x) - 1


# create label, text and offset for each batch of data
# text is the concatenated text for all text data in the batch
# you need to have the offsets (the end of the text index) for later when you separate texts and predict their label
def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for _label, _text in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))

    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)


# create data loaders for train, validation and test splits
BATCH_SIZE = 64

train_dataloader = DataLoader(
    split_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)

valid_dataloader = DataLoader(
    split_valid_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)

test_dataloader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)

label, text, offsets = next(iter(train_dataloader))
print(f"label: {label} text: {text} offsets: {offsets}")
print(
    f"label.shape: {label.shape} text.shape: {text.shape} offsets.shape: {offsets.shape}"
)


class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(
            glove_vectors_6B.vectors, freeze=True
        )
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text)
        # you get the average of word embeddings in the text
        means = []
        for i in range(1, len(offsets)):
            # this is like eme
            text_tmp = embedded[offsets[i - 1] : offsets[i]]
            means.append(text_tmp.mean(0))

        return self.fc(torch.stack(means))


# define an evaluate function to calculate the accuracy of model
def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)

            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count


# Create an instance of the model and check its prediction power before training:

# Define hyperparameters
vocab_size = len(vocab)
embedding_dim = 300
# Initialize the model
model = TextClassificationModel(vocab_size, embedding_dim, num_class).to(device)

avg_acc = evaluate(test_dataloader)
print(f"test_eval: {avg_acc}")


def train_TextClassification(model, dataloader, criterion, optimizer, epochs=10):

    cum_loss_list = []
    acc_epoch = []
    acc_old = 0

    for epoch in tqdm(range(1, EPOCHS + 1)):
        model.train()
        cum_loss = 0

        for idx, (label, text, offsets) in enumerate(train_dataloader):
            means = []
            optimizer.zero_grad()

            predicted_label = model(text, offsets)

            loss = criterion(predicted_label, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            cum_loss += loss.item()

        cum_loss_list.append(cum_loss / len(train_dataloader))
        accu_val = evaluate(valid_dataloader)
        acc_epoch.append(accu_val)

        if accu_val > acc_old:
            acc_old = accu_val
            torch.save(model.state_dict(), "./word2vec2/my_model.pt")

    return model, cum_loss_list, acc_epoch


# Define hyperparameters
LR = 0.1
EPOCHS = 10

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)

model, cum_loss_list, acc_epoch = train_TextClassification(model, train_dataloader, criterion, optimizer, epochs=EPOCHS)

def plot(COST,ACC):
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.plot(COST, color=color)
    ax1.set_xlabel('epoch', color=color)
    ax1.set_ylabel('total loss', color=color)
    ax1.tick_params(axis='y', color=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('accuracy', color=color)  # you already handled the x-label with ax1
    ax2.plot(ACC, color=color)
    ax2.tick_params(axis='y', color=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.savefig("./word2vec2/fig")


plot(cum_loss_list,acc_epoch)

evaluate(test_dataloader)


# You've acquired the skills to create and train embedding models, as well as utilize large pretrained models for practical applications. This knowledge opens up a world of possibilities where you can leverage the power of embeddings to improve various natural language processing tasks. 