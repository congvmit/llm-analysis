from collections import Counter, OrderedDict

from torchtext.vocab import vocab

counter = Counter(["a", "a", "b", "b", "b"])
sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
ordered_dict = OrderedDict(sorted_by_freq_tuples)
v1 = vocab(ordered_dict)


import io

from torchtext.vocab import build_vocab_from_iterator

file_path = "data.txt"


def yield_tokens(file_path):
    with io.open(file_path, encoding="utf-8") as f:
        for line in f:
            yield line.strip().split()


for d in yield_tokens(file_path):
    print(d)

vocab = build_vocab_from_iterator(yield_tokens(file_path), specials=["<unk>"])

# DEBUG
import mipkit

mipkit.debug.set_trace()
exit()
