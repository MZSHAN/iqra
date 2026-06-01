import pickle
from tqdm import tqdm

from pretokenization_utils import pretokenize_file
from training_utils import initialize_bpe, bpe_merge

NUM_CHUNKS = 4

def bpe_train(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    print ("Starting pre-tokenization")
    pretokenized_cache = pretokenize_file(
        input_path,
        special_tokens,
        NUM_CHUNKS
    )
    print ("Pre-tokenization complete")
    # int to bytes
    vocabulary = {i:bytes([i]) for i in range(256)}
    for i in range(len(special_tokens)):
        vocabulary[256+i] = special_tokens[i].encode('utf-8')
    
    num_merges = vocab_size - len(vocabulary)
    merges = []
    pair_counts, pair_to_subword_map =  initialize_bpe(pretokenized_cache)
    cache = pretokenized_cache

    progress_bar = tqdm(range(num_merges), desc="BPE Merging", unit="merge")
    
    for i in progress_bar:
        cache, pair_counts, pair_to_subword_map, merges = bpe_merge(cache, pair_counts, pair_to_subword_map, vocabulary, merges)
    
    byte_merges = [(vocabulary[i], vocabulary[j])  for i, j in merges]

    return vocabulary, byte_merges 


if __name__ == "__main__":
    input_file = "../data/owt_train.txt"
    vocab_size = 32000
    special_tokens = ["<|endoftext|>"]

    vocabulary, byte_merges = bpe_train(input_file, vocab_size, special_tokens)
    data_to_save = {
        "vocabulary": vocabulary,
        "byte_merges": byte_merges
    }

    # Serialize to disk
    with open("bpe_owt_train.pkl", "wb") as f:
        pickle.dump(data_to_save, f)

    print("Vocabulary and merges saved to bpe_owt_train.pkl")
