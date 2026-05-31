from pretokenization_utils import pretokenize_file
from tokenizer.training_utils import initialize_bpe, bpe_merge



NUM_CHUNKS = 4

def bpe_train(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    pretokenized_cache = pretokenize_file(
        input_path,
        special_tokens,
        NUM_CHUNKS
    )

    # int to bytes
    vocabulary = {i:bytes([i]) for i in range(256)}
    for i in range(len(special_tokens)):
        vocabulary[256+i] = special_tokens[i].encode('utf-8')
    
    num_merges = vocab_size - len(vocabulary)
    merges = []
    pair_counts, pair_to_subword_map =  initialize_bpe(pretokenized_cache)
    cache = pretokenized_cache
    for i in range(num_merges):
        cache, pair_counts, pair_to_subword_map, merges = bpe_merge(cache, pair_counts, pair_to_subword_map, vocabulary, merges)
    
    byte_merges = [(vocabulary[i], vocabulary[j])  for i, j in merges]

    return vocabulary, byte_merges 
