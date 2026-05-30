from collections import Counter

def get_counts(cache):
    """
    Get pair counts from the current pretokenized and merged cache
    """
    counts = Counter()

    for byte_subword in cache:
        for i in range(len(byte_subword)-1):
            key = (byte_subword[i], byte_subword[i+1])
            counts[key] += cache[byte_subword]
    
    return counts

def merge(cache, counts, vocabulary, merges):
    # add merged to vocab and to merges list
    # change cache to reflect merges
    
    max_occuring_pair = max(counts, key=lambda x: (counts[x], x))
    next_vocab_item = len(vocabulary)
    vocabulary[max_occuring_pair] = next_vocab_item
    merges.append(max_occuring_pair)

    new_cache = Counter()

    for byte_subword in cache:
        i = 0
        new_key = []
        while i < len(byte_subword):
            if i < len(byte_subword)-1 and (byte_subword[i], byte_subword[i+1]) == max_occuring_pair:
                new_key.append(next_vocab_item)
                i += 2
            else:
                new_key.append(byte_subword[i])
                i += 1
        
        new_cache[tuple(new_key)] = cache[byte_subword]
    
    return new_cache, merges