from collections import Counter, defaultdict


def initialize_bpe(pre_tokenized_cache) -> tuple[Counter[tuple], defaultdict[tuple, set]]:
    pair_counts = Counter()
    pair_to_subword_map = defaultdict(set)

    for subword in pre_tokenized_cache:
        for i in range(len(subword)-1):
            pair = (subword[i], subword[i+1])
            pair_counts[pair] += pre_tokenized_cache[subword]
            pair_to_subword_map[pair].add(subword)
    
    return pair_counts, pair_to_subword_map


def bpe_merge(cache, pair_counts, pair_to_subword_map,  vocabulary, merges):
    max_occuring_pair = max(pair_counts, key=lambda x: (pair_counts[x], x))
    next_vocab_item = len(vocabulary)
    left_token_id, right_token_id = max_occuring_pair
    vocabulary[next_vocab_item] = vocabulary[left_token_id] + vocabulary[right_token_id]
    merges.append(max_occuring_pair)
    

    # these subwords will be deleted(from the cache) and replaced with subwords with merged pair
    subwords_to_process = list(pair_to_subword_map.get(max_occuring_pair, []))
    
    # loop over subwords and reduce the counts of all pairs in subword that will change
    # later pairs in the new subword will be incremented, thus preventing re-calculating counts
    for subword in subwords_to_process:
        freq = cache[subword]

        del cache[subword]

        for i in range(len(subword)-1):
            pair = (subword[i], subword[i+1])
            pair_counts[pair] -= freq # reduce the counts in old sub-word

            if pair_counts[pair] <= 0:
                del pair_counts[pair] # not necessary to do, but will speedup max operation
            
            # since subword is deleted from cache, and subword pairs have pointers to the subword, remove them from the map
            pair_to_subword_map[pair].discard(subword) # discard = remove it exists
            
        new_sub_word = []
        i = 0
        while i < len(subword):
            if i < len(subword)-1 and (subword[i], subword[i+1]) == max_occuring_pair:
                new_sub_word.append(next_vocab_item)
                i += 2
            else:
                new_sub_word.append(subword[i])
                i += 1
        
        cache[tuple(new_sub_word)] = freq

        for i in range(len(new_sub_word)-1):
            pair = (new_sub_word[i], new_sub_word[i+1])
            pair_counts[pair] += freq
            pair_to_subword_map[pair].add(tuple(new_sub_word))

    return cache, pair_counts, pair_to_subword_map, merges    
