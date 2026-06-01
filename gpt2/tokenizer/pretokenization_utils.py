import os
import multiprocessing
import regex as re
from typing import BinaryIO, Iterator
from collections import Counter

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# ==========================================
# MACRO-CHUNKING 
# ==========================================
def find_chunk_boundaries(
    file: BinaryIO,
    num_chunks: int,
    split_special_tokens: list[bytes]
) -> list[int]:
    """
    Chunk the file into byte boundaries
    Each chunk ends with the split_special_token to ensure complete document inclusion
    Each chunk may contain zero or more split_special_tokens (one or more documents)

    May return fewer chunks if boundaries end up overlapping
    """
    
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // num_chunks
    chunk_boundaries = [i * chunk_size for i in range(num_chunks+1)]
    chunk_boundaries[-1] = file_size  # to ensure that final chunk is not beyond file size

    mini_chunk_size = 4096 # bytes to read in memory to determine boundaries
    max_size_special_token = max(split_special_tokens, key= lambda x: len(x))
    overlap_size = len(max_size_special_token)-1 # To ensure we can read special tokens at chunk boundaries

    delim_pattern = b"|".join(map(re.escape, split_special_tokens))
    delim_regex = re.compile(delim_pattern)

    # The starting and end boundaries of file remain fixed, we move others as per split token
    for idx in range(1, len(chunk_boundaries) - 1):        
        initial_boundary = chunk_boundaries[idx]
        file.seek(initial_boundary)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if not mini_chunk:
                chunk_boundaries[idx] = file_size
                break

            match = delim_regex.search(mini_chunk)
            if match:
                chunk_boundaries[idx] = initial_boundary + match.end() # keeps special token inside chunks
                break

            initial_boundary += mini_chunk_size - overlap_size
            file.seek(initial_boundary) # seek required as file handler is ahead by overlap size before this statement
    
    return sorted(list(set(chunk_boundaries)))

# ==========================================
# MICRO-CHUNKING 
# ==========================================
def stream_document_byte_range(
    filepath: str,
    start_byte: int,
    end_byte: int,
    split_special_tokens: list[bytes]
) -> Iterator[str]:
    """
    Memory-efficient generator that only reads within assigned byte boundaries
    """
    buffer = b""
    chunk_size = 65536 * 16 # 64 * 1024 * 16- 1MB
    bytes_to_read = end_byte - start_byte
    
    delim_pattern = b"|".join(map(re.escape, split_special_tokens))
    delim_regex = re.compile(delim_pattern)

    max_size_special_token = max(len(token) for token in split_special_tokens)
    
    with open(filepath, 'rb') as file:
        file.seek(start_byte)

        while bytes_to_read > 0:
            current_chunk_size = min(chunk_size, bytes_to_read)

            raw_bytes = file.read(current_chunk_size)
            bytes_to_read -= len(raw_bytes)

            if not raw_bytes:
                break

            
            search_start_idx = max(0, len(buffer) - max_size_special_token + 1)
            buffer += raw_bytes

            while True:
                match = delim_regex.search(buffer, search_start_idx)
                if not match:
                    break

                chunk = buffer[:match.start()]
                buffer = buffer[match.end():]
                search_start_idx = 0
                if chunk:
                    yield chunk.decode('utf-8', errors='ignore')
        
        if buffer:
            yield buffer.decode('utf-8', errors='ignore')

# ==========================================
# Pre-tokenization of File Chunks
# ==========================================
def pretokenize_file_byte_range(
    file_path: str,
    start_byte: int,
    end_byte: int,
    special_tokens: list[bytes]
):
    file_chunk_counts =  Counter()  

    for doc in stream_document_byte_range(file_path, start_byte, end_byte, special_tokens):
        for pattern_match in re.finditer(PAT, doc):
            byte_int_tuple = tuple(pattern_match.group(0).encode('utf-8'))
            file_chunk_counts[byte_int_tuple] += 1

    return file_chunk_counts
    

def pretokenize_file(
    file_path: str, 
    doc_split_tokens, 
    num_chunks
) -> Counter[tuple[int]]:
    special_tokens_bytes = [token.encode('utf-8') for token in doc_split_tokens]
    with open(file_path, 'rb') as  f:
        chunk_boundaries = find_chunk_boundaries(f, num_chunks, special_tokens_bytes)

        chunks = zip(chunk_boundaries[:-1], chunk_boundaries[1:])
        tasks = [(file_path, start, end, special_tokens_bytes) for start, end in chunks]
        num_tasks = len(tasks)

        pretokenized_cache = Counter()
        with multiprocessing.Pool(processes=num_tasks) as pool:
            results = pool.starmap(pretokenize_file_byte_range, tasks)

            for chunk_counts in results:
                pretokenized_cache.update(chunk_counts)

    return pretokenized_cache
