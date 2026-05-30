import os
import multiprocessing
import regex as re
from typing import BinaryIO, Iterator
from collections import Counter


NUM_CHUNKS = 4
DOC_SPLIT_TOKEN="<|endoftext|>"
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# ==========================================
# MACRO-CHUNKING 
# ==========================================
def find_chunk_boundaries(
    file: BinaryIO,
    num_chunks: int,
    split_special_token: bytes
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
    overlap_size = len(split_special_token)-1 # To ensure we can read special tokens at chunk boundaries

    # The starting and end boundaries of file remain fixed, we move others as per split token
    for idx in range(1, len(chunk_boundaries) - 1):
        initial_boundary = chunk_boundaries[idx]
        file.seek(initial_boundary)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if not mini_chunk:
                chunk_boundaries[idx] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[idx] = initial_boundary + found_at + len(split_special_token) # keeps special token inside chunks
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
    split_special_token: bytes
) -> Iterator[str]:
    """
    Memory-efficient generator that only reads within assigned byte boundaries
    """
    buffer = b""
    chunk_size = 65536 # 64 * 1024 - 64KB

    bytes_to_read = end_byte - start_byte
    
    with open(filepath, 'rb') as file:
        file.seek(start_byte)

        while bytes_to_read > 0:
            current_chunk_size = min(chunk_size, bytes_to_read)

            raw_bytes = file.read(current_chunk_size)
            bytes_to_read -= len(raw_bytes)

            if not raw_bytes:
                break

            buffer += raw_bytes

            while split_special_token in buffer:
                chunk, buffer = buffer.split(split_special_token, 1)
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
):
    token_bytes = DOC_SPLIT_TOKEN.encode('utf-8')
    file_chunk_counts =  Counter()  

    for doc in stream_document_byte_range(file_path, start_byte, end_byte, token_bytes):
        for pattern_match in re.finditer(PAT, doc):
            byte_int_tuple = tuple(pattern_match.group(0).encode('utf-8'))
            file_chunk_counts[byte_int_tuple] += 1

    return file_chunk_counts
    

def pretokenize_file(
    file_path: str
) -> Counter[tuple[int]]:
    token_bytes = DOC_SPLIT_TOKEN.encode('utf-8')
    with open(file_path, 'rb') as  f:
        chunk_boundaries = find_chunk_boundaries(f, NUM_CHUNKS, token_bytes)

        chunks = zip(chunk_boundaries[:-1], chunk_boundaries[1:])
        tasks = [(file_path, start, end) for start, end in chunks]
        num_tasks = len(tasks)

        pretokenized_cache = Counter()
        with multiprocessing.Pool(processes=num_tasks) as pool:
            results = pool.starmap(pretokenize_file_byte_range, tasks)

            for chunk_counts in results:
                pretokenized_cache.update(chunk_counts)

    return pretokenized_cache