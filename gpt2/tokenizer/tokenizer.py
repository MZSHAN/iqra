import pickle
from collections.abc import Iterable, Iterator

from regex import re

from pretokenization_utils import PAT

class Tokenizer:
    def __init__(
        self, 
        vocab: dict[int, bytes], 
        merges: list[tuple[bytes, bytes]], 
        special_tokens: list[str] | None = None
    ):
        """
        Construct a tokenizer from a given vocabulary, list of merges, 
        and (optionally) a list of special tokens.
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        self.special_tokens_set = set(special_tokens) if special_tokens else set()
        self.bytes_to_int = {j:i for i, j in vocab.items()}
        self.merge_priority = {merge:i for i, merge in enumerate(merges)}
        

    @classmethod
    def from_files(
        cls, 
        vocab_filepath: str, 
        merges_filepath: str, 
        special_tokens: list[str] | None = None
    ) -> "Tokenizer":
        """
        Class method that constructs and returns a Tokenizer from a serialized 
        vocabulary and list of merges (in the same format that your BPE training 
        code output) and (optionally) a list of special tokens.
        """
        with open(vocab_filepath, 'rb') as vocab_file:
            vocab = pickle.load(vocab_file)
        
        with open(merges_filepath, 'rb') as merges_file:
            merges = pickle.load(merges_file)           
            
        return cls(vocab, merges, special_tokens)

    def _pretokenized_input(self, text: str) -> Iterable[str]:
        """
        Pre-tokenize input before encoding. Strategy same as while training the 
        tokenizer
        """
        pattern = PAT
        if self.special_tokens:
            special_pattern = "|".join(map(re.escape, self.special_tokens))
            pattern = f"(?:{special_pattern})|{PAT}"

        for sub_word in re.findall(pattern, text):
            yield sub_word


    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        TODO: This is non optimal - optimize it
        """
        # Gemini conversation - https://gemini.google.com/share/7366e82f856c
        encoded_string = []
        
        for word in self._pretokenized_input(text):
            byte_word = word.encode('utf-8')
            if word in self.special_tokens_set:
                encoded_string.append(self.bytes_to_int[byte_word])
                continue

            byte_arr = [bytes([b]) for b in byte_word]

            while True:
                if len(byte_arr) == 1:
                    break

                byte_pairs = [(byte_arr[i], byte_arr[i+1]) for i in range(len(byte_arr)-1)]
                merge_pair = min(byte_pairs, key=lambda x: self.merge_priority.get(x, float('inf')))

                if merge_pair not in self.merge_priority:
                    break
                
                merged_byte_arr = []
                i = 0
                while i < len(byte_arr):
                    if i < len(byte_arr)-1 and merge_pair == (byte_arr[i], byte_arr[i+1]):
                        merged_byte_arr.append(byte_arr[i] + byte_arr[i+1])
                        i += 2
                    else:
                        merged_byte_arr.append(byte_arr[i])
                        i += 1
                byte_arr = merged_byte_arr
            
            encoded_string.extend(self.bytes_to_int[chunk] for chunk in byte_arr)

        return encoded_string


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a 
        generator that lazily yields token IDs. This is required for 
        memory-efficient tokenization of large files that we cannot directly 
        load into memory.
        """
        for text_chunk in iterable:
            chunk_ids = self.encode(text_chunk)
            yield from chunk_ids

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        bytes_string = b"".join([self.vocab[idx] for idx in ids])
        return bytes_string.decode('utf-8', errors='replace')
