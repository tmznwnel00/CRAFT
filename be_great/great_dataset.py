import random
import typing as tp
import numpy as np
import torch

from datasets import Dataset
from dataclasses import dataclass
from transformers import DataCollatorWithPadding


class GReaTDataset(Dataset):
    """GReaT Dataset

    The GReaTDataset overwrites the _getitem function of the HuggingFace Dataset Class to include the permutation step.

    Attributes:
        tokenizer (AutoTokenizer): Tokenizer from HuggingFace
        float_precision (int, optional): Number of decimal places to use for floating point numbers.
                                        If None, full precision is used.
    """

    def set_tokenizer(self, tokenizer, float_precision=None):
        """Set the Tokenizer

        Args:
            tokenizer: Tokenizer from HuggingFace
            float_precision: Number of decimal places to use for floating point numbers.
                           If None, full precision is used.
        """
        self.tokenizer = tokenizer
        self.float_precision = float_precision

    def _format_value(self, value):
        """Format a value based on its type.
        
        For floats, applies precision formatting if float_precision is set.
        
        Args:
            value: The value to format
            
        Returns:
            Formatted string value
        """
        if isinstance(value, (float, np.floating)) and self.float_precision is not None:
            # Format to a string with specified decimal places, removing trailing zeros
            formatted_value_str = f"{value:.{self.float_precision}f}"
            if '.' in formatted_value_str:
                formatted_value_str = formatted_value_str.rstrip('0').rstrip('.')
            return formatted_value_str
        return str(value).strip()

    def _getitem(
        self, key: tp.Union[int, slice, str], decoded: bool = True, **kwargs
    ) -> tp.Union[tp.Dict, tp.List]:
        """Get Item from Tabular Data

        Get one instance of the tabular data, permuted, converted to text and tokenized.
        """
        # If int, what else?
        row = self._data.fast_slice(key, 1)

        # shuffle_idx = list(range(row.num_columns))
        # random.shuffle(shuffle_idx)
        shuffle_idx = [1, 6, 4, 7, 8, 9, 0, 3, 5, 2, 10] # adult
        # shuffle_idx = [0, 1, 2, 3, 4, 5, 6, 7] # asia
        # shuffle_idx = [0, 1, 3, 2, 4, 5, 6] # healthcare

        segments = [
            "%s is %s"
            % (row.column_names[i], self._format_value(row.columns[i].to_pylist()[0]))
            for i in shuffle_idx
        ]
        shuffled_text = ", ".join(segments)

        tokenized_text = self._tokenize_with_column_ids(
            shuffled_text=shuffled_text,
            shuffle_idx=shuffle_idx,
            segments=segments,
        )
        return tokenized_text

    def _tokenize_with_column_ids(
        self,
        shuffled_text: str,
        shuffle_idx: tp.List[int],
        segments: tp.List[str],
    ) -> tp.Dict[str, tp.Any]:
        """Tokenize one serialized row and align tokens to source column ids."""
        try:
            tokenized = self.tokenizer(
                shuffled_text,
                padding=False,
                truncation=False,
                return_offsets_mapping=True,
            )
            offsets = tokenized.pop("offset_mapping", None)
            if offsets is None:
                raise ValueError("Tokenizer did not return offset_mapping.")

            col_char_map = self._build_char_to_col_map(shuffled_text, shuffle_idx, segments)
            column_ids = self._offsets_to_column_ids(offsets, col_char_map)
            tokenized["column_ids"] = column_ids
            return tokenized
        except Exception:
            # Slow-tokenizer or offset-mapping fallback:
            # tokenize by segments and assign the owning column id to all piece tokens.
            return self._segment_fallback_tokenize(shuffle_idx, segments)

    @staticmethod
    def _build_char_to_col_map(
        text: str,
        shuffle_idx: tp.List[int],
        segments: tp.List[str],
    ) -> tp.List[int]:
        """Map each character position in the serialized row to its column index."""
        char_to_col = [-1] * len(text)
        cursor = 0
        for pos, (col_idx, segment) in enumerate(zip(shuffle_idx, segments)):
            start = cursor
            end = start + len(segment)
            for i in range(start, min(end, len(char_to_col))):
                char_to_col[i] = col_idx
            cursor = end
            if pos < len(segments) - 1:
                cursor += 2  # ", "
        return char_to_col

    @staticmethod
    def _offsets_to_column_ids(
        offsets: tp.Sequence[tp.Tuple[int, int]],
        char_to_col: tp.List[int],
    ) -> tp.List[int]:
        """Convert tokenizer offsets to a per-token column id list."""
        column_ids: tp.List[int] = []
        for start, end in offsets:
            if end <= start:
                column_ids.append(-1)
                continue

            col_id = -1
            for i in range(start, min(end, len(char_to_col))):
                if char_to_col[i] != -1:
                    col_id = char_to_col[i]
                    break

            # Punctuation/space tokens: attach to nearest token on the left if possible.
            if col_id == -1 and start > 0:
                j = min(start - 1, len(char_to_col) - 1)
                while j >= 0 and char_to_col[j] == -1:
                    j -= 1
                if j >= 0:
                    col_id = char_to_col[j]

            column_ids.append(col_id)
        return column_ids

    def _segment_fallback_tokenize(
        self,
        shuffle_idx: tp.List[int],
        segments: tp.List[str],
    ) -> tp.Dict[str, tp.Any]:
        """Fallback tokenization path when offsets are unavailable."""
        input_ids: tp.List[int] = []
        attention_mask: tp.List[int] = []
        column_ids: tp.List[int] = []

        for pos, (col_idx, segment) in enumerate(zip(shuffle_idx, segments)):
            piece = segment if pos == 0 else f", {segment}"
            piece_ids = self.tokenizer(piece, add_special_tokens=False)["input_ids"]
            input_ids.extend(piece_ids)
            attention_mask.extend([1] * len(piece_ids))
            column_ids.extend([col_idx] * len(piece_ids))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "column_ids": column_ids,
        }

    def __getitems__(self, keys: tp.Union[int, slice, str, list]):
        if isinstance(keys, list):
            return [self._getitem(key) for key in keys]
        else:
            return self._getitem(keys)


@dataclass
class GReaTDataCollator(DataCollatorWithPadding):
    """GReaT Data Collator

    Overwrites the DataCollatorWithPadding to also pad the labels and not only the input_ids
    """

    def __call__(self, features: tp.List[tp.Dict[str, tp.Any]]):
        column_ids = [f.pop("column_ids", None) for f in features]
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        if any(c is not None for c in column_ids):
            max_len = int(batch["input_ids"].shape[1])
            pad_side = getattr(self.tokenizer, "padding_side", "right")
            padded_column_ids = []
            for cids in column_ids:
                ids = cids if cids is not None else []
                pad_len = max_len - len(ids)
                if pad_len < 0:
                    ids = ids[-max_len:]
                    pad_len = 0
                if pad_side == "left":
                    padded = ([-1] * pad_len) + ids
                else:
                    padded = ids + ([-1] * pad_len)
                padded_column_ids.append(padded)
            batch["column_ids"] = torch.tensor(padded_column_ids, dtype=torch.long)

        batch["labels"] = batch["input_ids"].clone()
        return batch
