import typing as tp
from collections import defaultdict, deque
from types import MethodType

import torch
import torch.nn as nn


def _normalize_edges(
    dag_edges: tp.Iterable[tp.Sequence[str]],
    columns: tp.Sequence[str],
) -> tp.List[tp.Tuple[int, int]]:
    """Normalize DAG edges into (parent_idx, child_idx) tuples."""
    col_to_idx = {c: i for i, c in enumerate(columns)}
    normalized: tp.List[tp.Tuple[int, int]] = []
    for edge in dag_edges:
        if len(edge) != 2:
            raise ValueError(
                "Each DAG edge must have exactly 2 elements: (parent, child)."
            )
        parent, child = edge[0], edge[1]
        if parent not in col_to_idx:
            raise ValueError(f"DAG parent column {parent!r} is not in dataset columns.")
        if child not in col_to_idx:
            raise ValueError(f"DAG child column {child!r} is not in dataset columns.")
        p_idx = col_to_idx[parent]
        c_idx = col_to_idx[child]
        if p_idx == c_idx:
            raise ValueError(f"Self-loop edge {parent!r}->{child!r} is not allowed.")
        normalized.append((p_idx, c_idx))
    # Keep insertion order but drop exact duplicates
    deduped = list(dict.fromkeys(normalized))
    return deduped


def _assert_acyclic(num_nodes: int, edges: tp.List[tp.Tuple[int, int]]) -> None:
    """Raise if the graph contains a cycle."""
    children = defaultdict(list)
    indegree = [0] * num_nodes
    for parent, child in edges:
        children[parent].append(child)
        indegree[child] += 1

    q = deque([n for n in range(num_nodes) if indegree[n] == 0])
    visited = 0
    while q:
        node = q.popleft()
        visited += 1
        for nxt in children[node]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                q.append(nxt)

    if visited != num_nodes:
        raise ValueError("Provided edges do not form a DAG (cycle detected).")


def _build_relation_code_matrix(
    num_nodes: int,
    edges: tp.List[tp.Tuple[int, int]],
) -> torch.Tensor:
    """Build relation code matrix indexed as [query_col, key_col].

    Codes:
      1: key is direct parent of query
      2: key is ancestor (but not direct parent) of query
      3: no relation
    """
    children = [[] for _ in range(num_nodes)]
    parent_mask = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
    for parent, child in edges:
        children[parent].append(child)
        parent_mask[child, parent] = True

    # reach[src, dst] = src is an ancestor of dst (including direct parent)
    reach = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
    for src in range(num_nodes):
        stack = [src]
        seen = set()
        while stack:
            node = stack.pop()
            for nxt in children[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        for dst in seen:
            reach[src, dst] = True

    ancestor_full = reach.transpose(0, 1)  # [query=descendant, key=ancestor]
    ancestor_only = ancestor_full & ~parent_mask

    relation_code = torch.full((num_nodes, num_nodes), 3, dtype=torch.long)
    relation_code[parent_mask] = 1
    relation_code[ancestor_only] = 2
    return relation_code


class DAGAttentionBias(nn.Module):
    """Learnable DAG-based attention bias.

    Adds:
      alpha when key is parent of query
      beta when key is ancestor (excluding direct parent)
      -gamma when there is no relation
    """

    def __init__(
        self,
        columns: tp.Sequence[str],
        dag_edges: tp.Iterable[tp.Sequence[str]],
        alpha_init: float = 0.0,
        beta_init: float = 0.0,
        gamma_init: float = 0.0,
        learnable: bool = True,
    ):
        super().__init__()
        self.columns = list(columns)
        self.learnable = bool(learnable)
        normalized_edges = _normalize_edges(dag_edges, self.columns)
        _assert_acyclic(len(self.columns), normalized_edges)
        relation_code = _build_relation_code_matrix(len(self.columns), normalized_edges)
        self.register_buffer("relation_code", relation_code, persistent=True)

        self.alpha = nn.Parameter(
            torch.tensor(float(alpha_init)),
            requires_grad=self.learnable,
        )
        self.beta = nn.Parameter(
            torch.tensor(float(beta_init)),
            requires_grad=self.learnable,
        )
        self.gamma = nn.Parameter(
            torch.tensor(float(gamma_init)),
            requires_grad=self.learnable,
        )

        # Runtime batch context set by trainer for each forward pass.
        self._column_ids: tp.Optional[torch.Tensor] = None

    def set_batch(self, column_ids: tp.Optional[torch.Tensor]) -> None:
        self._column_ids = column_ids

    def clear_batch(self) -> None:
        self._column_ids = None

    def build_attention_bias(
        self,
        hidden_states: torch.Tensor,
        layer_past: tp.Optional[tp.Any] = None,
    ) -> tp.Optional[torch.Tensor]:
        """Build additive bias tensor for attention logits.

        Returns shape [batch, 1, query_len, key_len], or None if no batch context.
        """
        if self._column_ids is None:
            return None
        if self._column_ids.ndim != 2:
            return None

        column_ids = self._column_ids
        device = hidden_states.device
        dtype = hidden_states.dtype
        if column_ids.device != device:
            column_ids = column_ids.to(device)

        query_len = hidden_states.size(-2)
        past_len = 0
        if layer_past is not None and isinstance(layer_past, (tuple, list)) and len(layer_past) > 0:
            # GPT-2 cache key tensor shape is usually [batch, heads, past_len, head_dim].
            key_cache = layer_past[0]
            if hasattr(key_cache, "size"):
                past_len = int(key_cache.size(-2))

        key_len = past_len + query_len
        if column_ids.size(1) < key_len:
            # No valid mapping for this sequence length (e.g., generation context).
            return None

        key_col_ids = column_ids[:, :key_len]
        if past_len == 0:
            query_col_ids = column_ids[:, -query_len:]
        else:
            query_col_ids = column_ids[:, past_len:key_len]

        return self._pairwise_bias(
            query_col_ids=query_col_ids,
            key_col_ids=key_col_ids,
            device=device,
            dtype=dtype,
        )

    def _pairwise_bias(
        self,
        query_col_ids: torch.Tensor,
        key_col_ids: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        num_cols = self.relation_code.size(0)
        pad_index = num_cols

        # Extend lookup with a final "invalid/pad" row+col that maps to code=0 (zero bias).
        extended_lookup = torch.zeros(
            (num_cols + 1, num_cols + 1), dtype=torch.long, device=device
        )
        extended_lookup[:num_cols, :num_cols] = self.relation_code.to(device)

        safe_query = torch.where(
            query_col_ids >= 0,
            query_col_ids,
            torch.full_like(query_col_ids, pad_index),
        )
        safe_key = torch.where(
            key_col_ids >= 0,
            key_col_ids,
            torch.full_like(key_col_ids, pad_index),
        )

        relation_codes = extended_lookup[
            safe_query.unsqueeze(-1), safe_key.unsqueeze(-2)
        ]  # [batch, q_len, k_len]

        alpha = self.alpha.to(device=device, dtype=dtype)
        beta = self.beta.to(device=device, dtype=dtype)
        gamma = self.gamma.to(device=device, dtype=dtype)

        bias = torch.zeros_like(relation_codes, dtype=dtype, device=device)
        bias = torch.where(relation_codes == 1, alpha, bias)
        bias = torch.where(relation_codes == 2, beta, bias)
        bias = torch.where(relation_codes == 3, -gamma, bias)
        return bias.unsqueeze(1)


def attach_dag_bias_to_model(model: nn.Module, dag_bias: DAGAttentionBias) -> int:
    """Attach DAG bias module and patch GPT-2 attention blocks."""
    try:
        from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
    except Exception as exc:
        raise ValueError(
            "DAG attention bias currently requires GPT-2 attention modules to be available."
        ) from exc

    model.dag_attention_bias = dag_bias

    patched = 0
    for module in model.modules():
        if isinstance(module, GPT2Attention):
            _patch_gpt2_attention_module(module, dag_bias)
            patched += 1

    if patched == 0:
        raise ValueError(
            "No GPT2Attention modules were found. DAG bias support currently targets GPT-2 style models."
        )
    return patched


def _patch_gpt2_attention_module(
    attn_module: nn.Module,
    dag_bias_controller: DAGAttentionBias,
) -> None:
    """Monkey-patch one GPT-2 attention module to add DAG bias into attention_mask."""
    # If an older patch registered the controller as a child module, remove it.
    if "_great_dag_bias_controller" in getattr(attn_module, "_modules", {}):
        del attn_module._modules["_great_dag_bias_controller"]
    if hasattr(attn_module, "_great_dag_bias_controller"):
        try:
            delattr(attn_module, "_great_dag_bias_controller")
        except Exception:
            pass

    # Repatching is allowed so closure can point to the latest controller.
    if not getattr(attn_module, "_great_dag_bias_patched", False):
        original_forward = attn_module.forward
        attn_module._great_dag_forward_original = original_forward

    def _forward_with_dag_bias(self, hidden_states, *args, **kwargs):
        controller = dag_bias_controller
        encoder_hidden_states = kwargs.get("encoder_hidden_states", None)
        if controller is not None and encoder_hidden_states is None:
            dag_bias = controller.build_attention_bias(
                hidden_states=hidden_states,
                layer_past=kwargs.get("layer_past", None),
            )
            if dag_bias is not None:
                attn_mask = kwargs.get("attention_mask", None)
                if attn_mask is None:
                    kwargs["attention_mask"] = dag_bias
                else:
                    kwargs["attention_mask"] = attn_mask + dag_bias.to(attn_mask.dtype)
        return self._great_dag_forward_original(hidden_states, *args, **kwargs)

    attn_module.forward = MethodType(_forward_with_dag_bias, attn_module)
    attn_module._great_dag_bias_patched = True
