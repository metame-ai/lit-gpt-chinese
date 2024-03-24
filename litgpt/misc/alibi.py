import torch
import math


def update_attention_mask(inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, 
                          alibi_mask: torch.Tensor) -> torch.Tensor:
    if len(attention_mask.shape) == 2:
            expanded_mask = attention_mask.to(alibi_mask.dtype)
            expanded_mask = torch.tril(
                torch.gt(expanded_mask[:, :, None] * expanded_mask[:, None, :], 0)
            ) * torch.eq(expanded_mask[:, :, None] - expanded_mask[:, None, :], 0)
    else:
        expanded_mask = attention_mask

    if len(expanded_mask.shape) == 3:
        expanded_mask = expanded_mask.unsqueeze(1)
    bsz = inputs_embeds.size(0)
    src_len, tgt_len = alibi_mask.size()[-2:]
    expanded_mask = (
        expanded_mask.expand(bsz, 1, src_len, tgt_len)
        .to(alibi_mask.dtype)
    )
    inverted_mask = 1.0 - expanded_mask
    inverted_mask = inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(alibi_mask.dtype).min
    )
    attention_mask = inverted_mask + alibi_mask.unsqueeze(0)
    return attention_mask

    
def do_attention(head_size: int, seq_len: int,
        query_states: torch.Tensor, key_states: torch.Tensor, value_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(head_size)

        if attention_mask is not None:
            if seq_len == 1:  # inference with cache
                if len(attention_mask.size()) == 4:
                    attention_mask = attention_mask[:, :, -1:, :]
                else:
                    attention_mask = attention_mask[:, -1:, :]
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2)
        return attn_output

        
def _get_interleave(n):
    def _get_interleave_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return _get_interleave_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return (
            _get_interleave_power_of_2(closest_power_of_2)
            + _get_interleave(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )


def _fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


def _buffered_future_mask(tensor, maxpos, alibi, attn_heads):
    _future_mask = torch.triu(_fill_with_neg_inf(torch.zeros([maxpos, maxpos])), 1)
    _future_mask = _future_mask.unsqueeze(0) + alibi
    new_future_mask = _future_mask.to(tensor)
    return new_future_mask[: tensor.shape[0] * attn_heads, :maxpos, :maxpos]


def build_alibi_mask(n_head, max_pos):
    slopes = torch.Tensor(_get_interleave(n_head))
    position_point = torch.arange(max_pos) - max_pos + 1
    position_point = position_point.unsqueeze(0).unsqueeze(0).expand(n_head, -1, -1)
    diag = torch.diag(position_point[0])
    position_point = position_point - diag.unsqueeze(0).unsqueeze(0).transpose(-1, -2)
    slopes = slopes.to(position_point.device)
    alibi = slopes.unsqueeze(1).unsqueeze(1) * position_point
    alibi = alibi.view(n_head, 1, max_pos)
    alibi_mask = torch.triu(_fill_with_neg_inf(torch.zeros([max_pos, max_pos])), 1)
    alibi_mask = alibi_mask.unsqueeze(0) + alibi
    return alibi_mask