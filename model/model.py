from transformers import PretrainedConfig

#huggingface_类
class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )


import torch
import math
import torch.nn as nn
from torch.nn import init
from typing import Optional, Tuple, List, Union
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

#
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))#self.weight：可训练的参数（形状为dim），初始值全 1，用于对归一化后的结果做缩放（RMSNorm 无偏置项，这是和 LayerNorm 的核心区别之一）。

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)#RMSNorm的公式
#前向传播，将rmsnorm应用于输入x，并乘以权重参数self.weight，最后返回结果。
    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs(
    dim: int,
    end: int = int(32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: Optional[dict] = None,
):
    # 1. ³õÊ¼»¯±ê×¼ RoPE ÆµÂÊ¡£
    # torch.arange(0, dim, 2) Éú³É [0, 2, 4, ... dim-2]
    # ¼ÆËã³öµÄ freqs ¾ÍÊÇ±ê×¼µÄ 1 / (base ** (2i / d))
    freqs, attn_factor = (
        1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)),
        1.0,
    )

    if rope_scaling is not None:
        # 2. ´ÓÅäÖÃ×ÖµäÖÐÌáÈ¡ YaRN µÄ³¬²ÎÊý
        # orig_max: Ä£ÐÍÔ¤ÑµÁ·Ê±µÄÔ­Ê¼×î´ó³¤¶È£¨ÀýÈç Llama-2 ÊÇ 2048 »ò 4096£©
        # factor: ÒªÀ©Õ¹µÄ±¶Êý s (±ÈÈç´Ó 2k À©Õ¹µ½ 32k£¬factor ¾ÍÊÇ 16)
        # beta_fast (¶ÔÓ¦ÂÛÎÄÖÐµÄ ¦Á): ¸ßÆµ±ß½ç£¬²¨³¤±ÈÀý´óÓÚ´ËÖµµÄÎ¬¶È²»Ëõ·Å
        # beta_slow (¶ÔÓ¦ÂÛÎÄÖÐµÄ ¦Â): µÍÆµ±ß½ç£¬²¨³¤±ÈÀýÐ¡ÓÚ´ËÖµµÄÎ¬¶ÈÈ«Á¿Ëõ·Å
        # attn_factor: ×¢ÒâÁ¦ÎÂ¶È²¹³¥£¬ÓÉÓÚ¾àÀëÀ­³¤µ¼ÖÂ×¢ÒâÁ¦·Ö²¼·¢É¢£¨±äÆ½»º£©£¬ÐèÒª³ËÉÏÒ»¸öÏµÊýÈÃ×¢ÒâÁ¦ÖØÐÂ¡°¾Û½¹¡±
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0),
        )

        # Ö»ÓÐµ±ÒªÍÆ¶ÏµÄ³¤¶È´óÓÚÔ­Ê¼ÑµÁ·³¤¶ÈÊ±£¬²ÅÓ¦ÓÃËõ·Å
        if end / orig_max > 1.0:
            # 3. Ê¹ÓÃÇ°ÎÄÍÆµ¼µÄ¹«Ê½£¬¶¨Òå²¨³¤±ÈÀý b µ½Î¬¶ÈË÷Òý i µÄÓ³Éäº¯Êý
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (
                2 * math.log(rope_base)
            )

            # 4. ¼ÆËã¸ßÆµÇøºÍµÍÆµÇøµÄÎ¬¶ÈÇÐ·Öµã
            # low: ²»ÐèÒªËõ·ÅµÄ¸ßÆµ²¿·ÖµÄ×î¸ßË÷Òý
            # high: ÐèÒªÍêÈ«Ëõ·ÅµÄµÍÆµ²¿·ÖµÄ×îµÍË÷Òý
            low, high = (
                max(math.floor(inv_dim(beta_fast)), 0),
                min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1),
            )

            # 5. ¼ÆËã»ìºÏÒò×Ó ¦Ã (Ramp)
            # ÔÚ low Ö®Ç°£¬ramp Îª 0£»ÔÚ high Ö®ºó£¬ramp Îª 1£»ÔÚ low ºÍ high Ö®¼ä£¬ÏßÐÔ¹ý¶É¡£
            # clamp º¯ÊýÏÞÖÆÁËÊýÖµÖ»ÄÜÔÚ [0, 1] Ö®¼ä¡£
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low)
                / max(high - low, 0.001),
                0,
                1,
            )

            # 6. ÆµÂÊÈÚºÏ¹«Ê½£ºf'(i) = f(i) * ((1-¦Ã) + ¦Ã/s)
            # µ± ramp=0 Ê±£¨¸ßÆµ£©£ºÏµÊýÎª 1£¬±£³ÖÔ­ÆµÂÊ²»±ä¡£
            # µ± ramp=1 Ê±£¨µÍÆµ£©£ºÏµÊýÎª 1/factor£¬¼´¶ÔÆµÂÊ½øÐÐÏßÐÔ²åÖµËõ·Å¡£
            # rampÔÚ0-1Ö®¼äÊ±£ºÆ½»¬¹ý¶É¡£
            freqs = freqs * (1 - ramp + ramp / factor)

    # 7. ¸ù¾ÝÄ¿±ê³¤¶È end£¬Éú³ÉÎ»ÖÃË÷ÒýÏòÁ¿ t
    t = torch.arange(end, device=freqs.device)

    # 8. ¼ÆËãÍâ»ý£º½«Î»ÖÃ t Óë´¦ÀíºÃµÄÆµÂÊ freqs Ïà³Ë£¬µÃµ½Ã¿¸öÎ»ÖÃµÄÐý×ª½Ç¶È ¦È
    freqs = torch.outer(t, freqs).float()

    # 9. ¼ÆËã Cos ºÍ Sin£¬²¢Ó¦ÓÃ×¢ÒâÁ¦²¹³¥ÏµÊý (attn_factor)
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor

    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat(
            (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
        )

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    )
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
    )
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x

    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: MokioMindConfig):
        super().__init__()

        self.num_key_value_heads = (
            args.num_attention_heads
            if args.num_key_value_heads is None
            else args.num_key_value_heads
        )

        assert args.num_attention_heads % self.num_key_value_heads == 0

        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads

        self.q_proj = nn.Linear(
            args.hidden_size, args.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            args.num_attention_heads * self.head_dim, args.hidden_size, bias=False
        )

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
            and args.flash_attention
        )

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # kv_cacheÊµÏÖ
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2),
        )

        if (
            self.flash
            and (seq_len > 1)
            and (past_key_value is None)
            and (attention_mask is None or torch.all(attention_mask == 1))
        ):
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores[:, :, :, -seq_len:] += torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1,
            )

            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, config: MokioMindConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)

        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        gated = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(gated))


class MoEGate(nn.Module):
    def __init__(self, config: MokioMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)

        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(
                    bsz, self.n_routed_experts, device=hidden_states.device
                )
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device),
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                    dim=1
                ).mean() * self.alpha
            else:
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = scores.new_zeros(1).squeeze()
        return topk_idx, topk_weight, aux_loss


class MoEFeedForward(nn.Module):  # £¡ÐÞÕý£ºÔ­MoEFeedForawardÆ´Ð´´íÎó
    def __init__(self, config: MokioMindConfig):
        super().__init__()
        self.config = config
        # ×¨¼Ò²ã
        self.experts = nn.ModuleList(
            [FeedForward(config) for _ in range(config.n_routed_experts)]
        )
        # ÃÅ¿Ø²ã
        self.gate = MoEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList(
                [FeedForward(config) for _ in range(config.n_shared_experts)]
            )

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, h = orig_shape

        # Ê¹ÓÃÃÅ¿Ø»úÖÆÑ¡Ôñ×¨¼Ò
        topk_idx, topk_weight, aux_loss = self.gate(x)
        # Õ¹¿ªxÒÔ±ã´¦Àí
        x = x.view(-1, x.shape[-1])

        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            # °´ÕÕ¶¨ÒåµÄnum_experts_per_tokÖØ¸´ÊäÈëtoken
            # Ã¿¸ötoken°²ÅÅnum_experts_per_tok¸ö×¨¼Ò´¦Àí
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            # yÊÇ¿ÕÕÅÁ¿£¬ºÍxÐÎ×´ÏàÍ¬
            y = torch.empty_like(x, dtype=x.dtype)
            # ±éÀúËùÓÐ×¨¼Ò
            for i, expert in enumerate(self.experts):
                # ÕÒµ½ËùÓÐÖ¸Ïò×¨¼ÒiµÄtoken
                # È»ºó½«ÕâÐ©tokenÊäÈë×¨¼Òi½øÐÐ´¦Àí
                # ×îºó½«½á¹û·Å»Øy¶ÔÓ¦Î»ÖÃ
                expert_out = expert(x[flat_topk_idx == i])
                if expert_out.shape[0] > 0:
                    y[flat_topk_idx == i] = expert_out.to(y.dtype)
                else:
                    y[flat_topk_idx == i] = expert_out.to(y.dtype) + 0 * sum(
                        p.sum() for p in expert.parameters()
                    )
            # ¼ÓÈ¨ÇóºÍ
            # ×îºóµÄyÒâÒåÊÇÃ¿¸ötoken¾­¹ý×¨¼Ò´¦ÀíºóµÄ¼ÓÈ¨½á¹û
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        # Èç¹ûÊÇÍÆÀí½×¶Î
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(
                *orig_shape
            )
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    # MoEÍÆÀí·½·¨
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        # Ê¹ÓÃcache£¬´´½¨Ò»¸öºÍxÐÎ×´ÏàÍ¬µÄÁãÕÅÁ¿
        expert_cache = torch.zeros_like(x)
        # ¶Ô×¨¼ÒË÷Òý½øÐÐÅÅÐò£¬×îºóÊÇ[0,0,0,1,1,2,2,2,...]ÕâÑùµÄË³Ðò
        # ·Ö¼ð
        idxs = flat_expert_indices.argsort()
        # Í³¼ÆÃ¿¸ö×¨¼Ò±»·ÖÅäµ½µÄtokenÊýÁ¿
        # ´ò°ü
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        # ¼ÆËãÃ¿¸ötoken¶ÔÓ¦µÄ×¨¼ÒË÷Òý
        token_idxs = idxs // self.config.num_experts_per_tok
        # ¶ÔÃ¿¸ö´ò°üºÃµÄ°ü½øÐÐ´¦Àí
        for i, end_idx in enumerate(tokens_per_expert):
            # ¼ÆËãµ±Ç°°üµÄÆðÊ¼Î»ÖÃ
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            # È¡³öµ±Ç°°ü¶ÔÓ¦µÄ×¨¼Ò
            expert = self.experts[i]
            # È¡³ötoken¶ÔÓ¦µÄÔ­Ê¼id
            exp_token_idx = token_idxs[start_idx:end_idx]
            # È¡³ötoken¶ÔÓ¦µÄÊý¾Ý
            expert_tokens = x[exp_token_idx]
            # ¼ÆËã×¨¼ÒÊä³ö£¬Ò»´ÎÐÔ´¦Àíµ±Ç°°üµÄËùÓÐtoken
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            # ¼ÓÈ¨
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # ½«½á¹ûÉ¢µã¼Óµ½»º´æÖÐ¶ÔÓ¦Î»ÖÃ
            expert_cache.scatter_add_(
                0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out
            )

        return expert_cache


class MokioMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MokioMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attention = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = (
            FeedForward(config)
            if not config.use_moe
            else MoEFeedForward(config)  # £¡ÐÞÕý£ºÔ­MoEFeedForawardÆ´Ð´´íÎó
        )

    def forward(
        self,
        hidden_states,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        res = hidden_states

        hidden_states, present_key_value = self.self_attention(
            self.input_layernorm(hidden_states),  # pre-norm
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )

        hidden_states = res + hidden_states

        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states, present_key_value


class MokioMindModel(nn.Module):
    def __init__(self, config: MokioMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = (
            config.vocab_size,
            config.num_hidden_layers,
        )
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [MokioMindBlock(l, config) for l in range(self.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        # input_ids: [bsz, seq_len]
        batch_size, seq_length = input_ids.shape

        if hasattr(past_key_values, "layers"):
            past_key_values = None

        past_key_values = past_key_values or [None] * len(self.layers)

        # ¼ÆËãstart_pos£ºÈç¹û´æÔÚpast£¬Ôòstart_posÎªÒÑÓÐpastÐòÁÐ³¤¶È
        start_pos = (
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )

        # Embedding + dropout
        hidden_states = self.dropout(
            self.embed_tokens(input_ids)
        )  # [bsz, seq_len, hidden]

        position_embeddings = (
            self.freqs_cos[start_pos : start_pos + seq_length],
            self.freqs_sin[start_pos : start_pos + seq_length],
        )
        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(
            zip(self.layers, past_key_values)
        ):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)

        aux_loss = sum(
            [
                layer.mlp.aux_loss
                for layer in self.layers
                if isinstance(
                    layer.mlp, MoEFeedForward
                )  # £¡ÐÞÕý£ºÔ­MoEFeedForawardÆ´Ð´´íÎó
            ],
            hidden_states.new_zeros(1).squeeze(),
        )

        return hidden_states, presents, aux_loss


class MokioMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MokioMindConfig

    def __init__(self, config: MokioMindConfig):
        super().__init__(config)
        self.model = MokioMindModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **args,
    ):
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args,
        )

        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )
        output.aux_loss = aux_loss
        return output