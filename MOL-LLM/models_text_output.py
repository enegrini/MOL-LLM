import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from transformers import GPT2Tokenizer, GPT2Config  # GPT2Model
from models_gpt2_source import GPT2Model
from data_gen.preprocess import (
    batch_tokenize,
    number_tokenize,
    get_word_embeddings,
    get_data_embeddings,
    batch_inputs,
    batch_indices,
)


# from models_gpt2_reproduce_mask import ReproduceGPT2

N_MAX_POSITIONS = 512


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


def create_sinusoidal_embeddings(n_pos, dim, out):
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    out.detach_()
    out.requires_grad = False
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))


class MultiHeadAttention(nn.Module):
    NEW_ID = itertools.count()

    def __init__(self, n_heads, dim, src_dim, dropout):
        super().__init__()
        self.layer_id = next(MultiHeadAttention.NEW_ID)
        self.dim = dim
        self.src_dim = src_dim
        self.n_heads = n_heads
        self.dropout = dropout
        assert self.dim % self.n_heads == 0

        self.attn = nn.MultiheadAttention(self.dim, self.n_heads, dropout=self.dropout, batch_first=True)

    def forward(self, input, mask=None, kv=None):
        """
        Self-attention (if kv is None)
        or attention over source sentence (provided by kv).
            input   (bs, qlen, dim)
            mask    (bs, klen) (non-causal) or (bs, klen, klen)
        """
        bs, qlen, dim = input.size()
        if kv is None:
            klen = qlen
        else:
            klen = kv.size(1)
        assert dim == self.dim, "Dimensions do not match: %s input vs %s configured" % (
            dim,
            self.dim,
        )

        n_heads = self.n_heads

        if kv is None:
            kv = input

        key_padding_mask = None
        attn_mask = None

        if mask is not None:
            if mask.dim() == 3:
                # causal
                mask_reshape = (bs, 1, qlen, klen)
                mask = (mask == 0).view(mask_reshape).expand(bs, n_heads, qlen, klen)
                attn_mask = mask.reshape(bs * n_heads, qlen, klen).bool()
            else:
                # non-causal
                key_padding_mask = mask == 0

        # context (bs, qlen, dim), weights (bs, n_heads, qlen, klen)
        context, weights = self.attn(
            input,  # query
            kv,  # key
            kv,  # value
            key_padding_mask=key_padding_mask,
            average_attn_weights=False,
            attn_mask=attn_mask,
        )

        return context


class TransformerFFN(nn.Module):
    def __init__(self, in_dim, dim_hidden, out_dim, hidden_layers, dropout):
        super().__init__()
        self.dropout = dropout
        self.hidden_layers = hidden_layers
        self.midlin = nn.ModuleList()
        self.lin1 = nn.Linear(in_dim, dim_hidden)
        for i in range(1, self.hidden_layers):
            self.midlin.append(nn.Linear(dim_hidden, dim_hidden))
        self.lin2 = nn.Linear(dim_hidden, out_dim)

    def forward(self, input):
        x = self.lin1(input)
        x = F.gelu(x)
        for mlin in self.midlin:
            x = mlin(x)
            x = F.gelu(x)
        x = self.lin2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class CrossAttnTransformerModel(nn.Module):
    def __init__(
        self,
        config,
        with_output=True,
    ):
        """
        Cross Attention Transformer model for data output.

        Inputs:
            params should contain model configurations and output dimension
        """
        super().__init__()

        self.dtype = torch.float
        self.with_output = with_output

        # model parameters

        self.dim = config["emb_dim"]  # 512 by default
        self.src_dim = config["emb_dim"]
        self.hidden_dim = self.dim * 4  # 2048 by default
        self.n_hidden_layers = config["n_hidden_layers"]
        self.n_heads = config["n_heads"]  # 8 by default
        self.n_layers = config["n_layers"]

        self.dropout = config["dropout"]
        self.attention_dropout = config["attention_dropout"]
        assert self.dim % self.n_heads == 0, "transformer dim must be a multiple of n_heads"

        # embeddings

        self.query_embedder = nn.Linear(config["query_src_dim"], self.dim)

        positional_embeddings = config["positional_embeddings"]
        if positional_embeddings is None or len(positional_embeddings) == 0:
            self.position_embeddings = None
        elif positional_embeddings == "sinusoidal":
            self.position_embeddings = Embedding(N_MAX_POSITIONS, self.dim)
            create_sinusoidal_embeddings(N_MAX_POSITIONS, self.dim, out=self.position_embeddings.weight)
        elif positional_embeddings == "learnable":
            self.position_embeddings = Embedding(N_MAX_POSITIONS, self.dim)
        else:
            raise NotImplementedError

        self.layer_norm_emb = nn.LayerNorm(self.dim, eps=1e-12)

        # transformer layers

        self.attentions = nn.ModuleList()
        self.layer_norm1 = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.layer_norm2 = nn.ModuleList()

        for layer_id in range(self.n_layers):
            self.attentions.append(
                MultiHeadAttention(
                    self.n_heads,
                    self.dim,
                    self.dim,
                    dropout=self.attention_dropout,
                )
            )
            self.layer_norm1.append(nn.LayerNorm(self.dim, eps=1e-12))
            self.ffns.append(
                TransformerFFN(
                    self.dim,
                    self.hidden_dim,
                    self.dim,
                    self.n_hidden_layers,
                    dropout=self.dropout,
                )
            )
            self.layer_norm2.append(nn.LayerNorm(self.dim, eps=1e-12))

        # output layer
        if self.with_output:
            self.proj = nn.Linear(self.dim, config["output_dimension"], bias=True)

    def get_query_emb(self, query_times):
        slen = query_times.size(0)
        query_times = query_times.view(slen, 1)
        query_emb = self.query_embedder(query_times)
        return query_emb  # (slen, dim)

    # def forward(self, mode, **kwargs):
    #     """
    #     Forward function with different forward modes.
    #     ### Small hack to handle PyTorch distributed.
    #     """
    #     if mode == "fwd":
    #         return self.fwd(**kwargs)
    #     elif mode == "query_emb":
    #         return self.get_query_emb(**kwargs)
    #     elif mode == "predict":
    #         return self.predict(**kwargs)
    #     else:
    #         raise Exception("Unknown mode: %s" % mode)

    def forward(
        self,
        query_emb,
        src_enc,
        src_len,
        positions=None,
    ):
        """
        Inputs:
            query_emb   (slen, dim), embedding of times for evaluation
            src_enc     (bs, slen, dim), output from backbone model
            src_len     LongTensor(bs), containing the length of src_enc
            positions   LongTensor(slen, bs), containing word positions
        """

        # check inputs
        bs = src_enc.size(0)
        slen, dim = query_emb.size(0), query_emb.size(1)
        query_emb = query_emb.view(1, slen, dim).expand(bs, slen, dim)

        max_len = src_enc.size(1) #src_len.max() #TODO check this: should be same slen as encoded tensor for mask to work.
        src_mask = torch.arange(max_len, dtype=torch.long, device=query_emb.device) < src_len[:, None]  # (bs, data_len)

        # positions
        if self.position_embeddings is not None:
            if positions is None:
                positions = query_emb.new(slen).long()
                positions = torch.arange(slen, out=positions).unsqueeze(0)
            else:
                assert positions.size() == (slen, bs)
                positions = positions.transpose(0, 1)

        tensor = query_emb  # (bs, slen, dim)

        if self.position_embeddings is not None:
            tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        tensor = self.layer_norm_emb(tensor)
        tensor = F.dropout(tensor, p=self.dropout, training=self.training)

        # transformer layers
        for i in range(self.n_layers):
            # self attention
            attn = self.attentions[i](tensor, src_mask, kv=src_enc)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            tensor = tensor + attn
            tensor = self.layer_norm1[i](tensor)

            # FFN
            tensor = tensor + self.ffns[i](tensor)
            tensor = self.layer_norm2[i](tensor)

        # # move back sequence length to dimension 0
        # tensor = tensor.transpose(0, 1)

        # return tensor  # (slen, bs, dim)
        return self.proj(tensor)  # (bs, slen, output_dim)

    # def predict(self, tensor, pred_mask, y, weight=None):
    #     """
    #     Given the last hidden state, compute output and/or the loss.
    #     Inputs:
    #         tensor     (slen, bs, dim)
    #         pred_mask  (slen, bs, output_dim) mask for different dimension/length
    #         y          (pred_mask.sum(), ) labels for prediction
    #         weight     (pred_mask.sum(), ) weight for loss function
    #     """
    #     scores = self.proj(tensor)  # (slen, bs, output_dim)
    #     scores = scores[pred_mask]
    #     loss = F.mse_loss(scores.float(), y, reduction="none")
    #     if weight is None:
    #         # no reweighting, loss is just regular MSE
    #         loss = torch.mean(loss)
    #     else:
    #         # reweight by weight
    #         loss = torch.sum(loss * weight)
    #     return scores, loss

    def generate(
        self,
        src_enc,
        src_len,
        query_emb,
    ):
        """
        Generate a sequence at times specified in query_emb
        Inputs:
            src_enc    (bs, slen, dim) output from backbone model
            src_len    (bs, ) lengths of src_enc
            query_emb  (slen, dim)
        """

        tensor = self.forward(
            # "fwd",
            query_emb=query_emb,
            src_enc=src_enc,
            src_len=src_len,
        )  # (slen, bs, dim)

        return self.proj(tensor).float()  # (slen, bs, output_dim)


# modified from file models_gpt2_reproduce_mask.py
class ReproduceGPT2(nn.Module):
    def __init__(self, model_name="gpt2", pretrained=True):
        super(ReproduceGPT2, self).__init__()

        # Load the pre-trained GPT-2 model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        # Add the extra number and pad token to the vocabulary
        extra_tokens = ["<number_token>", "<pad>"]
        self.tokenizer.add_tokens(extra_tokens)

        config = GPT2Config.from_pretrained(model_name)
        config.resid_pdrop = 0
        config.attn_pdrop = 0
        config.embd_pdrop = 0
        config.summary_first_dropout = 0
        if pretrained:
            self.gpt2 = GPT2Model.from_pretrained(model_name, config=config)  # without LM Head
            # print(self.gpt2.config)
            # resize model for extra tokens
            self.gpt2.resize_token_embeddings(len(self.tokenizer))
        else:
            self.gpt2 = GPT2Model(config=config)  # without LM Head
            # print(self.gpt2.config)
            # resize model for extra tokens
            self.gpt2.resize_token_embeddings(len(self.tokenizer))

        self.token_type_embeddings = Embedding(2, config.n_embd)

        # # Define the language modeling head and tie its weights to the token embeddings
        self.lm_head = nn.Linear(self.gpt2.config.n_embd, self.tokenizer.vocab_size, bias=False)
        self.lm_head.weight = self.gpt2.wte.weight

    def forward(self, input_embeds, position_ids=None, token_type_ids=None, mask=None):
        """
        INPUTS:
            input_embeds:    (batch_size, seq_length, dim)
            position_ids:    (batch_size, )
            token_type_ids:  (batch_size, seq_length)
            mask:            (batch_size, seq_length, seq_length)
                             will replace the causal mask in the original model if not None
        """

        seq_len = input_embeds.size(1)

        if mask == "full":
            mask = torch.ones(seq_len, seq_len, dtype=bool).unsqueeze(0)  # (1, seq_length, seq_length)

        if position_ids is None:
            # Get position embeddings
            position_ids = (
                torch.arange(0, seq_len, dtype=torch.long, device=input_embeds.device)
                .unsqueeze(0)
                .expand(input_embeds.size(0), -1)
            )  # (batch_size, seq_length)
        position_embeddings = self.gpt2.wpe(position_ids)  # (batch_size, seq_length, hidden_size)

        if token_type_ids is not None:
            token_embeddings = self.token_type_embeddings(token_type_ids)
            transformed_input = (
                input_embeds + position_embeddings + token_embeddings
            )  # (batch_size, seq_length, hidden_size)
        else:
            transformed_input = input_embeds + position_embeddings  # (batch_size, seq_length, hidden_size)

        # Pass the transformed input through GPT-2
        # build causal mask
        hidden_state = self.gpt2(inputs_embeds=transformed_input, attention_mask=mask)[
            0
        ]  # (batch_size, seq_length, hidden_size)

        return hidden_state


class ModelWrapper(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        self.model = ReproduceGPT2()
        # self.tokenizer = self.model.tokenizer

        self.embedder_data = nn.Sequential(
            nn.Linear(1 + model_config["input_dimension"], model_config["emb_dim"]),
            nn.GELU(),
            nn.Linear(model_config["emb_dim"], model_config["emb_dim"]),
        )
        self.embedder_control = nn.Sequential(
            nn.Linear(1 + model_config["input_dimension"], model_config["emb_dim"]),
            nn.GELU(),
            nn.Linear(model_config["emb_dim"], model_config["emb_dim"]),
        )
        self.embedder_coeffs = nn.Sequential(
            nn.Linear(model_config["max_number_coeffs"], model_config["emb_dim"]),
            nn.GELU(),
            nn.Linear(model_config["emb_dim"], model_config["emb_dim"]),
        )

        self.output_head = CrossAttnTransformerModel(model_config)

    def text_test_output(self, test_tensor, position_ids, token_type_ids):
        """
        input: test_tensor (cropped tensor without descriptions )
        output: token_logits (one extra token than test_tensor for autoregressive generation)
        """     
        encoded_tensor = self.model(
            input_embeds=test_tensor, position_ids=position_ids, token_type_ids=token_type_ids, mask=None
        )  # (bs, slen, dim)

        token_logits = self.model.lm_head(encoded_tensor)  # (batch_size, seq_length, vocab_size)
        return token_logits

    def number_test_output(self, test_samples, queries):
        # first prepare and batch inputs
        tokenized_texts = batch_tokenize(self.model.tokenizer, test_samples["text"])
        word_embedding = get_word_embeddings(tokenized_texts, self.model.gpt2)
        data_embedding = get_data_embeddings([test_samples["data"], test_samples["control"],test_samples["coefficients"]],test_samples["dim"], self.embedder_data, self.embedder_control, self.embedder_coeffs)
        input_tensor, lengths, input_lengths, token_type_ids, position_ids, text_mask,_ = batch_inputs(
            data_embedding, word_embedding
        )

        encoded_tensor = self.model(
            input_embeds=input_tensor, position_ids=position_ids, token_type_ids=token_type_ids, mask=None
        )  # (bs, slen, dim)

        query_emb = self.output_head.get_query_emb(queries)  # (query_len, dim)

        numeric_output = self.output_head(
            query_emb,
            encoded_tensor,
            input_lengths,  # only input no descriptions input_lengths
        )  # (bs, query_len, output_dim)

        return numeric_output
    
    def forward(self, samples, queries):
        """
        queries:  (slen, )
        """
        # first prepare and batch inputs
        tokenized_texts = batch_tokenize(self.model.tokenizer, samples["text"])
        word_embedding = get_word_embeddings(tokenized_texts, self.model.gpt2)
        data_embedding = get_data_embeddings(
            [samples["data"], samples["control"], samples["coefficients"]],
            samples["dim"],
            self.embedder_data, 
            self.embedder_control, 
            self.embedder_coeffs
            )
        input_tensor, lengths, input_lengths, token_type_ids, position_ids, text_mask,_ = batch_inputs(
            data_embedding, word_embedding
        )
        # (bs, slen, dim), (bs, ), (bs, slen), (bs, slen)
        #TODO DONE number tokenize will not work if we don't have "control" 
        tokenized_data = number_tokenize(self.model.tokenizer, [samples["data"], samples["control"],samples["coefficients"]],samples["dim"] )
        # print("TOKENIZED DATA INFO", len(tokenized_data), tokenized_data[0].shape, len(tokenized_data[1]),tokenized_data[1][0].shape )
        
        label_indices = batch_indices(
            tokenized_data, tokenized_texts, self.model.tokenizer
        )  # (batch_size, seq_length+1) constains end of text index

        encoded_tensor = self.model(
            input_embeds=input_tensor, position_ids=position_ids, token_type_ids=token_type_ids, mask=None
        )  # (bs, slen, dim)

        query_emb = self.output_head.get_query_emb(queries)  # (query_len, dim)

        numeric_output = self.output_head(
            query_emb,
            encoded_tensor,
            input_lengths,  # only input no descriptions input_lengths
        )  # (bs, query_len, output_dim)

        token_logits = self.model.lm_head(encoded_tensor)  # (batch_size, seq_length, vocab_size)

        return numeric_output, token_logits, label_indices, text_mask

    #def text_test_model:
    #def number_test_model
