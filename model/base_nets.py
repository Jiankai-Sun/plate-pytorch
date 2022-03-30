from torch import nn
import pdb
import torch
from model.minGPT import Block
import torch.nn.functional as F

class ObjNameCoordStateEncode(nn.Module):
    def __init__(self, output_dim=128, num_node_name_classes=102, num_node_states=5):
        super(ObjNameCoordStateEncode, self).__init__()
        assert output_dim % 2 == 0
        self.output_dim = output_dim

        self.class_embedding = nn.Embedding(num_node_name_classes, int(output_dim / 2))
        self.state_embedding = nn.Linear(num_node_states, int(output_dim / 2))
        self.coord_embedding = nn.Sequential(nn.Linear(6, int(output_dim / 2)),
                                             nn.ReLU(),
                                             nn.Linear(int(output_dim / 2), int(output_dim / 2)))
        inp_dim = int(3*output_dim/2)
        self.combine = nn.Sequential(nn.ReLU(), nn.Linear(inp_dim, output_dim))

    def forward(self, class_name_ids, state_ids, coords):

        class_embedding = self.class_embedding(class_name_ids.long())
        state_embedding = self.state_embedding(state_ids) # [4, 81, 5] -> [4, 81, 64]
        coord_embedding = self.coord_embedding(coords) # [4, 81, 6] -> [4, 81, 64]
        inp = torch.cat([class_embedding, coord_embedding, state_embedding], dim=2) # [4, 81, 192]

        return self.combine(inp)




class Transformer(nn.Module):
    def __init__(self, in_feat, out_feat, dropout=0.2, activation='relu', nhead=2):
        super(Transformer, self).__init__()
        encoder_layer = nn.modules.TransformerEncoderLayer(d_model=in_feat, nhead=nhead,
                                                           dim_feedforward=out_feat, dropout=dropout)

        self.transformer = nn.modules.TransformerEncoder(
            encoder_layer,
            num_layers=1)

        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, inputs, mask_nodes=None):


        mask_nodes = 1 - mask_nodes # mask_nodes.sum(-1) shouldn't be 0
        outputs = self.transformer(inputs.transpose(0, 1), src_key_padding_mask=mask_nodes.bool()) # [81, 4, 128]
        outputs = outputs.transpose(0, 1)
        return outputs

class TaskEncoder(nn.Module):
    def __init__(self, num_node_name_classes, out_feat):
        super(TaskEncoder, self).__init__()

        # FIXME: assuming that thepad character is the last element of node name vocabulary
        self.object_embedding = nn.Embedding(num_node_name_classes, out_feat)

        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(out_feat, out_feat)
        )

    def forward(self, env_task_goal_index_tem, env_task_goal_mask_tem):
        env_task_goal = self.object_embedding(env_task_goal_index_tem.long())
        env_task_goal = self.fc(env_task_goal)

        # Mean pool over subgoals
        env_task_goal_mask_tem = env_task_goal_mask_tem.unsqueeze(-1)
        env_task_goal = (env_task_goal * env_task_goal_mask_tem).sum(1) / (1e-9 + env_task_goal_mask_tem.sum(1))

        return env_task_goal #.mean(1)




class LangEncode(nn.Module):
    def __init__(self, hidden_size=128, max_message_len=None):
        super(LangEncode, self).__init__()
        self.n_layer = 1  # 8
        self.n_head = 1  # 8
        self.n_embd = hidden_size
        self.embd_pdrop = 0.3
        self.resid_pdrop = 0.3
        self.attn_pdrop = 0.3
        self.max_message_len = max_message_len
        self.block_size = max_message_len + 1


        ## shared by language encoder and decoder
        self.pos_emb = nn.Parameter(torch.zeros(1, self.block_size, self.n_embd))
        self.drop = nn.Dropout(self.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(self) for _ in range(self.n_layer)])
        self.ln_f = nn.LayerNorm(self.n_embd)


        self.enc_comm = nn.Linear(hidden_size, hidden_size)
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, last_opponent_message, last_opponent_message_mask_tem, last_message, last_message_mask_tem):

        position_embeddings = self.pos_emb[:, :self.max_message_len, :]
        a0_lang_output = self.drop(last_opponent_message + position_embeddings)
        a0_lang_output = self.blocks(a0_lang_output)
        a0_lang_output = self.ln_f(a0_lang_output) # [4, 40, 128]

        a1_lang_output = self.drop(last_message + position_embeddings)
        a1_lang_output = self.blocks(a1_lang_output)
        a1_lang_output = self.ln_f(a1_lang_output) # [4, 40, 128]

        a0_lang_output = self.enc_comm(a0_lang_output)
        a1_lang_output = self.enc_comm(a1_lang_output)

        # Mean pool over words
        last_opponent_message_mask_tem = last_opponent_message_mask_tem.unsqueeze(-1)
        a0_lang_output = (a0_lang_output * last_opponent_message_mask_tem).sum(1) / (1e-9 + last_opponent_message_mask_tem.sum(1)) # [4, 128]

        last_message_mask_tem = last_message_mask_tem.unsqueeze(-1)
        a1_lang_output = (a1_lang_output * last_message_mask_tem).sum(1) / (1e-9 + last_message_mask_tem.sum(1)) # [4, 128]

        return a0_lang_output, a1_lang_output


class LangDecode(nn.Module):
    def __init__(self, hidden_size=128, max_message_len=None, num_classes=None, sa_type=None, state_size=0):
        super(LangDecode, self).__init__()
        self.n_layer = 8  # 8
        self.n_head = 1  # 8
        self.n_embd = hidden_size
        self.embd_pdrop = 0.1
        self.resid_pdrop = 0.1
        self.attn_pdrop = 0.1
        self.block_size = max_message_len + 1
        self.num_classes = num_classes
        self.sa_type = sa_type
        self.state_size = state_size

        ## shared by language encoder and decoder
        self.pos_emb = nn.Parameter(torch.zeros(1, self.block_size, self.n_embd))
        self.drop = nn.Dropout(self.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(self) for _ in range(self.n_layer)])
        self.ln_f = nn.LayerNorm(self.n_embd)
        if self.sa_type == 'feature_concat':
            self.gen_comm = nn.Linear(self.n_embd - self.state_size, self.num_classes, bias=False)
        else:
            self.gen_comm = nn.Linear(hidden_size, self.num_classes, bias=False)
        self.apply(self._init_weights)
        print("number of gpt parameters: {}".format(sum(p.numel() for p in self.parameters())))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, message_input, message_mask_tem=None):
        b, t, n_embd = message_input.size()  # (bs, T, 1024)

        position_embeddings = self.pos_emb[:, :t, :]
        message_output = self.drop(message_input + position_embeddings)  # (bs, T, 1024)
        message_output = self.blocks(message_output)  # (bs, T, 1024)
        message_output = self.ln_f(message_output)  # (bs, T, 1024)

        message_output = self.gen_comm(message_output)  # (bs, T, 34)
        message_output = message_output[:,:t,:]

        return message_output

    def get_temporal_feat(self, message_input, message_mask_tem=None):
        b, t, n_embd = message_input.size()

        position_embeddings = self.pos_emb[:, :t, :]
        message_output = self.drop(message_input + position_embeddings)
        message_output = self.blocks(message_output)
        message_output_feat = self.ln_f(message_output)

        message_output = self.gen_comm(message_output_feat)
        message_output = message_output[:,:t,:]

        return message_output, message_output_feat

    def get_feat(self, message_input, message_mask_tem=None):
        b, t, n_embd = message_input.size()

        position_embeddings = self.pos_emb[:, :t, :]
        message_output = self.drop(message_input + position_embeddings)
        message_output = self.blocks(message_output)
        message_output = self.ln_f(message_output)
        split_output = torch.split(message_output, [self.state_size, self.n_embd - self.state_size], dim=-1)
        state_output_feat, message_output_feat = split_output[0], split_output[1]
        state_output_feat = state_output_feat[:, :t, :]

        message_output = self.gen_comm(message_output_feat)
        message_output = message_output[:,:t,:]

        return message_output, state_output_feat

class SimpleAttention(nn.Module):
    def __init__(self, n_features, n_hidden, key=True, query=False, memory=False):
        super().__init__()
        self.key = key
        self.query = query
        self.memory = memory
        if self.key:
            self.make_key = nn.Linear(n_features, n_hidden)
        if self.query:
            self.make_query = nn.Linear(n_features, n_hidden)
        if self.memory:
            self.make_memory = nn.Linear(n_features, n_hidden)
        self.n_out = n_hidden

    def forward(self, features, hidden, mask=None):
        if self.key:
            key = self.make_key(features)
        else:
            key = features

        if self.memory:
            memory = self.make_memory(features)
        else:
            memory = features

        if self.query:
            query = self.make_query(hidden)
        else:
            query = hidden

        # attention
        scores = (key * query).sum(dim=2)
        if mask is not None:
            scores += mask * -99999

        distribution = F.softmax(scores, dim=1)
        weighted = (memory * distribution.unsqueeze(2))
        summary = weighted.sum(dim=1)
        return summary, scores