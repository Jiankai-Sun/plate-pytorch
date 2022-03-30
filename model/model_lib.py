import torch
import torch.nn as nn
import torch.nn.functional as F
from model.resnet3d_xl import Net
import numpy as np
from model import base_nets
from model.model_utils import init
from queue import PriorityQueue
import operator

class BC_MODEL(nn.Module):
    def __init__(self, args):
        super(BC_MODEL, self).__init__()
        args.hidden_size = 1024
        self.base = GoalAttentionModel(args, recurrent=True, hidden_size=args.hidden_size, mode=args.gpt_repr)
        self.train()

    def forward(self, global_img_input, local_img_input, box_input, video_label, last_action, roi_feat, is_inference=False):
        value = self.base(global_img_input, local_img_input, box_input, video_label, last_action, roi_feat, is_inference=is_inference)
        return value

class GoalAttentionModel(nn.Module):
    def __init__(self, args, recurrent=False, hidden_size=128, mode='one'):
        """
        mode: one, start_goal, patch
        """
        super(GoalAttentionModel, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))
        self.mode = mode
        self.args = args
        self.pred_state_action = args.pred_state_action
        self.sa_type = self.args.sa_type
        self.beam_width = self.args.beam_width
        if self.mode == 'patch':
            self.hidden_size = 1024  # 256
            # patch mode
            self.patch_dim = 256
            self.num_patches = int((1024 // self.patch_dim))
            self.flatten_dim = self.patch_dim
            self.linear_encoding = nn.Linear(self.flatten_dim, self.hidden_size)
            self.max_message_len = args.max_traj_len + 4 * self.num_patches
        elif self.mode == 'start_goal':
            self.max_message_len = args.max_traj_len + 1
            self.hidden_size = hidden_size
        else:
            self.max_message_len = args.max_traj_len
            self.hidden_size = hidden_size
        if self.pred_state_action:
            if self.sa_type == 'temporal_concat':
                self.max_message_len = self.max_message_len * 2 - 1  # (se, a_0, s_0, a_1, s_1, a_2)
        self.nr_frames = 4
        self.n_steps = {'23521': 6, '59684': 5, '71781': 8, '113766': 11, '105222': 6, '94276': 6, '53193': 6,
                        '105253': 11, '44047': 8, '76400': 10, '16815': 3, '95603': 7, '109972': 5, '44789': 8,
                        '40567': 11, '77721': 5, '87706': 9, '91515': 8}
        self.task_border = np.cumsum(list(self.n_steps.values()))

        self.dropout = nn.Dropout(0.3)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        ## lang encoder
        print('args.num_classes: ', args.num_classes)
        self.word_embeddings = nn.Embedding(args.num_classes, self.hidden_size)
        self.keep_size = 1
        ## language decoder
        if self.sa_type == 'feature_concat':
            self.lang_decoding = base_nets.LangDecode(hidden_size=self.hidden_size + self.keep_size * self.nr_frames, # self.hidden_size * 2,
                                                      max_message_len=self.max_message_len,
                                                      num_classes=args.num_classes,
                                                      sa_type=self.sa_type,
                                                      state_size=self.keep_size * self.nr_frames)
        else:
            self.lang_decoding = base_nets.LangDecode(hidden_size=self.hidden_size, max_message_len=self.max_message_len,
                                                  num_classes=args.num_classes)
        if self.args.dataset == 'actionet':
            self.i3D = Net(1*3, extract_features=True, loss_type='softmax')
            self.conv = nn.Conv3d(2048, 512, kernel_size=(1, 1, 1), stride=1)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train()

    def forward(self, global_img_input, local_img_input, box_input, video_label, last_action, roi_feat, is_inference):
        if self.args.dataset == 'crosstask':
            global_img_input = global_img_input.view(global_img_input.shape[0], global_img_input.shape[1] // 2, 2, global_img_input.shape[2])
            if self.pred_state_action:
                intermediate_img_input = global_img_input  # [:, :, 1:]
            global_img_input = global_img_input[:, [0, -1]].view(global_img_input.shape[0], -1, global_img_input.shape[-1])
            bs, T, feat = global_img_input.shape  # [30, 2, 3200]  # torch.Size([30, 4, 1024])
            H = W = int(np.sqrt(feat // (self.hidden_size // 4)))
        else:
            if self.pred_state_action:
                intermediate_img_input = global_img_input  # [:, :, 1:]
                global_img_input = global_img_input[:, :, [0, -1]]
            y_i3d, org_features = self.i3D(global_img_input)
            # Reduce dimension video_features - [V x 512 x T / 2 x 14 x 14]
            videos_features = self.conv(org_features)  # torch.Size([10, 512, 2, 14, 14])
            global_img_input = videos_features.view(videos_features.shape[0], self.nr_frames * (self.hidden_size // 4), -1)
            bs, T, feat = global_img_input.shape
            if int(np.sqrt(feat)) == np.sqrt(feat):
                H = W = int(np.sqrt(feat))
            else:
                H = int(np.sqrt(feat // 2))
                W = 2 * H

        if self.mode == 'one':
            videos_features = global_img_input.view(bs, self.nr_frames * (self.hidden_size // 4), 1, H, W).float()
            global_features = self.avgpool(videos_features).squeeze()
            global_features = self.dropout(global_features).unsqueeze(1)  # torch.Size([30, 1024])
        elif self.mode == 'start_goal':
            video_1 = global_img_input[:, :int(T / 2)]
            video_2 = global_img_input[:, int(T / 2):]
            video_1_features = video_1.view(bs, self.nr_frames * (self.hidden_size // 4), 1, H, -1).float()
            video_1_features = self.avgpool(video_1_features).squeeze()
            video_1_features = self.dropout(video_1_features)  # torch.Size([30, 1024])
            video_2_features = video_2.view(bs, self.nr_frames * (self.hidden_size // 4), 1, H, -1).float()
            video_2_features = self.avgpool(video_2_features).squeeze()
            video_2_features = self.dropout(video_2_features)  # torch.Size([30, 1024])
            global_features = torch.cat((video_1_features.unsqueeze(1), video_2_features.unsqueeze(1)), dim=1)
        elif self.mode == 'patch':
            x = (global_img_input.unfold(2, self.patch_dim, self.patch_dim).contiguous())
            x = x.view(bs, T, self.num_patches, self.patch_dim)
            x = x.permute(0, 2, 3, 1).contiguous()
            x = x.view(x.size(0), T * self.num_patches, self.flatten_dim).float()
            global_features = self.linear_encoding(x)
        message = self.word_embeddings(video_label.squeeze(-1).long())
        if self.pred_state_action:  # and self.args.dataset != 'crosstask'
            if self.args.dataset == 'crosstask':
                inter_bs, inter_T, two, feat_dim = intermediate_img_input.shape
            else:
                inter_bs, c, inter_T, h, w = intermediate_img_input.shape
            state_emb = []
            for i in range(inter_T-1):
                if self.args.dataset == 'crosstask':
                    inter_img_input = intermediate_img_input[:, i:(i+2)].view(global_img_input.shape[0], -1, global_img_input.shape[-1])
                    H = W = int(np.sqrt(feat // (self.keep_size)))
                else:
                    inter_img_input = intermediate_img_input[:, :, i:(i+2)] # (0,1) group, (1,2) group, (2,3) group for inter_T = 4
                    inter_y_i3d, inter_org_features = self.i3D(inter_img_input)
                    # Reduce dimension video_features - [V x 512 x T / 2 x 14 x 14]
                    inter_videos_features = self.conv(inter_org_features)  # torch.Size([10, 512, 2, 14, 14])
                    inter_img_input = inter_videos_features.view(inter_videos_features.shape[0], self.nr_frames * self.keep_size, -1)
                if self.mode in ['one', 'patch']:
                    inter_videos_features = inter_img_input.view(bs, self.nr_frames * self.keep_size, 1, H, W).float()
                    inter_global_features = self.avgpool(inter_videos_features).squeeze()
                    inter_global_features = self.dropout(inter_global_features).unsqueeze(1)  # torch.Size([30, 1024])
                elif self.mode == 'start_goal':
                    inter_video_1 = inter_img_input[:, :int(T / 2)]
                    inter_video_2 = inter_img_input[:, int(T / 2):]
                    inter_video_1_features = inter_video_1.view(bs, self.nr_frames * self.keep_size, 1, H, -1).float()
                    inter_video_1_features = self.avgpool(inter_video_1_features).squeeze()
                    inter_video_1_features = self.dropout(inter_video_1_features)  # torch.Size([30, 1024])
                    inter_video_2_features = inter_video_2.view(bs, self.nr_frames * self.keep_size, 1, H, -1).float()
                    inter_video_2_features = self.avgpool(inter_video_2_features).squeeze()
                    inter_video_2_features = self.dropout(inter_video_2_features)  # torch.Size([30, 1024])
                    inter_global_features = torch.cat((inter_video_1_features.unsqueeze(1), inter_video_2_features.unsqueeze(1)), dim=1)
                elif self.mode == 'patch':
                    inter_x = (inter_img_input.unfold(2, self.patch_dim, self.patch_dim).contiguous())
                    inter_x = inter_x.view(bs, T, self.num_patches, self.patch_dim)
                    inter_x = inter_x.permute(0, 2, 3, 1).contiguous()
                    inter_x = inter_x.view(inter_x.size(0), T * self.num_patches, self.flatten_dim).float()
                    inter_global_features = self.linear_encoding(inter_x)
                state_emb.append(inter_global_features)
            state_emb = torch.cat(state_emb, dim=1)
            if self.sa_type == 'temporal_concat':
                message_state = []
                for i in range(message.shape[1] - 1):
                    message_state.append(message[:, i:(i+1)])
                    message_state.append(state_emb[:, i:(i+1)])
                message_state.append(message[:, -1:])
                message = torch.cat(message_state, dim=1)
            elif self.sa_type == 'feature_concat':
                empty_state = torch.zeros((state_emb.shape[0], 1, state_emb.shape[-1])).cuda()
                state_emb = torch.cat((state_emb, empty_state), dim=1)
                message = torch.cat((state_emb, message), dim=-1)
                first_step = torch.cat((global_features, torch.zeros(message.shape[0], global_features.shape[1], message.shape[-1] - global_features.shape[-1]).cuda()), dim=-1)
                message = torch.cat((first_step, message), dim=1)

        if self.pred_state_action and self.sa_type == 'feature_concat':
            message_input = message  # torch.cat((state_emb, message), dim=-1)
        else:
            message_input = torch.cat([global_features, message], dim=1)
        if self.pred_state_action:
            if self.sa_type == 'temporal_concat':
                cls_output, state_feat = self.lang_decoding.get_temporal_feat(message_input)
                cls_output = cls_output[:, :-1]
                state_feat = state_feat[:, :-1]
            elif self.sa_type == 'feature_concat':
                cls_output, state_feat = self.lang_decoding.get_feat(message_input)
                cls_output = cls_output[:, :-1]
                state_feat = state_feat[:, :-2]
        else:
            cls_output = self.lang_decoding(message_input)[:, :-1]
        if self.mode == 'start_goal':
            cls_output = cls_output[:, 1:]
            if self.pred_state_action:
                state_feat = state_feat[:, 1:]
        elif self.mode == 'patch':
            cls_output = cls_output[:, (T * self.num_patches - 1):]
            if self.pred_state_action:
                state_feat = state_feat[:, (T * self.num_patches - 1):]
        if self.pred_state_action:
            if self.sa_type == 'temporal_concat':
                state_feat = state_feat[:, 1::2]
                state_loss = torch.tensor(0.)
                cls_output = cls_output[:, 0::2]
            elif self.sa_type == 'feature_concat':
                state_loss = F.mse_loss(state_feat, state_emb[:, :-1], reduction='mean') * 0.1  # / state_emb.shape[0] * 0.1
        else:
            state_loss = torch.tensor(0.)
        return cls_output, state_loss

    def model_get_action(self, global_img_input, local_img_input, box_input, video_label, last_action, roi_feat, is_inference):
        if self.args.dataset == 'crosstask':
            global_img_input = global_img_input.view(global_img_input.shape[0], global_img_input.shape[1] // 2, 2,
                                                     global_img_input.shape[2])
            global_img_input = global_img_input[:, [0, -1]].view(global_img_input.shape[0], -1, global_img_input.shape[-1])
            bs, T, feat = global_img_input.shape  # [30, 2, 3200]  # torch.Size([30, 4, 1024])
            H = W = int(np.sqrt(feat // (self.hidden_size // 4)))
        else:
            if self.pred_state_action:
                global_img_input = global_img_input[:, :, [0, -1]]
            # org_features - [V x 2048 x T / 2 x 14 x 14]
            y_i3d, org_features = self.i3D(global_img_input)
            videos_features = self.conv(org_features)  # torch.Size([10, 512, 2, 14, 14])
            global_img_input = videos_features.view(videos_features.shape[0], self.nr_frames * self.keep_size, -1)
            bs, T, feat = global_img_input.shape
            if int(np.sqrt(feat)) == np.sqrt(feat):
                H = W = int(np.sqrt(feat))
            else:
                H = int(np.sqrt(feat // 2))
                W = 2 * H

        if self.mode == 'one':
            videos_features = global_img_input.view(bs, self.nr_frames * (self.hidden_size // 4), 1, H, W).float()
            global_features = self.avgpool(videos_features).squeeze()
            global_features = self.dropout(global_features).unsqueeze(1)  # torch.Size([30, 1024])
        elif self.mode == 'start_goal':
            video_1 = global_img_input[:, :int(T / 2)]
            video_2 = global_img_input[:, int(T / 2):]
            video_1_features = video_1.view(bs, self.nr_frames * (self.hidden_size // 4), 1, H, -1).float()
            video_1_features = self.avgpool(video_1_features).squeeze()
            video_1_features = self.dropout(video_1_features)  # torch.Size([30, 1024])
            video_2_features = video_2.view(bs, self.nr_frames * (self.hidden_size // 4), 1, H, -1).float()
            video_2_features = self.avgpool(video_2_features).squeeze()
            video_2_features = self.dropout(video_2_features)  # torch.Size([30, 1024])
            global_features = torch.cat((video_1_features.unsqueeze(1), video_2_features.unsqueeze(1)), dim=1)
        elif self.mode == 'patch':
            x = (global_img_input.unfold(2, self.patch_dim, self.patch_dim).contiguous())
            x = x.view(bs, T, self.num_patches, self.patch_dim)
            x = x.permute(0, 2, 3, 1).contiguous()
            x = x.view(x.size(0), T * self.num_patches, self.flatten_dim).float()
            global_features = self.linear_encoding(x)

        if self.pred_state_action and self.sa_type == 'feature_concat':
            empty_state = torch.zeros(global_features.shape[0], 1, self.hidden_size + self.nr_frames * self.keep_size - global_features.shape[-1]).cuda()
            global_features = torch.cat((global_features, empty_state), dim=-1)

        message = self.word_embeddings(video_label.squeeze(-1).long())

        max_message_len = self.args.max_traj_len
        domain_prior_list = []
        if self.args.search_method == 'beam':
            message, domain_prior_list = self.beam_decode(max_message_len, global_features, message, task_border=self.task_border)
        else:
            message = self.gen_message(max_message_len, global_features, message)
        return message, domain_prior_list

    def gen_message(self, max_message_len, belief_goal_context, message, sample=True):
        temperature = 0.9
        sampled_ids = []
        sampled_probs = []
        message_previous = belief_goal_context
        for message_step in range(max_message_len):
            if self.sa_type == 'feature_concat':
                message_output, state_feat = self.lang_decoding.get_feat(message_previous)
                state_feat = state_feat[:, -1:, :]
            else:
                message_output = self.lang_decoding(message_previous)
            message_output = message_output[:, -1, :] / temperature
            message_probs = F.softmax(message_output, dim=-1)
            if sample:
                message_prediction = torch.multinomial(message_probs, num_samples=1)
            else:
                _, message_prediction = torch.topk(message_probs, k=1, dim=-1)
            message = self.word_embeddings(message_prediction)  # [1, 1, 64]
            if self.sa_type == 'feature_concat':
                message = torch.cat([state_feat, message], dim=-1)
            message_previous = torch.cat([message_previous, message], dim=1)
            sampled_ids.append(message_prediction[:, -1])
            sampled_probs.append(message_probs.unsqueeze(1))
            if self.pred_state_action and message_step < (max_message_len-1) and self.sa_type == 'temporal_feature':
                message_output = self.lang_decoding(message_previous)
                message_output = message_output[:, -1, :] / temperature
                message_probs = F.softmax(message_output, dim=-1)
                if sample:
                    message_prediction = torch.multinomial(message_probs, num_samples=1)
                else:
                    _, message_prediction = torch.topk(message_probs, k=1, dim=-1)
                message = self.word_embeddings(message_prediction)  # [1, 1, 64]
                message_previous = torch.cat([message_previous, message], dim=1)

        message_next = torch.stack(sampled_probs, dim=1)
        return message_next

    def beam_decode(self, max_message_len, belief_goal_context, message, task_border, use_task_border=False):
        '''
        # https://github.com/budzianowski/PyTorch-Beam-Search-Decoding/blob/master/decode_beam.py
        :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
        :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
        :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
        :return: decoded_batch
        '''

        topk = 1  # how many sentence do you want to generate
        decoded_batch = []
        temperature = 0.9
        domain_prior_list = []
        # decoding goes sentence by sentence
        for idx in range(belief_goal_context.size(0)):
            # Start with the start of the sentence token
            message_previous = belief_goal_context[idx:(idx+1)]

            # Number of sentence to generate
            endnodes = []

            # starting node -  hidden vector, previous node, word id, logp, length
            node = BeamSearchNode(hiddenstate=message_previous, previousNode=None, wordId="", logProb=0, length=1)
            nodes = PriorityQueue()

            # start the queue
            nodes.put((-node.eval(), node))
            domain_prior_list_each_img = []
            # start beam search
            for message_step in range(max_message_len):
                # fetch the best node
                prev_score, n = nodes.get()
                # decoder_input = n.wordid
                message_previous = n.h
                if self.sa_type == 'feature_concat':
                    message_output, state_feat = self.lang_decoding.get_feat(message_previous)
                    state_feat = state_feat[:, -1:, :]
                else:
                    message_output = self.lang_decoding(message_previous)
                message_output = message_output[:, -1, :] / temperature
                message_probs = F.softmax(message_output, dim=-1)

                # PUT HERE REAL BEAM SEARCH OF TOP
                if self.args.dataset == 'crosstask' and use_task_border:
                    _, indexes_1 = torch.topk(message_probs, self.beam_width)
                    valid_interval_start = valid_interval_end = None
                    for task_idx in range(task_border.shape[0]):
                        if task_idx == 0 and (indexes_1[0][0] < task_border[task_idx]):
                            valid_interval_start = 0
                            valid_interval_end = task_border[task_idx]
                        elif task_idx > 0 and indexes_1[0][0] >= task_border[task_idx-1] \
                                and indexes_1[0][0] < task_border[task_idx]:
                            valid_interval_start = task_border[task_idx-1]
                            valid_interval_end = task_border[task_idx]
                        elif indexes_1[0][0] == 133:
                            valid_interval_start = task_border[-2]
                            valid_interval_end = task_border[-1]
                            assert valid_interval_end == 133, 'valid_interval_end: {}'.format(valid_interval_end)
                    assert valid_interval_start is not None and valid_interval_end is not None, \
                        'valid_interval_start: {}, valid_interval_end: {}, indexes_1: {}'.format(valid_interval_start, valid_interval_end, indexes_1)

                    message_probs[:, :valid_interval_start] = 0.
                    message_probs[:, valid_interval_end:] = 0.

                log_prob, indexes = torch.topk(message_probs, self.beam_width)
                if use_task_border:
                    if self.args.dataset == 'crosstask':
                        domain_prior_list_each_img.append(1. - torch.prod((indexes == indexes_1).float()).cpu().numpy())
                    else:
                        domain_prior_list_each_img.append(0)

                nextnodes = []

                for new_k in range(self.beam_width):
                    decoded_t = indexes[0][new_k].view(1, -1)
                    message = self.word_embeddings(decoded_t)  # [1, 1, 64]
                    if self.sa_type == 'feature_concat':
                        message = torch.cat([state_feat, message], dim=-1)
                    message_previous = torch.cat([message_previous, message], dim=1)  # torch.Size([1, 2, 1024])
                    if self.pred_state_action and message_step < (max_message_len-1) and self.sa_type == 'temporal_feature':
                        message_output = self.lang_decoding(message_previous)
                        message_output = message_output[:, -1, :] / temperature
                        message_probs = F.softmax(message_output, dim=-1)
                        _, message_prediction = torch.topk(message_probs, k=1, dim=-1)
                        message = self.word_embeddings(message_prediction)  # [1, 1, 64]
                        message_previous = torch.cat([message_previous, message], dim=1)

                    log_p = log_prob[0][new_k].item()  # the larger, the better
                    node = BeamSearchNode(hiddenstate=message_previous, previousNode=n, wordId=decoded_t,
                                          logProb=n.logp + log_p, length=n.leng + 1, message_prob=log_p)
                    score = -node.eval()  # the less, the better
                    nextnodes.append((score, node))
                    assert score < prev_score, 'score: {} should < prev_score: {}'.format(score, prev_score)

                # put them into queue
                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    nodes.put((score, nn))
            if use_task_border:
                domain_prior_list.append(max(domain_prior_list_each_img))
            # choose nbest paths, back trace them
            if len(endnodes) == 0:
                # The lowest valued entries are retrieved first, -node.eval()
                endnodes = [nodes.get() for _ in range(topk)]

            utterances = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance = []
                utterance.append(n.wordid)
                while n.prevNode != None:
                    n = n.prevNode
                    if n.wordid != "":
                        utterance.append(n.wordid)
                utterance = utterance[::-1]
                utterance = torch.stack(utterance, dim=1)
                utterances.append(utterance)
                utterances = torch.cat(utterances, dim=0)

            decoded_batch.append(utterances)
        decoded_batch = torch.cat(decoded_batch, dim=0)
        decoded_batch = torch.nn.functional.one_hot(decoded_batch, num_classes=self.args.num_classes).unsqueeze(-2).double()
        return decoded_batch, domain_prior_list

class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length, message_prob=None):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length
        self.message_prob = message_prob

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward
        return self.logp + alpha * reward

    # defining comparators less_than and equals
    def __lt__(self, other):
        return self.logp < other.logp
