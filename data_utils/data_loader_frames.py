import os
from os.path import join
from torchvision.transforms import Compose
import numpy as np
from PIL import Image
import torch
from data_utils import gtransforms
import json
import pickle
import cv2
import re
import math


def read_assignment(T, task, all_n_steps, path):
    base = 0
    for k, v in all_n_steps.items():
        if k == task:
            break
        base += v
    Y = np.zeros([T, sum(all_n_steps.values())], dtype=np.uint8)
    legal_range = []
    with open(path, 'r') as f:
        for line in f:
            step, start, end = line.strip().split(',')
            start = int(math.floor(float(start)))
            end = int(math.ceil(float(end)))
            step = int(step) - 1 + base
            Y[start:end, step] = 1
            legal_range.append((start, end))

    return Y, legal_range

def get_vids(path):
    task_vids = {}
    with open(path, 'r') as f:
        for line in f:
            task, vid, url = line.strip().split(',')
            if task not in task_vids:
                task_vids[task] = []
            task_vids[task].append(vid)
    return task_vids


def read_task_info(path):
    titles = {}
    urls = {}
    n_steps = {}
    steps = {}
    with open(path, 'r') as f:
        idx = f.readline()
        while idx is not '':
            idx = idx.strip()
            titles[idx] = f.readline().strip()
            urls[idx] = f.readline().strip()
            n_steps[idx] = int(f.readline().strip())
            steps[idx] = f.readline().strip().split(',')
            next(f)
            idx = f.readline()
    return {'title': titles, 'url': urls, 'n_steps': n_steps, 'steps': steps}


def random_split(task_vids, test_tasks, n_train):
    train_vids = {}
    test_vids = {}
    for task, vids in task_vids.items():
        if task in test_tasks:
            train_vids[task] = np.random.choice(
                vids, int(len(vids) * n_train), replace=False).tolist()
            test_vids[task] = [
                vid for vid in vids if vid not in train_vids[task]]
        else:
            train_vids[task] = vids
    return train_vids, test_vids


def random_split_v0(task_vids, test_tasks, n_train):
    train_vids = {}
    test_vids = {}
    for task, vids in task_vids.items():
        if task in test_tasks and len(vids) > n_train:
            train_vids[task] = np.random.choice(
                vids, n_train, replace=False).tolist()
            test_vids[task] = [
                vid for vid in vids if vid not in train_vids[task]]
        else:
            train_vids[task] = vids
    return train_vids, test_vids

class VideoFolder(torch.utils.data.Dataset):
    """
    Something-Something dataset based on *frames* extraction
    """
    def __init__(self,
                 root,
                 file_input,
                 file_labels,
                 frames_duration,
                 args=None,
                 multi_crop_test=False,
                 sample_rate=2,
                 is_test=False,
                 is_val=False,
                 num_boxes=10,
                 model=None,
                 if_augment=True,
                 max_sentence_length=None,
                 clean_inst=True,
                 max_traj_len=10):
        """
        :param root: data root path
        :param file_input: inputs path
        :param file_labels: labels path
        :param frames_duration: number of frames
        :param multi_crop_test:
        :param sample_rate: FPS
        :param is_test: is_test flag
        :param k_split: number of splits of clips from the video
        :param sample_split: how many frames sub-sample from each clip
        """
        self.in_duration = frames_duration
        self.coord_nr_frames = self.in_duration // 2
        self.multi_crop_test = multi_crop_test
        self.sample_rate = sample_rate
        self.if_augment = if_augment
        self.is_val = is_val
        self.use_all_vocab = False
        self.data_root = root
        self.args = args

        self.max_traj_len = max_traj_len
        print('self.max_traj_len: ', self.max_traj_len)
        if args.dataset == 'crosstask':
            '''
            .
            └── crosstask
                ├── crosstask_features
                └── crosstask_release
                    ├── tasks_primary.txt
                    ├── videos.csv
                    └── videos_val.csv
            '''

            val_csv_path = os.path.join(
                root, 'crosstask_release', 'videos_val.csv')
            video_csv_path = os.path.join(
                root, 'crosstask_release', 'videos.csv')
            self.features_path = os.path.join(root, 'crosstask_features')
            # baseline
            self.constraints_path = os.path.join(
                root, 'crosstask_release', 'annotations')

            all_task_vids = get_vids(video_csv_path)
            val_vids = get_vids(val_csv_path)
            if is_val:
                task_vids = val_vids
            else:
                task_vids = {task: [vid for vid in vids if task not in val_vids or vid not in val_vids[task]] for
                             task, vids in
                             all_task_vids.items()}
            primary_info = read_task_info(os.path.join(
                root, 'crosstask_release', 'tasks_primary.txt'))
            test_tasks = set(primary_info['steps'].keys())

            self.n_steps = primary_info['n_steps']
            all_tasks = set(self.n_steps.keys())


            task_vids = {task: vids for task,
                         vids in task_vids.items() if task in all_tasks}

            cross_task_data_name = 'cross_task_data_{}.json'.format(is_val)
            if os.path.exists(cross_task_data_name):
                with open(cross_task_data_name, 'r') as f:
                    self.json_data = json.load(f)
                print('Loaded {}'.format(cross_task_data_name))
            else:
                all_vids = []
                for task, vids in task_vids.items():
                    all_vids.extend([(task, vid) for vid in vids])
                json_data = []
                for idx in range(len(all_vids)):
                    task, vid = all_vids[idx]
                    video_path = os.path.join(
                        self.features_path, str(vid)+'.npy')
                    json_data.append({'id': {'vid': vid, 'task': task, 'feature': video_path, 'bbox': ''},
                                      'instruction_len': self.n_steps[task]})
                print('All primary task videos: {}'.format(len(json_data)))
                self.json_data = json_data
                with open('cross_task_data.json', 'w') as f:
                    json.dump(json_data, f)
                print('Save to {}'.format(cross_task_data_name))
        elif args.dataset == 'actionet':
            with open(root, 'rb') as handle:
                self.json_data = pickle.load(handle)
        else:
            raise NotImplementedError(
                'Dataset {} is not implemented'.format(args.dataset))

        self.model = model
        self.num_boxes = num_boxes
        # Prepare data for the data loader
        self.prepare_data()
        self.pre_resize_shape = (256, 340)
        self.ignore_class = ['dining_room', ]
        # boxes_path = args.tracked_boxes
        # self.box_annotations = []
        self.M = 2
        print('... Loading box annotations might take a minute ...')

        self.img_mean = [0.485, 0.456, 0.406]
        self.img_std = [0.229, 0.224, 0.225]

        # Transformations
        if not self.is_val:
            self.transforms = [
                gtransforms.GroupResize((224, 224)),
            ]
        elif self.multi_crop_test:
            self.transforms = [
                gtransforms.GroupResize((256, 256)),
                gtransforms.GroupRandomCrop((256, 256)),
            ]
        else:
            self.transforms = [
                gtransforms.GroupResize((224, 224))
                # gtransforms.GroupCenterCrop(256),
            ]
        self.transforms += [
            gtransforms.ToTensor(),
            gtransforms.GroupNormalize(self.img_mean, self.img_std),
        ]
        self.transforms = Compose(self.transforms)

        if self.if_augment:
            if not self.is_val:  # train, multi scale cropping
                self.random_crop = gtransforms.GroupMultiScaleCrop(output_size=224,
                                                                   scales=[1, .875, .75])
            else:  # val, only center cropping
                self.random_crop = gtransforms.GroupMultiScaleCrop(output_size=224,
                                                                   scales=[1],
                                                                   max_distort=0,
                                                                   center_crop_only=True)
        else:
            self.random_crop = None

    def prepare_data(self):
        """
        This function creates 3 lists: vid_names, labels and frame_cnts
        :return:
        """
        if self.args.dataset == 'crosstask':
            print("Loading label strings")
            vid_names = []
            frame_cnts = []
            for listdata in self.json_data:
                vid_names.append(listdata['id'])
                frame_cnts.append(listdata['instruction_len'])
            self.vid_names = vid_names
            self.frame_cnts = frame_cnts
        elif self.args.dataset == 'actionet':
            self.trajectory = []
            for task_id in self.json_data.keys():
                for task_name in self.json_data[task_id].keys():
                    trajectory = self.json_data[task_id][task_name]
                    if len(trajectory) > 0:
                        self.trajectory.append(trajectory)
            trajectory_num = len(self.trajectory)
            if self.is_val:
                self.trajectory = self.trajectory[int(trajectory_num * 0.8):]
            else:
                self.trajectory = self.trajectory[:int(trajectory_num * 0.8)]
            print('self.trajectory: ', len(self.trajectory))

    # todo: might consider to replace it to opencv, should be much faster
    def load_frame(self, vid_name, frame_idx):
        """
        Load frame
        :param vid_name: video name
        :param frame_idx: index
        :return:
        """
        return Image.open(join(os.path.dirname(self.data_root), 'frames', vid_name, '%04d.jpg' % (frame_idx + 1))).convert('RGB')

    def _sample_indices(self, nr_video_frames):
        average_duration = nr_video_frames * 1.0 / self.coord_nr_frames
        if average_duration > 0:
            offsets = np.multiply(list(range(self.coord_nr_frames)), average_duration) \
                + np.random.uniform(0, average_duration,
                                    size=self.coord_nr_frames)
            offsets = np.floor(offsets)
        elif nr_video_frames > self.coord_nr_frames:
            offsets = np.sort(np.random.randint(
                nr_video_frames, size=self.coord_nr_frames))
        else:
            offsets = np.zeros((self.coord_nr_frames,))
        offsets = list(map(int, list(offsets)))
        return offsets

    def _get_val_indices(self, nr_video_frames):
        if nr_video_frames > self.coord_nr_frames:
            tick = nr_video_frames * 1.0 / self.coord_nr_frames
            offsets = np.array([int(tick / 2.0 + tick * x)
                                for x in range(self.coord_nr_frames)])
        else:
            offsets = np.zeros((self.coord_nr_frames,))
        offsets = list(map(int, list(offsets)))
        return offsets

    def _mask_state(self, state, char_index):
        chars = [node for node in state["nodes"]
                 if node["category"] == "Characters"]
        chars.sort(key=lambda node: node['id'])

        character = chars[char_index]
        # find character
        character_id = character["id"]
        id2node = {node['id']: node for node in state['nodes']}
        inside_of, is_inside, edge_from = {}, {}, {}

        grabbed_ids = []
        for edge in state['edges']:

            if edge['relation_type'] == 'INSIDE':

                if edge['to_id'] not in is_inside.keys():
                    is_inside[edge['to_id']] = []

                is_inside[edge['to_id']].append(edge['from_id'])
                inside_of[edge['from_id']] = edge['to_id']

            elif 'HOLDS' in edge['relation_type']:
                if edge['from_id'] == character['id']:
                    grabbed_ids.append(edge['to_id'])

        character_inside_ids = inside_of[character_id]
        room_id = character_inside_ids

        object_in_room_ids = is_inside[room_id]

        # Some object are not directly in room, but we want to add them
        curr_objects = list(object_in_room_ids)
        while len(curr_objects) > 0:
            objects_inside = []
            for curr_obj_id in curr_objects:
                new_inside = is_inside[curr_obj_id] if curr_obj_id in is_inside.keys() else [
                ]
                objects_inside += new_inside

            object_in_room_ids += list(objects_inside)
            new_curr_objects = list(objects_inside)
            if set(new_curr_objects) == set(curr_objects):
                break
            else:
                curr_objects = new_curr_objects

        self_rooms_ids = []
        for node in state["nodes"]:
            if node["category"] == "Rooms":
                self_rooms_ids.append(node['id'])

        # Only objects that are inside the room and not inside something closed
        # TODO: this can be probably speed up if we can ensure that all objects are either closed or open
        def object_hidden(ido): return inside_of[ido] not in self_rooms_ids and 'OPEN' not in id2node[inside_of[ido]][
            'states']
        observable_object_ids = [object_id for object_id in object_in_room_ids if
                                 not object_hidden(object_id)] + self_rooms_ids
        observable_object_ids += grabbed_ids

        partilly_observable_state = {
            "edges": [edge for edge in state['edges'] if
                      edge['from_id'] in observable_object_ids and edge['to_id'] in observable_object_ids],
            "nodes": [id2node[id_node] for id_node in observable_object_ids]
        }

        return partilly_observable_state

    def curate_dataset(self, images, labels_matrix, legal_range, orig_bbox, M=2):
        images_list = []
        labels_onehot_list = []
        idx_list = []
        bbox_list = []
        for start_idx, end_idx in legal_range:
            idx = (end_idx + start_idx) // 2
            idx_list.append(idx)
            label_one_hot = labels_matrix[idx]
            image_start_idx = max(0, (idx - M // 2))
            image_start = images[image_start_idx: image_start_idx+M]
            images_list.append(image_start)
            labels_onehot_list.append(label_one_hot)
        images_list.append(images[end_idx-M:end_idx])
        return images_list, labels_onehot_list, bbox_list, idx_list

    def sample_single(self, index):
        """
        Choose and Load frames per video
        :param index:
        :return:
        """
        box_categories = torch.ones(
            (self.coord_nr_frames, self.num_boxes)) * 80.

        if self.args.dataset == 'crosstask':
            folder_id = self.vid_names[index]
            box_tensors = torch.zeros(
                (self.coord_nr_frames, self.num_boxes, 4), dtype=torch.float32)  # (cx, cy, w, h)
            roi_feat_tensors = torch.zeros(
                (self.coord_nr_frames, self.num_boxes, 256, 7, 7), dtype=torch.float32)  # (cx, cy, w, h)
            frames = []
            labels = []
            orig_bbox = []
            images = np.load(os.path.join(self.features_path,
                                          folder_id['vid']+'.npy'))[:, :1024]  # (179, 3200)
            cnst_path = os.path.join(
                self.constraints_path, folder_id['task'] + '_' + folder_id['vid'] + '.csv')
            labels_matrix, legal_range = read_assignment(
                images.shape[0], folder_id['task'], self.n_steps, cnst_path)
            legal_range = [(start_idx, end_idx) for (
                start_idx, end_idx) in legal_range if end_idx < images.shape[0]+1]
            # print('len(legal_range): ', len(legal_range))
            if self.args.model_type == 'woT':
                images, labels_matrix, _, idx_list = self.curate_dataset(
                    images, labels_matrix, legal_range, orig_bbox, M=self.M)
                if len(labels_matrix) > self.args.max_traj_len:
                    idx = np.random.randint(
                        0, len(labels_matrix) - self.args.max_traj_len)
                else:
                    idx = 0
                frames = []
                for i in range(self.args.max_traj_len):
                    frames.extend(
                        images[min(idx + i, len(images) - 1)])  # goal
                frames = torch.tensor(frames)

                labels = []
                if idx - 1 < 0:
                    labels.append([0])
                else:
                    label = labels_matrix[idx - 1]
                    ind = np.unravel_index(
                        np.argmax(label, axis=-1), label.shape)
                    labels.append(ind)
                for i in range(self.args.max_traj_len):
                    if idx + i < len(labels_matrix):
                        label = labels_matrix[idx+i]
                        # print('label: ', label)
                        ind = np.unravel_index(
                            np.argmax(label, axis=-1), label.shape)
                        # print('ind: ', ind)
                        labels.append(ind)
                    else:
                        labels.append([0])
                labels_tensor = torch.tensor(labels, dtype=torch.float32)
            else:
                try:
                    legal_range_idx = np.random.randint(0, len(legal_range))
                    start_range_idx = legal_range[legal_range_idx]
                    start_idx = np.random.randint(
                        start_range_idx[0], min(start_range_idx[1], len(orig_bbox)))
                    label = labels_matrix[start_idx]  # labels:  (2, 133,)
                except:
                    start_idx = 0
                    label = labels_matrix[start_idx]  # labels:  (2, 133,)

                ind = np.unravel_index(np.argmax(label, axis=-1), label.shape)
                labels.append(ind)
                if start_idx+1 < len(labels_matrix):
                    label = labels_matrix[start_idx+1]
                    ind = np.unravel_index(
                        np.argmax(label, axis=-1), label.shape)
                    labels.append(ind)
                else:
                    labels.append([0])

                labels_tensor = torch.tensor(labels, dtype=torch.float32)

                # print(start_idx)
                img_counter = 0
                for img_idx in range(start_idx, images.shape[0]):
                    img_counter += 1
                    if img_counter == (self.coord_nr_frames - 1):
                        break
                frames.append(images[start_idx])
                if start_idx+1 < len(images):
                    frames.append(images[start_idx+1])
                else:
                    frames.append(np.zeros_like(images[start_idx]))
                frames = torch.tensor(frames)

            if self.args.random_coord:
                torch.manual_seed(index)
                box_tensors = torch.rand(
                    (self.coord_nr_frames, self.num_boxes, 4), dtype=torch.float32) * 224  # (cx, cy, w, h)
        return frames, frames, roi_feat_tensors, box_categories, labels_tensor

    def sample_single_actionet(self, index):
        """
        Choose and Load frames per video
        :param index:
        :return:
        """

        trajectory = self.trajectory[index]
        trajectory_len = len(trajectory)
        frames, box_tensors, roi_feat_tensors, box_categories, labels_tensor = [], [], [], [], []
        if trajectory_len > self.args.max_traj_len:
            start_idx = np.random.randint(
                0, trajectory_len - self.args.max_traj_len)
            for i in range(start_idx, start_idx + self.args.max_traj_len):
                data = trajectory[i][1]
                frame = Image.fromarray(data, 'RGB')
                frames.append(frame)
                labels_tensor.append(trajectory[i][0])
                box_tensors.append(0.)
                roi_feat_tensors.append(0.)
                box_categories.append(0)
        else:
            for i in range(0, trajectory_len):
                data = trajectory[i][1]
                frame = Image.fromarray(data, 'RGB')
                frames.append(frame)
                labels_tensor.append(trajectory[i][0])
                box_tensors.append(0.)
                roi_feat_tensors.append(0.)
                box_categories.append(0)
            for i in range(trajectory_len, self.args.max_traj_len):
                data = trajectory[-1][1]
                frame = Image.fromarray(data, 'RGB')
                frames.append(frame)
                labels_tensor.append(trajectory[-1][0])
                box_tensors.append(0.)
                roi_feat_tensors.append(0.)
                box_categories.append(0)
        box_tensors = torch.tensor(box_tensors)
        roi_feat_tensors = torch.tensor(roi_feat_tensors)
        box_categories = torch.tensor(box_categories)
        labels_tensor = torch.tensor(labels_tensor)

        return frames, frames, roi_feat_tensors, box_categories, labels_tensor

    def __getitem__(self, index):
        if self.args.dataset == 'crosstask':
            frames, box_tensors, roi_feat_tensors, box_categories, labels = self.sample_single(
                index)
            if self.args.dataset == 'crosstask':
                if self.args.model_type == 'model_T':
                    global_img_tensors = frames[1:2]  # torch.Size([2, 3200])
                    box_tensors = box_tensors[1:2]
                    roi_feat_tensors = roi_feat_tensors[1:2]
                    box_categories = box_categories[1:2]
                else:
                    global_img_tensors = frames  # torch.Size([2, 3200])
        elif self.args.dataset == 'actionet':
            frames, box_tensors, roi_feat_tensors, box_categories, labels = self.sample_single_actionet(
                index)
            frames = self.transforms(frames)
            global_img_tensors = frames.permute(
                1, 0, 2, 3)  # torch.Size([3, 2, 224, 224])
            box_tensors = box_tensors
            roi_feat_tensors = roi_feat_tensors
            box_categories = box_categories
            labels = labels
        else:
            raise NotImplementedError(self.args.dataset)

        return global_img_tensors, box_tensors, roi_feat_tensors, box_categories, labels

    def __len__(self):
        if self.args.dataset != 'actionet':
            return min(len(self.json_data), len(self.frame_cnts))
        else:
            return len(self.trajectory)

    def unnormalize(self, img, divisor=255):
        """
        The inverse operation of normalization
        Both the input & the output are in the format of BxCxHxW
        """
        for c in range(len(self.img_mean)):
            img[:, c, :, :].mul_(self.img_std[c]).add_(self.img_mean[c])

        return img / divisor

    def img2np(self, img):
        """
        Convert images in torch tensors of BxCxTxHxW [float32] to a numpy array of BxHxWxC [0-255, uint8]
        Take the first frame along temporal dimension
        if C == 1, that dimension is removed
        """
        img = self.unnormalize(img[:, :, 0, :, :], divisor=1).to(
            torch.uint8).permute(0, 2, 3, 1)
        if img.shape[3] == 1:
            img = img.squeeze(3)
        return img.cpu().numpy()

    def _process_instruction(self, ins):
        if self.inst_dict is not None:
            instruction, length = self.inst_dict.parse(ins, True)
            inst_idx = self.inst_dict.get_inst_idx(ins)
            return np.array(instruction, dtype=np.int64), length, inst_idx
        else:
            assert False

    def load_inst_dict(self, inst_dict_path):
        print('loading cmd dict from: ', inst_dict_path)
        if inst_dict_path is None or inst_dict_path == '':
            return None
        inst_dict = pickle.load(open(inst_dict_path, 'rb'))
        inst_dict.set_max_sentence_length(self.max_sentence_length)
        return inst_dict
