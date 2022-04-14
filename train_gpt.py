import argparse
import os
import shutil
import time
import numpy as np
from collections import OrderedDict
import torch
import torch.backends.cudnn as cudnn
from callbacks import AverageMeter, Logger
from data_utils.data_loader_frames import VideoFolder
from utils import save_results

MAX_TRAJ_LEN = 3
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='PyTorch Smth-Else')

# Path related arguments

parser.add_argument('--model',
                    default='coord')
parser.add_argument('--root_frames', default='', type=str, help='path to the folder with frames')
parser.add_argument('--json_data_train', default='', type=str, help='path to the json file with train video meta data')
parser.add_argument('--json_data_val', default='', type=str, help='path to the json file with validation video meta data')
parser.add_argument('--json_file_labels', default='', type=str, help='path to the json file with ground truth labels')
parser.add_argument('--img_feature_dim', default=256, type=int, metavar='N',
                    help='intermediate feature dimension for image-based features')
parser.add_argument('--coord_feature_dim', default=128, type=int, metavar='N',
                    help='intermediate feature dimension for coord-based features')
parser.add_argument('--clip_gradient', '-cg', default=5, type=float,
                    metavar='W', help='gradient norm clipping (default: 5)')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--size', default=224, type=int, metavar='N',
                    help='primary image input size')
parser.add_argument('--start_epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', '-b', default=72, type=int,
                    metavar='N', help='mini-batch size (default: 72)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[24, 35, 45], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=0.0001, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--print_freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--log_freq', '-l', default=10, type=int,
                    metavar='N', help='frequency to write in tensorboard (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--num_classes', default=192, type=int,
                    help='num of class in the model')
parser.add_argument('--num_boxes', default=4, type=int,
                    help='num of boxes for each image')
parser.add_argument('--num_frames', default=4, type=int,
                    help='num of frames for the model')
parser.add_argument('--action_feature_dim', default=8, type=int,
                    help='action feature dim')
parser.add_argument('--dataset', default='crosstask',
                    help='which dataset to train')
parser.add_argument('--logdir', default='./logs',
                    help='folder to output tensorboard logs')
parser.add_argument('--logname', default='exp',
                    help='name of the experiment for checkpoints and logs')
parser.add_argument('--ckpt', default='./ckpt',
                    help='folder to output checkpoints')
parser.add_argument('--fine_tune', help='path with ckpt to restore')
parser.add_argument('--tracked_boxes', type=str, help='choose tracked boxes')
parser.add_argument('--shot', default=5)
parser.add_argument('--restore_i3d')
parser.add_argument('--restore_custom')

parser.add_argument('--lang_model', default='', type=str, metavar='LANG',
                    help='language model (default: generative)')
parser.add_argument('--dataset_mode', default='proc_plan', type=str, metavar='DATA',
                    help='dataset mode (default: '')')
parser.add_argument('--model_type', default='model_T', type=str, metavar='MODELT',
                    help='forward dynamics model (model_T) or conjugate dynamics model (model_P)')
parser.add_argument('--max_sentence_len', default=3, type=int, metavar='MAXMESS',
                    help='max message length (default: 5)')
parser.add_argument('--max_traj_len', default=7, type=int, metavar='MAXTRAJ',
                    help='max trajectory length (default: 54)')
parser.add_argument('--beam_width', default=2, type=int, metavar='BEAMWID',
                    help='beam widths (default: 2)')
parser.add_argument('--roi_feature', type=str2bool,
                        default=True,
                        help='Using RoIAlign Feature')
parser.add_argument('--random_coord', type=str2bool,
                        default=False,
                        help='Use random coord')
parser.add_argument('--use_rnn', type=str2bool,
                        default=False,
                        help='Use RNN')
# GPT
parser.add_argument('--search_method', default='beam', type=str, metavar='SEAR',
                    help='search method for GPT: None (default: beam)')
parser.add_argument('--gpt_repr', default='all', type=str, metavar='SEAR',
                    help='representation method for GPT: all / start_goal / path (default: beam)')
parser.add_argument('--generation_method', default='autoregression', type=str, metavar='REGR',
                    help='generation method for GPT: autoregression / non-autoregression (default: autoregression)')
parser.add_argument('--sample_eval_gpt', type=str2bool,
                        default=True,
                        help='Evaluate gpt using sample, True for sample (multinomial), False for topk (default: True')
parser.add_argument('--pred_state_action', type=str2bool,
                        default=True,
                        help='Use GPT to predicate state-action')
parser.add_argument('--sa_type', default='feature_concat', type=str, metavar='SA',
                    help='state-action gpt type: temporal_concat (s-a-s-a) / feature_concat (s|a-s|a) (default: feature_concat)')

best_loss = 1000000
best_top1_acc = -np.inf
best_inst_acc = -np.inf
best_success_rate = -np.inf
best_miou = -np.inf

def eval():
    global args, best_loss, best_top1_acc, best_inst_acc, best_success_rate, best_miou
    args = parser.parse_args()
    # create model
    if args.dataset == 'crosstask':
        args.root_frames = '../../crosstask/crosstask_features'
        args.max_traj_len = MAX_TRAJ_LEN
        args.num_classes = 133  # max_n_step = 11, sum 133
        args.model_type = 'woT'  # 'model_T'
        args.roi_feature = False
        # args.dataset_mode = 'proc_plan'
        args.num_frames = 8
        args.model = 'global_gpt'
    elif args.dataset == 'actionet':
        args.root_frames = '../../actionet/actionet_dataset.pkl'
        args.max_traj_len = MAX_TRAJ_LEN
        args.num_classes = 34  # max_n_step = 11, sum 133
        args.model_type = 'woT'  # 'model_T'
        args.roi_feature = False
        # args.dataset_mode = 'proc_plan'
        args.num_frames = 8
        args.model = 'global_gpt'
    print(args)

    from model.model_lib import BC_MODEL as VideoModel
    model = VideoModel(args)

    args.resume = "ckpt/exp_20201226182330_crosstask_global_gpt_30_8_best.pth.tar"
    # optionally resume from a checkpoint
    assert os.path.isfile(args.resume), "No checkpoint found at '{}'".format(args.resume)
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    if args.start_epoch is None:
        args.start_epoch = checkpoint['epoch']
    best_loss = checkpoint['best_loss']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {}, top1_acc {})"
          .format(args.resume, checkpoint['epoch'], checkpoint['best_top1_acc']))

    if args.start_epoch is None:
        args.start_epoch = 0

    model = model.cuda()
    cudnn.benchmark = True

    dataset_val = VideoFolder(root=args.root_frames,
                              num_boxes=args.num_boxes,
                              file_input=args.json_data_val,
                              file_labels=args.json_file_labels,
                              frames_duration=args.num_frames,
                              args=args,
                              is_val=True,
                              if_augment=True,
                              model=args.model,
                              max_sentence_length=args.max_sentence_len,
                              max_traj_len=args.max_traj_len
                              )

    val_loader = torch.utils.data.DataLoader(
        dataset_val, drop_last=True,
        batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    criterion = torch.nn.CrossEntropyLoss()
    loss, top1_acc, inst_acc, success_rate = validate(val_loader, model, criterion)


def main():
    global args, best_loss, best_top1_acc, best_inst_acc, best_success_rate, best_miou
    args = parser.parse_args()
    # create model
    if args.dataset == 'crosstask':
        args.root_frames = '../../crosstask/crosstask_features'
        args.max_traj_len = MAX_TRAJ_LEN
        args.num_classes = 133  # max_n_step = 11, sum 133
        args.model_type = 'woT'  # 'model_T'
        args.roi_feature = False
        # args.dataset_mode = 'proc_plan'
        args.num_frames = 8
        args.model = 'global_gpt'
    elif args.dataset == 'actionet':
        args.root_frames = '../../actionet/actionet_dataset2.pkl'
        args.max_traj_len = MAX_TRAJ_LEN
        args.num_classes = 34  # max_n_step = 11, sum 133
        args.model_type = 'woT'  # 'model_T'
        args.roi_feature = False
        # args.dataset_mode = 'proc_plan'
        args.num_frames = 8
        args.model = 'global_gpt'
    print(args)


    from model.model_lib import BC_MODEL as VideoModel
    model = VideoModel(args)

    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume), "No checkpoint found at '{}'".format(args.resume)
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        if args.start_epoch is None:
            args.start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}, top1_acc {})"
              .format(args.resume, checkpoint['epoch'], checkpoint['best_top1_acc']))

    if args.start_epoch is None:
        args.start_epoch = 0

    # model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()
    cudnn.benchmark = True

    # create training and validation dataset
    dataset_train = VideoFolder(root=args.root_frames,
                                num_boxes=args.num_boxes,
                                file_input=args.json_data_train,
                                file_labels=args.json_file_labels,
                                frames_duration=args.num_frames,
                                args=args,
                                is_val=False,
                                if_augment=True,
                                model=args.model,
                                max_sentence_length=args.max_sentence_len,
                                max_traj_len=args.max_traj_len
                                )
    dataset_val = VideoFolder(root=args.root_frames,
                              num_boxes=args.num_boxes,
                              file_input=args.json_data_val,
                              file_labels=args.json_file_labels,
                              frames_duration=args.num_frames,
                              args=args,
                              is_val=True,
                              if_augment=True,
                              model=args.model,
                              max_sentence_length=args.max_sentence_len,
                              max_traj_len=args.max_traj_len
                              )

    # create training and validation loader
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, drop_last=True,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val, drop_last=True,
        batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    optimizer = torch.optim.SGD(model.parameters(), momentum=args.momentum,
                                lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    if args.evaluate:
        validate(val_loader, model, criterion, class_to_idx=dataset_val.classes_dict)
        return

    # training, start a logger
    time_pre = time.strftime("%Y%m%d%H%M%S", time.localtime())
    args.logname = args.logname + '_' + time_pre + '_' + args.dataset + '_' + args.model + '_' + str(args.num_boxes) + '_' + str(args.num_frames)
    tb_logdir = os.path.join(args.logdir, args.logname)
    if not (os.path.exists(tb_logdir)):
        os.makedirs(tb_logdir)
    tb_logger = Logger(tb_logdir)
    tb_logger.log_info(args)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_steps)

        # train for one epoch
        train(train_loader, model, optimizer, epoch, criterion, tb_logger)

        # evaluate on validation set
        if (not args.fine_tune) or (epoch + 1) % 10 == 0:
            loss, top1_acc, inst_acc, success_rate, miou = validate(val_loader, model, criterion, epoch=epoch, tb_logger=tb_logger)
        else:
            loss = 100

        # remember best loss and save checkpoint
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        is_best_top1_acc = top1_acc > best_top1_acc
        best_top1_acc = max(top1_acc, best_top1_acc)
        best_inst_acc = max(inst_acc, best_inst_acc)
        best_success_rate = max(success_rate, best_success_rate)
        best_miou = max(miou, best_miou)
        print('Epoch {}: search_method: {} / gpt_repr: {} / sample_eval_gpt: {} - Best evaluation top1 accuracy: {:.2f}, '
              'instruction accuracy: {:.2f}, success rate: {:.2f}, miou: {:.2f}'
              .format(epoch, args.search_method, args.gpt_repr, args.sample_eval_gpt, best_top1_acc, best_inst_acc, best_success_rate, best_miou))
        if not os.path.exists(args.ckpt):
            os.makedirs(args.ckpt)
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'best_top1_acc': best_top1_acc,
            },
            is_best_top1_acc,  # is_best,
            os.path.join(args.ckpt, '{}'.format(args.logname)))


def train(train_loader, model, optimizer, epoch, criterion, tb_logger=None):
    global args, best_top1_acc
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    state_losses = AverageMeter()
    acc_top1 = AverageMeter()
    acc_top5 = AverageMeter()
    instruction_accuracy_meter = AverageMeter()
    trajectory_success_rate_meter = AverageMeter()
    MIoU_meter = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    all_pred_inst_list = []
    all_target_inst_list = []
    for i, (global_img_tensors, box_tensors, roi_feat_tensors, box_categories, video_label) in enumerate(train_loader):
        model.zero_grad()
        # measure data loading time
        data_time.update(time.time() - end)

        global_img_tensors = global_img_tensors.cuda()
        video_label = video_label.cuda()
        # compute output
        if args.dataset == 'crosstask':
            last_action = video_label[:, 0]
            if args.model_type in ['woT']:
                if args.max_traj_len in [3, 4]:
                    video_label = video_label[:, :args.max_traj_len]
                else:
                    video_label = video_label[:, :-1]
            else:
                video_label = video_label[:, 1]
            output, state_loss = model(global_img_tensors, box_categories, box_tensors, video_label, last_action=last_action, roi_feat=roi_feat_tensors)
        elif args.dataset == 'actionet':
            # global_img_tensors = global_img_tensors[:, 0]
            last_action = video_label[:, 0]
            if args.model_type in ['woT']:
                if args.max_traj_len in [3, 4]:
                    video_label = video_label[:, :args.max_traj_len]
                else:
                    video_label = video_label[:, :-1]
            else:
                video_label = video_label[:, 1]
            output, state_loss = model(global_img_tensors, box_categories, box_tensors, video_label, last_action=last_action,
                           roi_feat=roi_feat_tensors)
        else:
            output = model(global_img_tensors, box_categories, box_tensors, video_label,
                           roi_feat=roi_feat_tensors)
        output_reshaped = output.contiguous().view(-1, output.shape[-1])
        video_label_reshaped = video_label.contiguous().view(-1)
        loss = criterion(output_reshaped, video_label_reshaped.long().cuda())
        if args.pred_state_action:
            loss = loss + state_loss
        # print('output_reshaped.shape, video_label_reshaped.shape: ', output_reshaped.shape, video_label_reshaped.shape)
        (acc1, acc5), instruction_accuracy, trajectory_success_rate, MIoU, (new_pred_inst_list, new_target_inst_list) = \
            accuracy(output_reshaped.cpu(), video_label_reshaped.cpu(), topk=(1, 5), max_traj_len=args.max_traj_len)

        all_pred_inst_list.extend(new_pred_inst_list)
        all_target_inst_list.extend(new_target_inst_list)
        # measure accuracy and record loss
        losses.update(loss.item(), global_img_tensors.size(0))
        state_losses.update(state_loss.item(), global_img_tensors.size(0))
        acc_top1.update(acc1.item(), global_img_tensors.size(0))
        acc_top5.update(acc5.item(), global_img_tensors.size(0))
        instruction_accuracy_meter.update(instruction_accuracy.item(),  global_img_tensors.size(0))
        trajectory_success_rate_meter.update(trajectory_success_rate.item(), global_img_tensors.size(0))
        MIoU_meter.update(MIoU, global_img_tensors.size(0) // args.max_traj_len)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if args.clip_gradient is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'State Loss {state_loss.val:.4f} ({state_loss.avg:.4f})\t'
                  'Token Acc1 {acc_top1.val:.2f} ({acc_top1.avg:.2f})\t'
                  'Token Acc5 {acc_top5.val:.2f} ({acc_top5.avg:.2f})\t'
                  'Instruction Acc {instruction_accuracy_meter.val:.2f} ({instruction_accuracy_meter.avg:.2f})\t'
                  'Trajectory Success Rate {trajectory_success_rate_meter.val:.2f} ({trajectory_success_rate_meter.avg:.2f})\t'
                  'MIoU {MIoU_meter.val:.2f} ({MIoU_meter.avg:.2f})\t'
                .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, state_loss=state_losses,
                   acc_top1=acc_top1, acc_top5=acc_top5,
                   instruction_accuracy_meter=instruction_accuracy_meter,
                   trajectory_success_rate_meter=trajectory_success_rate_meter,
                   MIoU_meter=MIoU_meter))

        # log training data into tensorboard
        if tb_logger is not None and i % args.log_freq == 0:
            logs = OrderedDict()
            logs['Train/IterLoss'] = losses.val
            logs['Train/Token_Acc@1'] = acc_top1.val
            logs['Train/Token_Acc@5'] = acc_top5.val
            logs['Train/Instruction_Acc@1'] = instruction_accuracy_meter.val
            logs['Train/Traj_Success_Rate'] = trajectory_success_rate_meter.val
            logs['Train/MIoU'] = MIoU_meter.val
            # how many iterations we have trained
            iter_count = epoch * len(train_loader) + i
            for key, value in logs.items():
                tb_logger.log_scalar(value, key, iter_count)

            tb_logger.flush()


def validate(val_loader, model, criterion, epoch=None, tb_logger=None, class_to_idx=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc_top1 = AverageMeter()
    acc_top5 = AverageMeter()
    instruction_accuracy_meter = AverageMeter()
    trajectory_success_rate_meter = AverageMeter()
    MIoU_meter = AverageMeter()

    acc_top1_3 = AverageMeter()
    acc_top5_3 = AverageMeter()
    instruction_accuracy_meter_3 = AverageMeter()
    trajectory_success_rate_meter_3 = AverageMeter()
    MIoU_meter_3 = AverageMeter()
    effect_domain_prior = AverageMeter()

    logits_matrix = []
    targets_list = []
    # switch to evaluate mode
    # model.eval()

    end = time.time()
    all_pred_inst_list = []
    all_target_inst_list = []
    for i, (global_img_tensors, box_tensors, roi_feat_tensors, box_categories, video_label) in enumerate(val_loader):
        # compute output
        global_img_tensors = global_img_tensors.cuda()
        video_label = video_label.cuda()
        with torch.no_grad():
            if args.dataset == 'crosstask':
                last_action = video_label[:, 0]
                if args.model_type in ['woT']:
                    if args.max_traj_len in [3, 4]:
                        video_label = video_label[:, :args.max_traj_len]
                    else:
                        video_label = video_label[:, :-1]
                else:
                    video_label = video_label[:, 1]
                if args.generation_method == 'autoregression':
                    output, domain_prior_list = model.base.model_get_action(global_img_tensors, box_categories, box_tensors, video_label, last_action=last_action, roi_feat=roi_feat_tensors, is_inference=True)
                elif args.generation_method == 'non-autoregression':
                    output = model(global_img_tensors, box_categories, box_tensors, video_label,
                                                 last_action=last_action, roi_feat=roi_feat_tensors, is_inference=True)
            elif args.dataset == 'actionet':
                last_action = video_label[:, 0]
                if args.model_type in ['woT']:
                    if args.max_traj_len in [3, 4]:
                        video_label = video_label[:, :args.max_traj_len]
                    else:
                        video_label = video_label[:, :-1]
                else:
                    video_label = video_label[:, 1]
                if args.generation_method == 'autoregression':
                    output, domain_prior_list = model.base.model_get_action(global_img_tensors, box_categories,
                                                                            box_tensors, video_label,
                                                                            last_action=last_action,
                                                                            roi_feat=roi_feat_tensors,
                                                                            is_inference=True)
                elif args.generation_method == 'non-autoregression':
                    output = model(global_img_tensors, box_categories, box_tensors, video_label,
                                   last_action=last_action, roi_feat=roi_feat_tensors, is_inference=True)
            else:
                output = model(global_img_tensors, box_categories, box_tensors, video_label, roi_feat=roi_feat_tensors, is_inference=True)

            output_reshaped = output.contiguous().view(-1, output.shape[-1])
            video_label_reshaped = video_label.contiguous().view(-1)
            loss = criterion(output_reshaped, video_label_reshaped.long().cuda())

            (acc1, acc5), instruction_accuracy, trajectory_success_rate, MIoU, (new_pred_inst_list, new_target_inst_list) = accuracy(output_reshaped.cpu(), video_label_reshaped.cpu(), topk=(1, 5), max_traj_len=args.max_traj_len)
            (acc1_3, acc5_3), instruction_accuracy_3, trajectory_success_rate_3, MIoU_3, (_, _) = accuracy(output[:, :3].contiguous().view(-1, output.shape[-1]).cpu(),
                                                                                         video_label[:, :3].contiguous().view(-1).cpu(),
                                                                                         topk=(1, 5), max_traj_len=3)
            all_pred_inst_list.extend(new_pred_inst_list)
            all_target_inst_list.extend(new_target_inst_list)
            if args.evaluate:
                logits_matrix.append(output.cpu().data.numpy())
                targets_list.append(video_label.cpu().numpy())

        # measure accuracy and record loss
        losses.update(loss.item(), global_img_tensors.size(0))
        acc_top1.update(acc1.item(), global_img_tensors.size(0))
        acc_top5.update(acc5.item(), global_img_tensors.size(0))
        instruction_accuracy_meter.update(instruction_accuracy.item(), global_img_tensors.size(0))
        trajectory_success_rate_meter.update(trajectory_success_rate.item(), global_img_tensors.size(0))
        MIoU_meter.update(MIoU, global_img_tensors.size(0) // args.max_traj_len)
        effect_domain_prior.update(sum(domain_prior_list), global_img_tensors.size(0))

        acc_top1_3.update(acc1_3.item(), global_img_tensors.size(0))
        acc_top5_3.update(acc5_3.item(), global_img_tensors.size(0))
        instruction_accuracy_meter_3.update(instruction_accuracy_3.item(), global_img_tensors.size(0))
        trajectory_success_rate_meter_3.update(trajectory_success_rate_3.item(), global_img_tensors.size(0))
        MIoU_meter_3.update(MIoU_3, global_img_tensors.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or i + 1 == len(val_loader):
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Token Acc1 {acc_top1.val:.2f} ({acc_top1.avg:.2f})\t'
                  'Token Acc5 {acc_top5.val:.2f} ({acc_top5.avg:.2f})\t'
                  'Instruction Acc {instruction_accuracy_meter.val:.2f} ({instruction_accuracy_meter.avg:.2f})\t'
                  'Trajectory Success Rate {trajectory_success_rate_meter.val:.2f} ({trajectory_success_rate_meter.avg:.2f})\t'
                  'MIoU {MIoU_meter.val:.1f} ({MIoU_meter.avg:.2f})\t'
                  'Token Acc1_3 {acc_top1_3.val:.2f} ({acc_top1_3.avg:.2f})\t'
                  'Token Acc5_3 {acc_top5_3.val:.2f} ({acc_top5_3.avg:.2f})\t'
                  'Instruction Acc_3 {instruction_accuracy_meter_3.val:.2f} ({instruction_accuracy_meter_3.avg:.2f})\t'
                  'Trajectory Success Rate_3 {trajectory_success_rate_meter_3.val:.2f} ({trajectory_success_rate_meter_3.avg:.2f})\t'
                  'MIoU_3 {MIoU_meter.val:.1f} ({MIoU_meter_3.avg:.2f})\t'
                  'effect_domain_prior {effect_domain_prior.val:.1f} ({effect_domain_prior.avg:.2f})\t'
                .format(
                i, len(val_loader), batch_time=batch_time, loss=losses, acc_top1=acc_top1, acc_top5=acc_top5,
                instruction_accuracy_meter=instruction_accuracy_meter, trajectory_success_rate_meter=trajectory_success_rate_meter,
                MIoU_meter=MIoU_meter,
                acc_top1_3=acc_top1_3, acc_top5_3=acc_top5_3,
                instruction_accuracy_meter_3=instruction_accuracy_meter_3, trajectory_success_rate_meter_3=trajectory_success_rate_meter_3,
                MIoU_meter_3=MIoU_meter_3, effect_domain_prior=effect_domain_prior
            ))

    if args.evaluate:
        logits_matrix = np.concatenate(logits_matrix)
        targets_list = np.concatenate(targets_list)
        save_results(logits_matrix, targets_list, class_to_idx, args)

    if epoch is not None and tb_logger is not None:
        logs = OrderedDict()
        logs['Val/EpochLoss'] = losses.avg
        logs['Val/EpochAcc@1'] = acc_top1.avg
        logs['Val/EpochAcc@5'] = acc_top5.avg
        logs['Val/Instruction_Acc@1'] = instruction_accuracy_meter.val
        logs['Val/Traj_Success_Rate'] = trajectory_success_rate_meter.val
        logs['Val/MIoU'] = MIoU_meter.val
        # how many iterations we have trained
        for key, value in logs.items():
            tb_logger.log_scalar(value, key, epoch + 1)

        tb_logger.flush()

    return losses.avg, acc_top1.avg, instruction_accuracy_meter.avg, trajectory_success_rate_meter.avg, MIoU_meter.avg


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename + '_latest.pth.tar')
    if is_best:
        shutil.copyfile(filename + '_latest.pth.tar', filename + '_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,), max_traj_len=0):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        # Token Accuracy
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # [5, 1620]

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        correct_1 = correct[:1]  # .view(-1, max_traj_len) # (bz, 1)
        # Instruction Accuracy
        _, new_pred = output.topk(1, 1, True, True)
        instruction_correct = torch.all(correct_1.view(correct_1.shape[1], -1), dim=1)
        instruction_accuracy = instruction_correct.sum() * 100.0 / instruction_correct.shape[0]

        # Success Rate
        trajectory_success = torch.all(instruction_correct.view(correct_1.shape[1] // max_traj_len, -1), dim=1)
        trajectory_success_rate = trajectory_success.sum() * 100.0 / trajectory_success.shape[0]

        # MIoU
        _, pred_token = output.topk(1, 1, True, True)
        pred_inst = pred_token.view(correct_1.shape[1], -1)
        pred_inst_set = set()
        target_inst = target.view(correct_1.shape[1], -1)
        target_inst_set = set()
        for i in range(pred_inst.shape[0]):
            # print(pred_inst[i], target_inst[i])
            pred_inst_set.add(tuple(pred_inst[i].tolist()))
            target_inst_set.add(tuple(target_inst[i].tolist()))
        MIoU = 100.0 * len(pred_inst_set.intersection(target_inst_set)) / len(pred_inst_set.union(target_inst_set))
        return res, instruction_accuracy, trajectory_success_rate, MIoU, ([1], [1])


if __name__ == '__main__':
    main()
    # eval()
