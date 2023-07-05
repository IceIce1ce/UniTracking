import os
import numpy as np
from skimage import io
import torch
import argparse
from Tracker.DeepSort.deep_sort import DeepSort
from Tracker.Sort.Sort import Sort
from Tracker.ByteTrack.byte_tracker import BYTETracker
from Tracker.SortOH.tracker import Sort_OH
from Tracker.SortOH import kalman_tracker
from Tracker.OCSort.ocsort import OCSort
from Tracker.DeepOCSort.ocsort import DeepOCSort
from Tracker.StrongSort.strong_sort import StrongSORT
from Tracker.BotSort.bot_sort import BoTSORT
from Tracker.motdt.motdt_tracker import Motdt

def xyxy2xywh(x):
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y

def parse_args():
    parser = argparse.ArgumentParser(description='SORT and its extension demo')
    parser.add_argument('--model', type=str, default='deepsort')
    parser.add_argument('--dataset', type=str, default='mot16')
    parser.add_argument('--track_thresh', type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument('--track_buffer', type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument('--match_thresh', type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument('--phase', type=str, default='train')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if args.dataset == 'mot16':
        if args.phase == 'train':
            # run last 05 just for botsort because different resolution
            seq = ['02', '04', '05', '09', '10', '11', '13']
            len_seq = [600, 1050, 837, 525, 654, 900, 750]
        elif args.phase == 'test':
            # run last 06 just for botsort because different resolution
            seq = ['01', '03', '06', '07', '08', '12', '14']
            len_seq = [450, 1500, 1194, 500, 625, 900, 750]
        else:
            print('Phase not found!')
            exit()
    elif args.dataset == 'mot20':
        if args.phase == 'train':
            # run 03, then 05 just for botsort because different resolution
            seq = ['01', '02', '03', '05']
            len_seq = [429, 2782, 2405, 3315]
        elif args.phase == 'test':
            # run 04, then 07 just for botsort because different resolution
            seq = ['04', '06', '07', '08']
            len_seq = [2080, 1008, 585, 806]
        else:
            print('Phase not found!')
            exit()
    else:
        print('Dataset not found!')
        exit()
    if args.model == 'deepsort':
        if not os.path.exists('output/{}/{}/{}'.format(args.model, args.dataset.upper(), args.phase)):
            os.makedirs('output/{}/{}/{}'.format(args.model, args.dataset.upper(), args.phase))
        mot_tracker = DeepSort(model_path='Tracker/DeepSort/deep/checkpoint/ckpt.t7', use_efficientdet=False)
        for i in range(len(seq)):
            print('Processing MOT-{}'.format(seq[i]))
            seq_dets_fn = "data/{}/{}/{}-{}/det/det.txt".format(args.dataset.upper(), args.phase, args.dataset.upper(), seq[i])
            seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
            with open('output/{}/{}/{}/{}-{}.txt'.format(args.model, args.dataset.upper(), args.phase, args.dataset.upper(), seq[i]),'w') as out_file:
                for frame in range(1, len_seq[i] + 1):
                    images_path = "data/{}/{}/{}-{}/img1".format(args.dataset.upper(), args.phase, args.dataset.upper(), seq[i])
                    image_name = os.path.join(images_path, "{0:0=6d}".format(int(frame)) + ".jpg")
                    img = io.imread(image_name)
                    dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
                    dets[:, 2:4] += dets[:, 0:2]
                    bbox_xywh = xyxy2xywh(torch.tensor(dets[:, 0:4])).cpu()
                    trackers = mot_tracker.update(bbox_xywh=bbox_xywh, confidences=torch.tensor(dets[:, 4:5]), ori_img=img)
                    for d in trackers:
                        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]), file=out_file)
    elif args.model == 'sort':
        if not os.path.exists('output/{}/{}/{}'.format(args.model, args.dataset.upper(), args.phase)):
            os.makedirs('output/{}/{}/{}'.format(args.model, args.dataset.upper(), args.phase))
        mot_tracker = Sort(max_age=1, min_hits=3, iou_threshold=0.3)
        for i in range(len(seq)):
            print('Processing MOT-{}'.format(seq[i]))
            seq_dets_fn = "data/{}/{}/{}-{}/det/det.txt".format(args.dataset.upper(), args.phase, args.dataset.upper(), seq[i])
            seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
            with open('output/{}/{}/{}/{}-{}.txt'.format(args.model, args.dataset.upper(), args.phase, args.dataset.upper(), seq[i]),'w') as out_file:
                for frame in range(1, len_seq[i] + 1):
                    dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
                    dets[:, 2:4] += dets[:, 0:2]
                    trackers = mot_tracker.update(dets)
                    for d in trackers:
                        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]), file=out_file)
    elif args.model == 'bytetrack':
        if not os.path.exists('output/{}/{}/{}'.format(args.model, args.dataset.upper(), args.phase)):
            os.makedirs('output/{}/{}/{}'.format(args.model, args.dataset.upper(), args.phase))
        mot_tracker = BYTETracker(args)
        for i in range(len(seq)):
            print('Processing MOT-{}'.format(seq[i]))
            seq_dets_fn = "data/{}/{}/{}-{}/det/det.txt".format(args.dataset.upper(), args.phase, args.dataset.upper(), seq[i])
            seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
            with open('output/{}/{}/{}/{}-{}.txt'.format(args.model, args.dataset.upper(), args.phase, args.dataset.upper(), seq[i]),'w') as out_file:
                for frame in range(1, len_seq[i] + 1):
                    dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
                    dets[:, 2:4] += dets[:, 0:2]
                    trackers = mot_tracker.update(dets)
                    for d in trackers:
                        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]), file=out_file)
    elif args.model == 'sortoh':
        if not os.path.exists('output/{}/{}/{}'.format(args.model, args.dataset.upper(), args.phase)):
            os.makedirs('output/{}/{}/{}'.format(args.model, args.dataset.upper(), args.phase))
        for i in range(len(seq)):
            kalman_tracker.KalmanBoxTracker.count = 0
            mot_tracker = Sort_OH()
            mot_tracker.conf_trgt = 0.35
            mot_tracker.conf_objt = 0.75
            print('Processing MOT-{}'.format(seq[i]))
            seq_dets_fn = "data/{}/{}/{}-{}/det/det.txt".format(args.dataset.upper(), args.phase, args.dataset.upper(), seq[i])
            seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
            if args.phase == 'train':
                seq_gts_fn = "data/{}/{}/{}-{}/gt/gt.txt".format(args.dataset.upper(), args.phase, args.dataset.upper(), seq[i])
                seq_gts = np.loadtxt(seq_gts_fn, delimiter=',')
            with open('output/{}/{}/{}/{}-{}.txt'.format(args.model, args.dataset.upper(), args.phase, args.dataset.upper(), seq[i]),'w') as out_file:
                for frame in range(1, len_seq[i] + 1):
                    dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
                    for d in reversed(range(len(dets))):
                        if dets[d, 4] < 0.3:
                            dets = np.delete(dets, d, 0) # remove dets with low confidence
                    dets[:, 2:4] += dets[:, 0:2]
                    gts = []
                    if args.phase == 'train':
                        gts = seq_gts[seq_gts[:, 0] == frame, 2:7]
                        gts[:, 2:4] += gts[:, 0:2]
                    trackers, unmatched_tracker, unmatched_gts = mot_tracker.update(dets, gts)
                    for d in trackers:
                        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]), file=out_file)
    elif args.model == 'ocsort':
        if not os.path.exists('output/{}/{}/{}'.format(args.model, args.dataset.upper(), args.phase)):
            os.makedirs('output/{}/{}/{}'.format(args.model, args.dataset.upper(), args.phase))
        mot_tracker = OCSort(asso_func='giou', delta_t=1, det_thresh=0, inertia=0.3941737016672115, iou_threshold=0.22136877277096445, max_age=50, min_hits=1, use_byte=False)
        for i in range(len(seq)):
            print('Processing MOT-{}'.format(seq[i]))
            seq_dets_fn = "data/{}/{}/{}-{}/det/det.txt".format(args.dataset.upper(), args.phase, args.dataset.upper(), seq[i])
            seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
            with open('output/{}/{}/{}/{}-{}.txt'.format(args.model, args.dataset.upper(), args.phase, args.dataset.upper(), seq[i]),'w') as out_file:
                for frame in range(1, len_seq[i] + 1):
                    dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
                    dets[:, 2:4] += dets[:, 0:2]
                    trackers = mot_tracker.update(dets)
                    for d in trackers:
                        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]), file=out_file)
    elif args.model == 'deepocsort':
        if not os.path.exists('output/{}/{}/{}'.format(args.model, args.dataset.upper(), args.phase)):
            os.makedirs('output/{}/{}/{}'.format(args.model, args.dataset.upper(), args.phase))
        mot_tracker = DeepOCSort(det_thresh=0, max_age=50, min_hits=1, iou_threshold=0.22136877277096445, delta_t=1, asso_func='giou', inertia=0.3941737016672115)
        for i in range(len(seq)):
            print('Processing MOT-{}'.format(seq[i]))
            seq_dets_fn = "data/{}/{}/{}-{}/det/det.txt".format(args.dataset.upper(), args.phase, args.dataset.upper(), seq[i])
            seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
            with open('output/{}/{}/{}/{}-{}.txt'.format(args.model, args.dataset.upper(), args.phase, args.dataset.upper(), seq[i]),'w') as out_file:
                for frame in range(1, len_seq[i] + 1):
                    images_path = "data/{}/{}/{}-{}/img1".format(args.dataset.upper(), args.phase, args.dataset.upper(), seq[i])
                    image_name = os.path.join(images_path, "{0:0=6d}".format(int(frame)) + ".jpg")
                    img = io.imread(image_name)
                    dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
                    dets[:, 2:4] += dets[:, 0:2]
                    trackers = mot_tracker.update(dets=dets, img_numpy=img)
                    for d in trackers:
                        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]), file=out_file)
    elif args.model == 'strongsort':
        if not os.path.exists('output/{}/{}/{}'.format(args.model, args.dataset.upper(), args.phase)):
            os.makedirs('output/{}/{}/{}'.format(args.model, args.dataset.upper(), args.phase))
        mot_tracker = StrongSORT(model_path='Tracker/DeepSort/deep/checkpoint/ckpt.t7', use_cuda=True, use_efficientdet=False)
        for i in range(len(seq)):
            print('Processing MOT-{}'.format(seq[i]))
            seq_dets_fn = "data/{}/{}/{}-{}/det/det.txt".format(args.dataset.upper(), args.phase, args.dataset.upper(), seq[i])
            seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
            with open('output/{}/{}/{}/{}-{}.txt'.format(args.model, args.dataset.upper(), args.phase, args.dataset.upper(), seq[i]),'w') as out_file:
                for frame in range(1, len_seq[i] + 1):
                    images_path = "data/{}/{}/{}-{}/img1".format(args.dataset.upper(), args.phase, args.dataset.upper(), seq[i])
                    image_name = os.path.join(images_path, "{0:0=6d}".format(int(frame)) + ".jpg")
                    img = io.imread(image_name)
                    dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
                    dets[:, 2:4] += dets[:, 0:2]
                    trackers = mot_tracker.update(dets=dets, ori_img=img)
                    for d in trackers:
                        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]), file=out_file)
    elif args.model == 'botsort':
        if not os.path.exists('output/{}/{}/{}'.format(args.model, args.dataset.upper(), args.phase)):
            os.makedirs('output/{}/{}/{}'.format(args.model, args.dataset.upper(), args.phase))
        mot_tracker = BoTSORT(model_path='Tracker/DeepSort/deep/checkpoint/ckpt.t7', use_cuda=True)
        for i in range(len(seq)):
            print('Processing MOT-{}'.format(seq[i]))
            seq_dets_fn = "data/{}/{}/{}-{}/det/det.txt".format(args.dataset.upper(), args.phase, args.dataset.upper(), seq[i])
            seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
            with open('output/{}/{}/{}/{}-{}.txt'.format(args.model, args.dataset.upper(), args.phase, args.dataset.upper(), seq[i]),'w') as out_file:
                for frame in range(1, len_seq[i] + 1):
                    images_path = "data/{}/{}/{}-{}/img1".format(args.dataset.upper(), args.phase, args.dataset.upper(), seq[i])
                    image_name = os.path.join(images_path, "{0:0=6d}".format(int(frame)) + ".jpg")
                    img = io.imread(image_name)
                    dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
                    dets[:, 2:4] += dets[:, 0:2]
                    trackers = mot_tracker.update(output_results=dets, img=img)
                    for d in trackers:
                        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]), file=out_file)
    elif args.model == 'motdt':
        if not os.path.exists('output/{}/{}/{}'.format(args.model, args.dataset.upper(), args.phase)):
            os.makedirs('output/{}/{}/{}'.format(args.model, args.dataset.upper(), args.phase))
        mot_tracker = Motdt(model_path='Tracker/DeepSort/deep/checkpoint/ckpt.t7', use_cuda=True, use_efficientdet=False)
        for i in range(len(seq)):
            print('Processing MOT-{}'.format(seq[i]))
            seq_dets_fn = "data/{}/{}/{}-{}/det/det.txt".format(args.dataset.upper(), args.phase, args.dataset.upper(), seq[i])
            seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
            with open('output/{}/{}/{}/{}-{}.txt'.format(args.model, args.dataset.upper(), args.phase, args.dataset.upper(), seq[i]),'w') as out_file:
                for frame in range(1, len_seq[i] + 1):
                    images_path = "data/{}/{}/{}-{}/img1".format(args.dataset.upper(), args.phase, args.dataset.upper(), seq[i])
                    image_name = os.path.join(images_path, "{0:0=6d}".format(int(frame)) + ".jpg")
                    img = io.imread(image_name)
                    dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
                    dets[:, 2:4] += dets[:, 0:2]
                    trackers = mot_tracker.update(dets=dets, ori_img=img)
                    for d in trackers:
                        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]), file=out_file)
    else:
        print('Model not found!')
        exit()

if __name__ == '__main__':
    main()