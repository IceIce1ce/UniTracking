from timeit import default_timer as timer
import torch
import cv2
import numpy as np
from torch.backends import cudnn
from Detector.Efficientdet.backbone import EfficientDetBackbone
from Detector.Efficientdet.efficientdet.utils import BBoxTransform, ClipBoxes
from Detector.Efficientdet.utils.utils import invert_affine, postprocess, preprocess_video, xyxy_to_xywh
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

def parse_args():
    parser = argparse.ArgumentParser(description='Tracking with efficientdet')
    parser.add_argument('--model', type=str, default='deepsort')
    parser.add_argument('--c', type=int, default=0)
    parser.add_argument('--video_src', type=str, default='demo.mp4')
    parser.add_argument('--video_output', type=str, default='output.mp4')
    parser.add_argument('--text_output', type=str, default='result.txt')
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument('--iou_threshold', type=float, default=0.5)
    ### args for bytetrack
    parser.add_argument('--dataset', type=str, default='mot16')
    parser.add_argument('--track_thresh', type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument('--track_buffer', type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument('--match_thresh', type=float, default=0.8, help="matching threshold for tracking")
    ###
    args = parser.parse_args()
    return args

class MOT(object):
    def __init__(self):
        args = parse_args()
        self.video_src = args.video_src
        self.video_output = args.video_output
        self.text_output = args.text_output
        self.coco_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                            'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                            'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
                            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                            'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                            'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
                            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                            'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
        self.compound_coef = args.c
        self.threshold = args.threshold
        self.iou_threshold = args.iou_threshold
        self.input_size = self.input_sizes[self.compound_coef]
        self.use_float16 = False
        cudnn.fastest = True
        cudnn.benchmark = True
        model = EfficientDetBackbone(compound_coef=self.compound_coef, num_classes=len(self.coco_classes))
        model.load_state_dict(torch.load(f'Detector/Efficientdet/weights/efficientdet-d{self.compound_coef}.pth'))
        model.requires_grad_(False)
        model.eval()
        self.detector = model.cuda()
        if self.use_float16:
            self.detector = model.half()
        self.selected_target = [self.coco_classes.index('person')]
        self.trackers = []
        for num in range(0, len(self.selected_target)):
            if args.model == 'deepsort':
                self.trackers.append(DeepSort(model_path='Tracker/DeepSort/deep/checkpoint/ckpt.t7', use_efficientdet=True))
            elif args.model == 'sort':
                self.trackers.append(Sort(max_age=1, min_hits=3, iou_threshold=0.3))
            elif args.model == 'bytetrack':
                self.trackers.append(BYTETracker(args))
            elif args.model == 'sortoh':
                kalman_tracker.KalmanBoxTracker.count = 0
                mot_tracker = Sort_OH()
                mot_tracker.conf_trgt = 0.35
                mot_tracker.conf_objt = 0.75
                self.trackers.append(mot_tracker)
            elif args.model == 'ocsort':
                self.trackers.append(OCSort(asso_func='giou', delta_t=1, det_thresh=0, inertia=0.3941737016672115, iou_threshold=0.22136877277096445, max_age=50, min_hits=1, use_byte=False))
            elif args.model == 'deepocsort':
                self.trackers.append(DeepOCSort(det_thresh=0, max_age=50, min_hits=1, iou_threshold=0.22136877277096445, delta_t=1, asso_func='giou', inertia=0.3941737016672115))
            elif args.model == 'strongsort':
                self.trackers.append(StrongSORT(model_path='Tracker/DeepSort/deep/checkpoint/ckpt.t7', use_cuda=True, use_efficientdet=True))
            elif args.model == 'botsort':
                self.trackers.append(BoTSORT(model_path='Tracker/DeepSort/deep/checkpoint/ckpt.t7', use_cuda=True))
            elif args.model == 'motdt':
                self.trackers.append(Motdt(model_path='Tracker/DeepSort/deep/checkpoint/ckpt.t7', use_cuda=True, use_efficientdet=True))
            else:
                print('Model not found!')
                exit()
        self.frame_id = 0
        self.model_tracker = args.model

    def _display(self, preds, imgs, text_recorder=None, track_result=None):
        self.frame_id += 1
        palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
        if len(preds['rois']) == 0:
            return imgs
        for j in range(len(preds['rois'])):
            (x1, y1, x2, y2) = preds['rois'][j].astype(np.float32)
            cls_id = self.coco_classes[preds['class_ids'][j]]
            obj_id = int(preds['obj_ids'][j])
            color = [int((p * (obj_id ** 2 - obj_id + 1)) % 255) for p in palette]
            cv2.rectangle(imgs, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(imgs, '{}, {}'.format(cls_id, obj_id), (int(x1), int(y1) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            if text_recorder is not None:
                text_recorder.write(','.join([str(self.frame_id), str(float(obj_id - 1)), str(x1), str(y1), str(x2), str(y2), str(1), str(-1), str(-1), str(-1)]))
                text_recorder.write("\n")
                track_result[cls_id].add(obj_id)
        return imgs

    def detect_video(self):
        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()
        cap = cv2.VideoCapture(self.video_src)
        if not cap.isOpened():
            raise IOError("Couldn't open webcam or video")
        video_FourCC = int(cap.get(cv2.CAP_PROP_FOURCC))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        isVideoOutput = True if (self.video_output != "") else False
        if isVideoOutput:
            output2video = cv2.VideoWriter(self.video_output, video_FourCC, video_fps, video_size)
        isTextOutput = True if (self.text_output != "") or (self.text_output is not None) else False
        if isTextOutput:
            output2text = open(self.text_output, 'w', encoding='utf-8')
            track_result = {}
            for obj_cls in self.coco_classes:
                track_result[obj_cls] = set([])
        else:
            output2text = None
            track_result = None
        accum_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        prev_time = timer()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            ori_imgs, framed_imgs, framed_metas = preprocess_video(frame, max_size=self.input_size)
            x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
            x = x.to(torch.float32 if not self.use_float16 else torch.float16).permute(0, 3, 1, 2)
            with torch.no_grad():
                features, regression, classification, anchors = self.detector(x)
                out = postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, self.threshold, self.iou_threshold)
            out = invert_affine(framed_metas, out)
            bbox_xyxy = out[0]['rois']
            bbox_xywh = xyxy_to_xywh(bbox_xyxy)
            cls_ids = out[0]['class_ids']
            cls_conf = out[0]['scores']
            tracker_out = {'rois': np.empty(shape=(0, 4)), 'class_ids': np.empty(shape=(0,), dtype=np.int32), 'obj_ids': np.empty(shape=(0,), dtype=np.int32)}
            for index, target in enumerate(self.selected_target):
                mask = cls_ids == target
                bbox = bbox_xywh[mask]
                conf = cls_conf[mask]
                if self.model_tracker == 'deepsort':
                    outputs = self.trackers[index].update(bbox_xywh=bbox, confidences=conf, ori_img=frame)
                elif self.model_tracker == 'sort':
                    outputs = self.trackers[index].update(dets=np.column_stack([bbox_xyxy[mask], conf]))
                elif self.model_tracker == 'bytetrack':
                    # outputs[:, -1] is conf
                    outputs = np.array(self.trackers[index].update(output_results=np.column_stack([bbox_xyxy[mask], conf])))
                elif self.model_tracker == 'sortoh':
                    outputs, _, _ = self.trackers[index].update(dets=np.column_stack([bbox_xyxy[mask], conf]), gts=None)
                elif self.model_tracker == 'ocsort':
                    # outputs[:, -1] is conf
                    outputs = self.trackers[index].update(dets=np.column_stack([bbox_xyxy[mask], conf]))
                elif self.model_tracker == 'deepocsort':
                    outputs = self.trackers[index].update(dets=np.column_stack([bbox_xyxy[mask], conf]), img_numpy=frame)
                elif self.model_tracker == 'strongsort':
                    # outputs[:, -2] is conf
                    outputs = self.trackers[index].update(dets=np.column_stack([bbox_xyxy[mask], conf]), ori_img=frame)
                elif self.model_tracker == 'botsort':
                    # outputs[:, -1] is conf
                    outputs = np.array(self.trackers[index].update(output_results=np.column_stack([bbox_xyxy[mask], conf]), img=frame))
                elif self.model_tracker == 'motdt':
                    # outputs[:, -1] is conf
                    outputs = np.array(self.trackers[index].update(dets=np.column_stack([bbox_xyxy[mask], conf]), ori_img=frame))
                if len(outputs) > 0:
                    tracker_out['rois'] = np.append(tracker_out['rois'], outputs[:, 0:4], axis=0)
                    tracker_out['class_ids'] = np.append(tracker_out['class_ids'], np.repeat(target, outputs.shape[0]))
                    if self.model_tracker == 'deepsort' or self.model_tracker == 'sort' or self.model_tracker == 'deepocsort':
                        tracker_out['obj_ids'] = np.append(tracker_out['obj_ids'], outputs[:, -1])
                    elif self.model_tracker == 'strongsort':
                        tracker_out['obj_ids'] = np.append(tracker_out['obj_ids'], outputs[:, -3])
                    else:
                        tracker_out['obj_ids'] = np.append(tracker_out['obj_ids'], outputs[:, -2])
            img_show = self._display(tracker_out, ori_imgs[0], output2text, track_result)
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0
            cv2.putText(img_show, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.50, color=(255, 255, 0), thickness=2)
            cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
            cv2.imshow("frame", img_show)
            if isVideoOutput:
                output2video.write(img_show)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        output2text.close()

if __name__ == "__main__":
    detector = MOT()
    detector.detect_video()