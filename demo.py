import cv2
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='SORT and its extension demo')
    parser.add_argument('--model', type=str, default='deepsort')
    parser.add_argument('--dataset', type=str, default='mot16')
    parser.add_argument('--phase', type=str, default='train')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if args.dataset == 'mot16':
        if args.phase == 'train':
            seq = ['02', '04', '05', '09', '10', '11', '13']
            len_seq = [600, 1050, 837, 525, 654, 900, 750]
        elif args.phase == 'test':
            seq = ['01', '03', '06', '07', '08', '12', '14']
            len_seq = [450, 1500, 1194, 500, 625, 900, 750]
        else:
            print('Phase not found!')
            exit()
    elif args.dataset == 'mot20':
        if args.phase == 'train':
            seq = ['01', '02', '03', '05']
            len_seq = [429, 2782, 2405, 3315]
        elif args.phase == 'test':
            seq = ['04', '06', '07', '08']
            len_seq = [2080, 1008, 585, 806]
        else:
            print('Phase not found!')
            exit()
    else:
        print('Dataset not found!')
        exit()
    for s in range(len(seq)):
      print('Creating video for {}-{}'.format(args.dataset.upper(), seq[s]))
      outputFile = '{}-{}-{}.avi'.format(args.dataset.upper(), seq[s], args.model)
      images_path = "data/{}/{}/{}-{}/img1".format(args.dataset.upper(), args.phase, args.dataset.upper(), seq[s])
      image_name = os.path.join(images_path, "{0:0=6d}".format(1) + ".jpg")
      image = cv2.imread(image_name, cv2.IMREAD_COLOR)
      vid_writer = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15, (image.shape[1], image.shape[0]))
      for frame in range(1, len_seq[s] + 1):
          image_name = os.path.join(images_path, "{0:0=6d}".format(int(frame)) + ".jpg")
          image = cv2.imread(image_name, cv2.IMREAD_COLOR)
          with open('output/{}/{}/{}/{}-{}.txt'.format(args.model, args.dataset.upper(), args.phase, args.dataset.upper(), seq[s])) as fin:
            for line in fin:
              frame_cnt, id, top, left, height, width, conf, x, y, z = line.split(',')
              if int(frame_cnt) == frame and float(top) > 0 and float(left) > 0 and float(height) > 0 and float(width) > 0:
                  xmin = int(float(top))
                  ymin = int(float(left))
                  xmax = int(float(top) + float(height))
                  ymax = int(float(left) + float(width))
                  display_text = '%d' % (int(id))
                  color_rectangle = (0, 0, 255)
                  cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=color_rectangle, thickness=2)
                  font = cv2.FONT_HERSHEY_PLAIN
                  color_text = (255, 255, 255)
                  cv2.putText(image, display_text, (xmin + int((xmax - xmin) / 2), ymin + int((ymax - ymin) / 2)), fontFace=font, fontScale=1.3, color=color_text, thickness=2)
              elif int(frame_cnt) > frame:
                  break
          xmin = 0
          ymin = 0
          xmax = 1920
          ymax = 1080
          display_text = 'Frame %d' % (frame)
          font = cv2.FONT_HERSHEY_PLAIN
          color_text = (0, 0, 255)
          cv2.putText(image, display_text, (50, 50), fontFace=font, fontScale=1.3, color=color_text, thickness=2)
          vid_writer.write(image)
      cv2.destroyAllWindows()
      vid_writer.release()

if __name__ == '__main__':
    main()