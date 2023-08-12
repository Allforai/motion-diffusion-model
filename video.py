import os
import cv2


def image_to_video(file, output, fps):
    num = os.listdir(file)
    height = 1024
    weight = 1280
    # fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videowriter = cv2.VideoWriter(output, fourcc, fps, (weight, height))
    filenameNum = len(num)
    for i in range(0, filenameNum):
        path = file + '/frame_' + str(i).zfill(4) + '.png'
        frame = cv2.imread(path)
        videowriter.write(frame)

    videowriter.release()


if __name__ == '__main__':
    path = os.getcwd()
    keyids = 'crab_walk'
    relative_path = '/mnt/disk_1/jinpeng/motion-diffusion-model/compare/' + keyids + '.mp4'
    output = os.path.join(path, relative_path)
    file = '/mnt/disk_1/jinpeng/motion-diffusion-model/compare/crab_walk'
    image_to_video(file=file, output=output, fps=12)