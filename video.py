import os
import cv2


def image_to_video(file, output, fps):
    # num = os.listdir(file)
    # height = 1024
    # weight = 1280
    # # fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # videowriter = cv2.VideoWriter(output, fourcc, fps, (weight, height))
    # filenameNum = len(num)
    # for i in range(0, filenameNum):
    #     path = file + '/frame_' + str(i).zfill(4) + '.png'
    #     frame = cv2.imread(path)
    #     videowriter.write(frame)
    #
    # videowriter.release()
    os.system(r"ffmpeg -y -framerate 12 -i {}/frame_%04d.png  "
                  r"{}.mp4".format(file, output))

if __name__ == '__main__':
    # path = os.getcwd()

    # output = os.path.join(path, relative_path)
    # file = '/mnt/disk_1/jinpeng/motion-diffusion-model/save/pose_length/cross_length32/samples_cross_length32_000150000_seed10/render_7149_0/007149_repeat_0'
    # output = file + '.mp4'
    path = '/mnt/disk_1/jinpeng/motion-diffusion-model/0831_unseen'
    data_path = []
    for file in os.listdir(path):
        for name_1 in os.listdir(os.path.join(path, file)):
            for name_2 in os.listdir(os.path.join(path, file, name_1)):
                if os.path.isdir(os.path.join(path, file, name_1, name_2)):
                    data_path.append(os.path.join(path, file, name_1, name_2))
    for nn in data_path:
        if 'ballet' in nn:
            image_to_video(file=nn, output=nn, fps=12)
