import os
import cv2


def image_to_video(file, output, fps):
    os.system(r"ffmpeg -y -framerate 12 -i {}/frame_%04d.png  "
              r"{}.mp4".format(file, output))


if __name__ == '__main__':
    # path = os.getcwd()

    # output = os.path.join(path, relative_path)
    # file = '/mnt/disk_1/jinpeng/motion-diffusion-model/save/pose_length/cross_length32/samples_cross_length32_000150000_seed10/render_7149_0/007149_repeat_0'
    # output = file + '.mp4'
    path = '/mnt/disk_1/jinpeng/motion-diffusion-model/1006_gptprompt_3/2023-10-06-18-33-53'
    data_path = []
    for file in os.listdir(path):
        if '.json' not in file:
            data_path.append(os.path.join(path, file, 'vertices'))
            print(data_path)
    for nn in data_path:
        # if 'waltz' in nn:
        image_to_video(file=nn, output=nn, fps=12)



