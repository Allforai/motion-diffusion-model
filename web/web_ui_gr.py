import os
import numpy as np
import gradio as gr
# from web.tools import *
import pydevd_pycharm
# pydevd_pycharm.settrace('10.8.31.54', port=17778, stdoutToServer=True, stderrToServer=True)
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils.parser_util import generate_args
from data_loaders.p2m.dataset import HumanML3D
from torch.utils.data import DataLoader
from utils.fixseed import fixseed


from PIL import Image
import cv2

#
def generate_output(text):
    #
    img = Image.new('RGB', (300, 300), color=(255, 0, 0))
    img_path = 'output_image.jpg'
    img.save(img_path)

    #
    video_path = 'output_video.mp4'
    frame_width, frame_height = 640, 480
    fps = 30
    duration = 5
    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    for i in range(fps * duration):
        frame = cv2.rectangle(
            np.zeros((frame_height, frame_width, 3), dtype=np.uint8),
            (i % frame_width, i % frame_height),
            ((i + 100) % frame_width, (i + 100) % frame_height),
            (0, 255, 0),
            3
        )
        video_writer.write(frame)

    video_writer.release()

    return img_path, video_path

# ??Gradio??
iface = gr.Interface(
    fn=generate_output,
    inputs="text",
    outputs=["image", "video"],
    title="qwe",
    description="qw"
)

# ??Gradio??
iface.launch(share=True)
