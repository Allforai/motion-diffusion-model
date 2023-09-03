import os
import numpy as np
import streamlit as st
from web.tools import *
import pydevd_pycharm
# pydevd_pycharm.settrace('10.8.31.54', port=17778, stdoutToServer=True, stderrToServer=True)
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils.parser_util import generate_args
from data_loaders.p2m.dataset import HumanML3D
from torch.utils.data import DataLoader
from utils.fixseed import fixseed


@st.cache_resource
def get_model():
    args = generate_args()
    fixseed(args.seed)
    data = HumanML3D(datapath='dataset/p2m_humanml_opt.txt', split='test')
    train_loader = DataLoader(data, batch_size=1, shuffle=True, num_workers=8)
    model, diffusion = create_model_and_diffusion(args, train_loader)
    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)
    model.to('cuda')
    model.eval()  # disable random masking
    return model, diffusion

@st.cache_resource
def get_pose_model():
    pose_model_path = '/mnt/disk_1/jinpeng/motion-diffusion-model/text2pose/experiments/eccv22_posescript_models/CondTextPoser_textencoder-glovebigru_vocA1H1_latentD32/train-posescript-H1/wloss_kld0.2_v2v4.0_rot2.0_jts2.0_kldnpmul0.02_kldntmul0.0/B32_Adam_lr1e-05_wd0.0001_pretrained_gen_glovebigru_vocA1H1_dataA1/seed0/checkpoint_1999.pth'
    pose_model, _ = load_model(pose_model_path, 'cuda')
    body_model = BodyModel(bm_fname='/mnt/disk_1/jinpeng/motion-diffusion-model/body_models/SMPLH_NEUTRAL.npz',
                           num_betas=16)
    body_model.eval()
    body_model.to('cuda')
    return pose_model, body_model


def main():
    st.title("Zero-Shot Motion Generator")

    # 文本输入
    text_input = st.text_input("Please Enter Your Order: (a person cries, a man dance)")

    # gpt response
    split_text = gpt_response(text_input)

    # 文字输出
    st.header("GPT Response")
    st.write(split_text)

    # pose generation
    st.header("Pose Response")
    pose_model, body_model = get_pose_model()
    image_list, merged_image, pose_list = pose_response(split_text, pose_model, body_model)
    col1, col2, col3, col4 = st.columns(4)
    # 在每个列中显示图片
    with col1:
        st.image(image_list[0], use_column_width=True)
    with col2:
        st.image(image_list[2], use_column_width=True)
    with col3:
        st.image(image_list[4], use_column_width=True)
    with col4:
        st.image(image_list[6], use_column_width=True)

    with col1:
        st.image(image_list[1], use_column_width=True)
    with col2:
        st.image(image_list[3], use_column_width=True)
    with col3:
        st.image(image_list[5], use_column_width=True)
    with col4:
        st.image(image_list[7], use_column_width=True)

    st.image(merged_image)
    #
    st.header("Motion Response")
    model, diffusion = get_model()
    vertices = motion_response(pose_list, model, diffusion)
    vertices_path = '/mnt/disk_1/jinpeng/motion-diffusion-model/web/data/vertices.npy'
    np.save(vertices_path, vertices)
    os.system(r'blender --background --python web/render_webui.py')

    st.image('/mnt/disk_1/jinpeng/motion-diffusion-model/web/data/vertices.png')
    os.system(r"ffmpeg -y -framerate 12 -i {}/frame_%04d.png {}".format('/mnt/disk_1/jinpeng/motion-diffusion'
                                                                            '-model/web/picture_folder',
                                                                            '/mnt/disk_1/jinpeng/motion-diffusion'
                                                                            '-model/web/data/vertices.mp4'))
    st.video('/mnt/disk_1/jinpeng/motion-diffusion-model/web/data/vertices.mp4')


if __name__ == "__main__":
    main()
