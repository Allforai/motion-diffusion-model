from utils.fixseed import fixseed
from torch.utils.data import DataLoader
import os
import torch
from utils.parser_util import generate_args
from utils import dist_util
import numpy as np
from collections import OrderedDict
from datetime import datetime
from scipy import linalg
from model.temos_encoder import ActorAgnosticEncoder
from model.pose_encoder import PoseEncoder
from data_loaders.p2m.eval_dataset import Pose2Motion
torch.set_default_dtype(torch.float32)
from sklearn.metrics import mean_squared_error

import numpy as np

def main():
    args = generate_args()
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    dist_util.setup_dist(args.device)
    log_file = os.path.join(os.path.dirname(args.model_path), 'eval_humanml_{}_{}'.format(name, niter))
    log_file += '.log'
    # print(f'Will save to log file [{log_file}]')
    log_file = '/mnt/disk_1/jinpeng/motion-diffusion-model/save/0813_cross/eval_gpt_gptpose_0813_cross_000300000.log'
    print(f'Will save to log file [{log_file}]')
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_{}_seed{}'.format(name, niter, args.seed))
        if args.text_prompt != '':
            out_path += '_' + args.text_prompt.replace(' ', '_').replace('.', '')
        elif args.input_text != '':
            out_path += '_' + os.path.basename(args.input_text).replace('.txt', '').replace(' ', '_').replace('.', '')
    print("Loading Dataset")
    with open(log_file, 'w') as f:
        all_metrics = OrderedDict({'Matching Score': OrderedDict({}),
                                   'R_precision': OrderedDict({}),
                                   'FID': OrderedDict({}),
                                   'MSE': OrderedDict({}),
                                   'Diversity': OrderedDict({})})
        for replication in range(args.replication_times):
            # dataset = Pose2Motion(os.path.join(out_path, 'results.npy'), replication)
            dataset = Pose2Motion('/mnt/disk_1/jinpeng/motion-diffusion-model/GPT_response_8022_smplh/motion_data_gptpose.npy', replication)
            data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8)
            print(f'==================== Replication {replication} ====================')
            print(f'==================== Replication {replication} ====================', file=f, flush=True)
            # print(f'Time: {datetime.now()}')
            # print(f'Time: {datetime.now()}', file=f, flush=True)
            mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(data_loader, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            fid_score_dict, mse_dict = evaluate_fid_and_mse(data_loader, acti_dict, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            div_score_dict = evaluate_diversity(acti_dict, f, 10)

            print(f'!!! DONE !!!')
            print(f'!!! DONE !!!', file=f, flush=True)

            all_metrics['Matching Score']['repeat_' + str(replication)] = [mat_score_dict]

            all_metrics['R_precision']['repeat_' + str(replication)] = [R_precision_dict]

            all_metrics['FID']['repeat_' + str(replication)] = [fid_score_dict]

            all_metrics['MSE']['repeat_' + str(replication)] = [mse_dict]

            all_metrics['Diversity']['repeat_' + str(replication)] = [div_score_dict]

        mean_dict = {}
        for metric_name, metric_dict in all_metrics.items():
            print('========== %s Summary ==========' % metric_name)
            print('========== %s Summary ==========' % metric_name, file=f, flush=True)
            # print(metric_name, model_name)
            mean, conf_interval = get_metric_statistics(list(metric_dict.values()), args.replication_times)
            mean_dict[metric_name] = mean
            # print(mean, mean.dtype)
            if isinstance(mean, np.float64) or isinstance(mean, np.float32):
                print(f'--->  Mean: {mean:.8f} CInterval: {conf_interval:.4f}')
                print(f'--->  Mean: {mean:.8f} CInterval: {conf_interval:.4f}', file=f, flush=True)
            elif isinstance(mean, np.ndarray):
                line = f'---> '
                if len(mean.shape) == 2:
                    mean = mean.squeeze(0)
                    conf_interval = conf_interval.squeeze(0)
                for i in range(len(mean)):
                    line += '(top %d) Mean: %.8f CInt: %.8f;' % (i+1, mean[i], conf_interval[i])
                print(line)
                print(line, file=f, flush=True)
        return None

def evaluate_matching_score(data_loader, f):
    print('========== Evaluating Matching Score ==========')
    all_size = 0
    matching_score_sum = 0
    top_k_count = 0
    all_motion_embeddings = []
    pose_encoder, motion_encoder = build_models()
    with torch.no_grad():
        for _, cond, _, motion in data_loader:
            cond = cond.to('cuda').squeeze(1).squeeze(-2).permute(0, 2, 1).contiguous()
            motion = motion.to('cuda').squeeze(1).squeeze(-2).permute(0, 2, 1).contiguous()
            motion_embeddings, pose_embeddings = get_co_embeddings(motion, cond, pose_encoder, motion_encoder)
            dist_mat = euclidean_distance_matrix(pose_embeddings.cpu().numpy(), motion_embeddings.cpu().numpy())
            matching_score_sum += dist_mat.trace()

            argsmax = np.argsort(dist_mat, axis=1)
            top_k_mat = calculate_top_k(argsmax, top_k=3)
            top_k_count += top_k_mat.sum(axis=0)

            all_size += pose_embeddings.shape[0]

            all_motion_embeddings.append(motion_embeddings.cpu().numpy())
        all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
        matching_score = matching_score_sum / all_size
        R_precision = top_k_count / all_size
        print(f'---> Matching Score: {matching_score:.4f}')
        print(f'---> Matching Score: {matching_score:.4f}', file=f, flush=True)

        line = f'---> R_precision: '
        for i in range(len(R_precision)):
            line += '(top %d): %.4f ' % (i + 1, R_precision[i])
        print(line)
        print(line, file=f, flush=True)
        return matching_score, R_precision, all_motion_embeddings


def build_models():
    pose_enc = PoseEncoder(num_neurons=512, num_neurons_mini=32, latentD=256, role="retrieval")
    motion_enc = ActorAgnosticEncoder(nfeats=135, vae=False, latent_dim=256, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1, activation="gelu")
    checkpoint = torch.load('/mnt/disk_1/jinpeng/motion-diffusion-model/save/pmm/0816_1821/finest.tar', map_location='cuda')
    pose_enc.load_state_dict(checkpoint['pose_encoder'])
    motion_enc.load_state_dict(checkpoint['motion_encoder'])
    print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoint['epoch']))
    pose_enc.eval()
    motion_enc.eval()
    return pose_enc.to('cuda'), motion_enc.to('cuda')


def get_co_embeddings(motions, pose_features, pose_encoder, motion_encoder):
    device = 'cuda'  # use cpu
    with torch.no_grad():
        motion_encoder.eval()
        pose_encoder.eval()
        '''Movement Encoding'''
        motion_embedding = motion_encoder(motions)

        '''Pose Encoding'''
        pose_embedding = pose_encoder(pose_features)
    return motion_embedding, pose_embedding


def evaluate_fid_and_mse(data_loader, activation_dict, file):
    gt_motion_embeddings = []
    mse_loss = []
    print('========== Evaluating FID ==========')
    _, motion_encoder = build_models()
    loss_fn1 = torch.nn.MSELoss(reduction='none')
    with torch.no_grad():
        for motion, cond, _, sample in data_loader:
            loss1 = loss_fn1(sample, motion)
            motion = motion.to('cuda').squeeze(1).squeeze(-2).permute(0, 2, 1)
            motion_embedding = motion_encoder(motion)
            gt_motion_embeddings.append(motion_embedding)
            mse_loss.append(loss1)
    mse_loss = torch.concatenate(mse_loss, axis=0).cpu().numpy()
    gt_motion_embeddings = torch.concatenate(gt_motion_embeddings, axis=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)
    mu, cov = calculate_activation_statistics(activation_dict)
    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
    print(f'--->  FID: {fid:.8f}')
    print(f'--->  FID: {fid:.8f}', file=file, flush=True)
    print(f"=========Evaluating MSE: ============")
    mse = mse_loss.mean()
    print(f'--->  MSE: {mse:.4f}')
    print(f'--->  MSE: {mse:.4f}', file=file, flush=True)
    return fid, mse


def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists


def calculate_activation_statistics(activations):
    """
    Params:
    -- activation: num_samples x dim_feat
    Returns:
    -- mu: dim_feat
    -- sigma: dim_feat x dim_feat
    """
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def evaluate_diversity(activation_dict, file, diversity_times):
    eval_dict = OrderedDict({})
    print('========== Evaluating Diversity ==========')
    diversity = calculate_diversity(activation_dict, diversity_times)
    eval_dict = diversity
    print(f'---> Diversity: {diversity:.4f}')
    print(f'---> Diversity: {diversity:.4f}', file=file, flush=True)
    return eval_dict


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative dataset set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative dataset set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
        correct_vec = (correct_vec | bool_mat[:, i])
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat


def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


if __name__ == '__main__':
    main()
