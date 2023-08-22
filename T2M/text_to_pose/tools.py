from torch.distributions import kl_divergence, Normal
import torch


def compute_text2poses_similarity(tds, pds):
    kl_divs = []
    for td, pd in zip(tds, pds):  # (1x32), (n_generate, 32)
        kl = kl_divergence(pd, td).sum(-1, keepdims=True)  # (n_generate x 32) -> (n_generate x 1)
        kl_divs.append(kl)
    return kl_divs


def viterbi_algorithm(energies):
    num_nodes = len(energies)
    num_states = len(energies[0])

    # Initialize the Viterbi matrix and the backpointer matrix
    viterbi = [[0.0] * num_states for _ in range(num_nodes)]
    backpointers = [[0] * num_states for _ in range(num_nodes)]

    # Initialize the first column of the Viterbi matrix with initial probabilities
    for state in range(num_states):
        viterbi[0][state] = energies[0][state][0]

    # Forward pass: fill in the Viterbi matrix and backpointer matrix
    for node in range(1, num_nodes):
        for state in range(num_states):
            min_prob = float('inf')
            min_prev_state = -1
            for prev_state in range(num_states):
                prob = viterbi[node - 1][prev_state] + energies[node][state][prev_state]
                if prob < min_prob:
                    min_prob = prob
                    min_prev_state = prev_state
            viterbi[node][state] = min_prob
            backpointers[node][state] = min_prev_state

    # Backward pass: find the optimal path
    path = [0] * num_nodes
    min_final_prob = min(viterbi[-1])
    last_state = viterbi[-1].index(min_final_prob)

    path[-1] = last_state
    for node in range(num_nodes - 2, -1, -1):
        path[node] = backpointers[node + 1][path[node + 1]]

    return path


def search_optimal_path(pose_dists, text_kl_divs, device, n_generate, op='all'):
    kl_divs = [torch.zeros(n_generate, n_generate).to(device)]  # make len(kl_divs) == len(text_kl_divs)
    for i in range(len(pose_dists) - 1):
        pose_dist1 = pose_dists[i + 1]  # (n_generate x 32)
        pose_dist2 = pose_dists[i]  # (n_generate x 32)

        # (n_generate**2 x 32)
        pose_dist1_proxy = Normal(pose_dist1.loc.unsqueeze(1).repeat(1, n_generate, 1).view(-1, pose_dist1.loc.shape[-1]),
                                  pose_dist1.scale.unsqueeze(1).repeat(1, n_generate, 1).view(-1, pose_dist1.scale.shape[-1]))
        # (n_generate**2 x 32)
        pose_dist2_proxy = Normal(pose_dist2.loc.repeat(n_generate, 1),
                                  pose_dist2.scale.repeat(n_generate, 1))

        # (n_generate**2 x 32) -> (n_generate**2) -> (n_generate x n_generate)
        kl_div = kl_divergence(pose_dist1_proxy, pose_dist2_proxy).sum(-1).view(n_generate, n_generate)
        kl_divs.append(kl_div)

    pose_kl_divs = torch.stack(kl_divs)  # (8 x n_generate x n_generate)
    text_kl_divs = torch.stack(text_kl_divs).expand(-1, -1, n_generate) * 100.0  # (8 x n_generate x 1) -> (8 x n_generate x n_generate)

    if op == 'all':
        energies = pose_kl_divs + text_kl_divs
    elif op == 'text':
        energies = text_kl_divs
    elif op == 'pose':
        energies = pose_kl_divs
    else:
        raise ValueError(f'Unknown {op}')

    path = viterbi_algorithm(energies.tolist())
    return path