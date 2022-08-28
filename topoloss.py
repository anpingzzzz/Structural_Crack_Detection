import time
import numpy
import gudhi as gd
from pylab import *
import torch
from config import DEVICE, DIM_BETTI_NUMBER, ADD_PATCH_EDGE, INVERSE_IMAGE
import matplotlib.pyplot as plt

def edge_addition(lh):
    lh[0, :] = 1
    lh[:, 0] = 1
    lh[-1, :] = 1
    lh[:, -1] = 1
    return lh

def compute_dgm_force(lh_dgm, gt_dgm, pers_thresh, pers_thresh_perfect=0.99, do_return_perfect=False):
    """
    Compute the persistent diagram of the image

    Args:
        lh_dgm: likelihood persistent diagram.
        gt_dgm: ground truth persistent diagram.
        pers_thresh: Persistent threshold, which also called dynamic value, which measure the difference.
        between the local maximum critical point value with its neighouboring minimum critical point value.
        The value smaller than the persistent threshold should be filtered. Default: 0.03
        pers_thresh_perfect: The distance difference between two critical points that can be considered as
        correct match. Default: 0.99
        do_return_perfect: Return the persistent point or not from the matching. Default: False

    Returns:
        force_list: The matching between the likelihood and ground truth persistent diagram
        idx_holes_to_fix: The index of persistent points that requires to fix in the following training process
        idx_holes_to_remove: The index of persistent points that require to remove for the following training
        process

    """
    lh_pers = abs(lh_dgm[:, 1] - lh_dgm[:, 0])
    lh_distance = (lh_dgm[:, 1] - 1) ** 2 + lh_dgm[:, 0] ** 2
    # lh_to_diag = (lh_dgm[:, 1] - lh_dgm[:, 0]) ** 2 / 2
    if (gt_dgm.shape[0] == 0):
        gt_pers = None
        gt_n_holes = 0
    else:
        gt_pers = gt_dgm[:, 1] - gt_dgm[:, 0]
        gt_n_holes = gt_pers.size  # number of holes in gt

    if (gt_pers is None or gt_n_holes == 0):
        idx_holes_to_fix = list()
        idx_holes_to_remove = list(set(range(lh_pers.size)))
        idx_holes_perfect = list()
    else:
        # check to ensure that all gt dots have persistence 1
        tmp = gt_pers > pers_thresh_perfect
        # print(gt_pers[0])
        assert tmp.sum() >= 1
        # get "perfect holes" - holes which do not need to be fixed, i.e., find top
        # lh_n_holes_perfect indices
        # check to ensure that at least one dot has persistence 1 it is the hole
        # formed by the padded boundary
        # if no hole is ~1 (ie >.999) then just take all holes with max values
        tmp = lh_pers > pers_thresh_perfect  # old: assert tmp.sum() >= 1
        # lh_pers_sorted_indices = np.argsort(lh_pers)[::-1] # pers doesn't mean distance
        lh_pers_sorted_indices = np.argsort(lh_distance)
        if np.sum(tmp) >= 1:
            lh_n_holes_perfect = tmp.sum()
            idx_holes_perfect = lh_pers_sorted_indices[:lh_n_holes_perfect]
        else:
            idx_holes_perfect = list()
        # find top gt_n_holes indices
        idx_holes_to_fix_or_perfect = lh_pers_sorted_indices[:gt_n_holes]
        # the difference is holes to be fixed to perfect
        idx_holes_to_fix = list(
            set(idx_holes_to_fix_or_perfect) - set(idx_holes_perfect))

        # remaining holes are all to be removed
        idx_holes_to_remove = lh_pers_sorted_indices[gt_n_holes:]
    # only select the ones whose persistence is large enough
    # set a threshold to remove meaningless persistence dots
    pers_thd = pers_thresh
    idx_valid = np.where(lh_pers > pers_thd)[0]
    idx_holes_to_remove = list(
        set(idx_holes_to_remove).intersection(set(idx_valid)))
    force_list = np.zeros(lh_dgm.shape)
    # push each hole-to-fix to (0,1)
    force_list[idx_holes_to_fix, 0] = 0 - lh_dgm[idx_holes_to_fix, 0]
    force_list[idx_holes_to_fix, 1] = 1 - lh_dgm[idx_holes_to_fix, 1]
    # push each hole-to-remove to line y=x
    force_list[idx_holes_to_remove, 0] = lh_pers[idx_holes_to_remove] / 2
    force_list[idx_holes_to_remove, 1] = -lh_pers[idx_holes_to_remove] / 2
    if (do_return_perfect):
        return force_list, idx_holes_to_fix, idx_holes_to_remove, idx_holes_perfect

    return force_list, idx_holes_to_fix, idx_holes_to_remove

def getCriticalPoints(likelihood):
    """
    Compute the critical points of the image (Value range from 0 -> 1)

    Args:
        likelihood: Likelihood image from the output of the neural networks

    Returns:
        pd_lh:  persistence diagram.
        bcp_lh: Birth critical points.
        dcp_lh: Death critical points.
        Bool:   Skip the process if number of matching pairs is zero.

    """
    lh = 1 - likelihood
    lh_vector = np.asarray(lh).flatten()

    lh_cubic = gd.CubicalComplex(
        dimensions=[lh.shape[0], lh.shape[1]],
        top_dimensional_cells=lh_vector
    )

    Diag_lh = lh_cubic.persistence(homology_coeff_field=2, min_persistence=0)
    pairs_lh = lh_cubic.cofaces_of_persistence_pairs()
    # plot persistence diagram
    # barcodes = lh_cubic.persistence_intervals_in_dimension(1)
    # gd.plot_persistence_diagram(barcodes)
    # plt.show()
    # If the paris is 0, return False to skip
    if (len(pairs_lh[0]) == 0): return 0, 0, 0, False
    # print(pairs_lh)
    pairs_lh_dim0 = pairs_lh[0][DIM_BETTI_NUMBER]
    # return persistence diagram, birth/death critical points
    pd_lh = []
    bcp_lh = []
    dcp_lh = []
    for birth, death in pairs_lh_dim0:
        pd_lh.append([lh_vector[birth], lh_vector[death]])
        bcp_lh.append([birth // lh.shape[1], birth % lh.shape[1]])
        dcp_lh.append([death // lh.shape[1], death % lh.shape[1]])
    pd_lh = np.array(pd_lh)
    bcp_lh = np.array(bcp_lh)
    dcp_lh = np.array(dcp_lh)

    return pd_lh, bcp_lh, dcp_lh, True


def getTopoLoss(likelihood_tensor, gt_tensor, topo_size=100):
    """
    Calculate the topology loss of the predicted image and ground truth image
    Warning: To make sure the topology loss is able to back-propagation, likelihood
    tensor requires to clone before detach from GPUs. In the end, you can hook the
    likelihood tensor to GPUs device.

    Args:
        likelihood_tensor:   The likelihood pytorch tensor.
        gt_tensor        :   The groundtruth of pytorch tensor.
        topo_size        :   The size of the patch is used. Default: 100

    Returns:
        loss_topo        :   The topology loss value (tensor)

    """

    likelihood = torch.sigmoid(likelihood_tensor).clone()
    gt = gt_tensor.clone()

    likelihood = torch.squeeze(likelihood).cpu().detach().numpy()
    gt = torch.squeeze(gt).cpu().detach().numpy()

    if INVERSE_IMAGE:
        likelihood = 1 - likelihood
        gt = 1 - gt
    topo_cp_weight_map = np.zeros(likelihood.shape)
    topo_cp_weight_map_scaling = np.zeros(likelihood.shape)  # holes to remove should be scaled by 1/2
    topo_cp_ref_map = np.zeros(likelihood.shape)

    for y in range(0, likelihood.shape[0], topo_size):
        for x in range(0, likelihood.shape[1], topo_size):

            lh_patch = likelihood[y:min(y + topo_size, likelihood.shape[0]),
                       x:min(x + topo_size, likelihood.shape[1])]
            gt_patch = gt[y:min(y + topo_size, gt.shape[0]),
                       x:min(x + topo_size, gt.shape[1])]

            if ADD_PATCH_EDGE:
                lh_patch = edge_addition(lh_patch)
                gt_patch = edge_addition(gt_patch)

            if (np.min(lh_patch) == 1 or np.max(lh_patch) == 0): continue
            if (np.min(gt_patch) == 1 or np.max(gt_patch) == 0): continue

            # Get the critical points of predictions and ground truth
            pd_lh, bcp_lh, dcp_lh, pairs_lh_pa = getCriticalPoints(lh_patch)
            pd_gt, bcp_gt, dcp_gt, pairs_lh_gt = getCriticalPoints(gt_patch)

            # If the pairs not exist, continue for the next loop
            if not (pairs_lh_pa): continue
            if not (pairs_lh_gt): continue

            force_list, idx_holes_to_fix, idx_holes_to_remove = compute_dgm_force(pd_lh, 
                                                                                  pd_gt, pers_thresh=0.03)
            # more readable version by myself
            if (len(idx_holes_to_fix) > 0 or len(idx_holes_to_remove) > 0):
                for hole_indx in idx_holes_to_fix:
                    birth_y = int(bcp_lh[hole_indx][0])
                    birth_x = int(bcp_lh[hole_indx][1])
                    death_y = int(dcp_lh[hole_indx][0])
                    death_x = int(dcp_lh[hole_indx][1])
                    if (birth_y >= 0 and birth_y < lh_patch.shape[0]
                            and birth_x >= 0 and birth_x < lh_patch.shape[1]):
                        topo_cp_weight_map[y + birth_y, x + birth_x] = 1
                        # push birth to 0 i.e. min birth prob or likelihood
                        topo_cp_weight_map_scaling[y + birth_y, x + birth_x] = 1
                        topo_cp_ref_map[y + birth_y, x + birth_x] = 0
                        # print(1-likelihood_tensor[y + birth_y, x + birth_x],end=" ")
                    if (death_y >= 0 and death_y < lh_patch.shape[0]
                            and death_x >= 0 and death_x < lh_patch.shape[1]):
                        topo_cp_weight_map[y + death_y, x + death_x] = 1
                        # push death to 1 i.e. max death prob or likelihood
                        topo_cp_weight_map_scaling[y + death_y, x + death_x] = 1
                        topo_cp_ref_map[y + death_y, x + death_x] = 1
                        # print(1- likelihood_tensor[y + death_y, x + death_x], end="\n")
                for hole_indx in idx_holes_to_remove:
                    birth_y = int(bcp_lh[hole_indx][0])
                    birth_x = int(bcp_lh[hole_indx][1])
                    death_y = int(dcp_lh[hole_indx][0])
                    death_x = int(dcp_lh[hole_indx][1])
                    if (birth_y >= 0 and birth_y < lh_patch.shape[0]
                            and birth_x >= 0 and birth_x < lh_patch.shape[1]):
                        topo_cp_weight_map[y + birth_y, x + birth_x] = 1
                        # push birth to death  # push to diagonal
                        topo_cp_weight_map_scaling[y + birth_y, x + birth_x] = 1 / 2
                        if (death_y >= 0 and death_y < lh_patch.shape[0]
                                and death_x >= 0 and death_x < lh_patch.shape[1]):
                            # topo_cp_ref_map[y + birth_y, x + birth_x] = likelihood[death_y, death_x]
                            # because we used 1 - lh in computing persistence diagram
                            topo_cp_ref_map[y + birth_y, x + birth_x] = 1 - likelihood[y + death_y, x + death_x]
                        else:
                            topo_cp_ref_map[y + birth_y, x + birth_x] = 1
                    if (death_y >= 0 and death_y < lh_patch.shape[0]
                            and death_x >= 0 and death_x < lh_patch.shape[1]):
                        topo_cp_weight_map[y + death_y, x + death_x] = 1  # push death to birth # push to diagonal
                        topo_cp_weight_map_scaling[y + death_y, x + death_x] = 1 / 2
                        if (birth_y >= 0 and birth_y < lh_patch.shape[0]
                                and birth_x >= 0 and birth_x < lh_patch.shape[1]):
                            # topo_cp_ref_map[y + death_y, x + death_x] = lh_patch[birth_y, birth_x]
                            topo_cp_ref_map[y + death_y, x + death_x] = 1 - likelihood[y + birth_y, x + birth_x]

                        else:
                            topo_cp_ref_map[y + death_y, x + death_x] = 0

    topo_cp_weight_map = torch.tensor(topo_cp_weight_map, dtype=torch.float).to(DEVICE)
    topo_cp_weight_map_scaling = torch.tensor(topo_cp_weight_map_scaling, dtype=torch.float).to(DEVICE)
    topo_cp_ref_map = torch.tensor(topo_cp_ref_map, dtype=torch.float).to(DEVICE)
    # Measuring the MSE loss between predicted critical points and reference critical points
    loss_topo = (((((1 - likelihood_tensor) * topo_cp_weight_map) - topo_cp_ref_map)
                  * topo_cp_weight_map_scaling) ** 2).sum()
    return loss_topo

if __name__ == "__main__":
    gt = imread('./test_img/gt.png')[:, :, 0]
    gt = np.round(1 - gt)
    lh = 1 - imread('./test_img/lh2.png')[:, :, 0]
    loss_topo = getTopoLoss(torch.from_numpy(lh), torch.from_numpy(gt), topo_size=50)
    print(loss_topo)
