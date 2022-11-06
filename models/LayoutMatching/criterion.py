# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils.box_util import generalized_box3d_iou
from utils.dist import all_reduce_average
from utils.misc import huber_loss
from scipy.optimize import linear_sum_assignment

import utils.pc_util as pc_util


class Matcher(nn.Module):
    def __init__(self, cost_class, cost_objectness, cost_giou, cost_center):
        """
        Parameters:
            cost_class:
        Returns:

        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_objectness = cost_objectness
        self.cost_giou = cost_giou
        self.cost_center = cost_center

    @torch.no_grad()
    def forward(self, outputs, targets):

        batchsize = outputs["sem_cls_prob"].shape[0]
        nqueries = outputs["sem_cls_prob"].shape[1]
        ngt = targets["gt_box_sem_cls_label"].shape[1]
        nactual_gt = targets["nactual_gt"]

        # classification cost: batch x nqueries x ngt matrix
        pred_cls_prob = outputs["sem_cls_prob"]
        gt_box_sem_cls_labels = (
            targets["gt_box_sem_cls_label"]
            .unsqueeze(1)
            .expand(batchsize, nqueries, ngt)
        )
        class_mat = -torch.gather(pred_cls_prob, 2, gt_box_sem_cls_labels)

        # objectness cost: batch x nqueries x 1
        objectness_mat = -outputs["objectness_prob"].unsqueeze(-1)

        # center cost: batch x nqueries x ngt
        center_mat = outputs["center_dist"].detach()

        # giou cost: batch x nqueries x ngt
        giou_mat = -outputs["gious"].detach()

        final_cost = (
            self.cost_class * class_mat
            + self.cost_objectness * objectness_mat
            + self.cost_center * center_mat
            + self.cost_giou * giou_mat
        )
        final_cost = final_cost.detach().cpu().numpy()
        assignments = []

        # auxiliary variables useful for batched loss computation
        batch_size, nprop = final_cost.shape[0], final_cost.shape[1]
        per_prop_gt_inds = torch.zeros(
            [batch_size, nprop], dtype=torch.int64, device=pred_cls_prob.device
        )
        proposal_matched_mask = torch.zeros(
            [batch_size, nprop], dtype=torch.float32, device=pred_cls_prob.device
        )
        for b in range(batchsize):
            assign = []
            if nactual_gt[b] > 0:
                assign = linear_sum_assignment(final_cost[b, :, : nactual_gt[b]])
                assign = [
                    torch.from_numpy(x).long().to(device=pred_cls_prob.device)
                    for x in assign
                ]
                per_prop_gt_inds[b, assign[0]] = assign[1]
                proposal_matched_mask[b, assign[0]] = 1
            assignments.append(assign)

        return {
            "assignments": assignments,
            "per_prop_gt_inds": per_prop_gt_inds,
            "proposal_matched_mask": proposal_matched_mask,
        }


class SetCriterion(nn.Module):
    def __init__(self, matcher, dataset_config, loss_weight_dict):
        super().__init__()
        self.dataset_config = dataset_config
        self.matcher = matcher
        self.loss_weight_dict = loss_weight_dict

        semcls_percls_weights = torch.ones(dataset_config.num_semcls + 1)
        semcls_percls_weights[-1] = loss_weight_dict["loss_no_object_weight"]
        del loss_weight_dict["loss_no_object_weight"]
        self.register_buffer("semcls_percls_weights", semcls_percls_weights)

        self.loss_functions = {
            "loss_sem_cls": self.loss_sem_cls,
            "loss_angle": self.loss_angle,
            "loss_center": self.loss_center,
            "loss_size": self.loss_size,
            "loss_giou": self.loss_giou,
            # this isn't used during training and is logged for debugging.
            # thus, this loss does not have a loss_weight associated with it.
            "loss_cardinality": self.loss_cardinality,
        }

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, assignments):
        # Count the number of predictions that are objects
        # Cardinality is the error between predicted #objects and ground truth objects

        pred_logits = outputs["sem_cls_logits"]
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        pred_objects = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(pred_objects.float(), targets["nactual_gt"])
        return {"loss_cardinality": card_err}

    def loss_sem_cls(self, outputs, targets, assignments):

        # # Not vectorized version
        # pred_logits = outputs["sem_cls_logits"]
        # assign = assignments["assignments"]

        # sem_cls_targets = torch.ones((pred_logits.shape[0], pred_logits.shape[1]),
        #                         dtype=torch.int64, device=pred_logits.device)

        # # initialize to background/no-object class
        # sem_cls_targets *= (pred_logits.shape[-1] - 1)

        # # use assignments to compute labels for matched boxes
        # for b in range(pred_logits.shape[0]):
        #     if len(assign[b]) > 0:
        #         sem_cls_targets[b, assign[b][0]] = targets["gt_box_sem_cls_label"][b, assign[b][1]]

        # sem_cls_targets = sem_cls_targets.view(-1)
        # pred_logits = pred_logits.reshape(sem_cls_targets.shape[0], -1)
        # loss = F.cross_entropy(pred_logits, sem_cls_targets, self.semcls_percls_weights, reduction="mean")

        pred_logits = outputs["sem_cls_logits"]
        gt_box_label = torch.gather(
            targets["gt_box_sem_cls_label"], 1, assignments["per_prop_gt_inds"]
        )
        gt_box_label[assignments["proposal_matched_mask"].int() == 0] = (
            pred_logits.shape[-1] - 1
        )
        loss = F.cross_entropy(
            pred_logits.transpose(2, 1),
            gt_box_label,
            self.semcls_percls_weights,
            reduction="mean",
        )

        return {"loss_sem_cls": loss}

    def loss_angle(self, outputs, targets, assignments):
        angle_logits = outputs["angle_logits"]
        angle_residual = outputs["angle_residual_normalized"]

        if targets["num_boxes_replica"] > 0:
            gt_angle_label = targets["gt_angle_class_label"]
            gt_angle_residual = targets["gt_angle_residual_label"]
            gt_angle_residual_normalized = gt_angle_residual / (
                np.pi / self.dataset_config.num_angle_bin
            )

            # # Non vectorized version
            # assignments = assignments["assignments"]
            # p_angle_logits = []
            # p_angle_resid = []
            # t_angle_labels = []
            # t_angle_resid = []

            # for b in range(angle_logits.shape[0]):
            #     if len(assignments[b]) > 0:
            #         p_angle_logits.append(angle_logits[b, assignments[b][0]])
            #         p_angle_resid.append(angle_residual[b, assignments[b][0], gt_angle_label[b][assignments[b][1]]])
            #         t_angle_labels.append(gt_angle_label[b, assignments[b][1]])
            #         t_angle_resid.append(gt_angle_residual_normalized[b, assignments[b][1]])

            # p_angle_logits = torch.cat(p_angle_logits)
            # p_angle_resid = torch.cat(p_angle_resid)
            # t_angle_labels = torch.cat(t_angle_labels)
            # t_angle_resid = torch.cat(t_angle_resid)

            # angle_cls_loss = F.cross_entropy(p_angle_logits, t_angle_labels, reduction="sum")
            # angle_reg_loss = huber_loss(p_angle_resid.flatten() - t_angle_resid.flatten()).sum()

            gt_angle_label = torch.gather(
                gt_angle_label, 1, assignments["per_prop_gt_inds"]
            )
            angle_cls_loss = F.cross_entropy(
                angle_logits.transpose(2, 1), gt_angle_label, reduction="none"
            )
            angle_cls_loss = (
                angle_cls_loss * assignments["proposal_matched_mask"]
            ).sum()

            gt_angle_residual_normalized = torch.gather(
                gt_angle_residual_normalized, 1, assignments["per_prop_gt_inds"]
            )
            gt_angle_label_one_hot = torch.zeros_like(
                angle_residual, dtype=torch.float32
            )
            gt_angle_label_one_hot.scatter_(2, gt_angle_label.unsqueeze(-1), 1)

            angle_residual_for_gt_class = torch.sum(
                angle_residual * gt_angle_label_one_hot, -1
            )
            angle_reg_loss = huber_loss(
                angle_residual_for_gt_class - gt_angle_residual_normalized, delta=1.0
            )
            angle_reg_loss = (
                angle_reg_loss * assignments["proposal_matched_mask"]
            ).sum()

            angle_cls_loss /= targets["num_boxes"]
            angle_reg_loss /= targets["num_boxes"]
        else:
            angle_cls_loss = torch.zeros(1, device=angle_logits.device).squeeze()
            angle_reg_loss = torch.zeros(1, device=angle_logits.device).squeeze()
        return {"loss_angle_cls": angle_cls_loss, "loss_angle_reg": angle_reg_loss}

    def loss_center(self, outputs, targets, assignments):
        center_dist = outputs["center_dist"]
        if targets["num_boxes_replica"] > 0:

            # # Non vectorized version
            # assign = assignments["assignments"]
            # center_loss = torch.zeros(1, device=center_dist.device).squeeze()
            # for b in range(center_dist.shape[0]):
            #     if len(assign[b]) > 0:
            #         center_loss += center_dist[b, assign[b][0], assign[b][1]].sum()

            # select appropriate distances by using proposal to gt matching
            center_loss = torch.gather(
                center_dist, 2, assignments["per_prop_gt_inds"].unsqueeze(-1)
            ).squeeze(-1)
            # zero-out non-matched proposals
            center_loss = center_loss * assignments["proposal_matched_mask"]
            center_loss = center_loss.sum()

            if targets["num_boxes"] > 0:
                center_loss /= targets["num_boxes"]
        else:
            center_loss = torch.zeros(1, device=center_dist.device).squeeze()

        return {"loss_center": center_loss}

    def loss_giou(self, outputs, targets, assignments):
        gious_dist = 1 - outputs["gious"]

        # # Non vectorized version
        # giou_loss = torch.zeros(1, device=gious_dist.device).squeeze()
        # assign = assignments["assignments"]

        # for b in range(gious_dist.shape[0]):
        #     if len(assign[b]) > 0:
        #         giou_loss += gious_dist[b, assign[b][0], assign[b][1]].sum()

        # select appropriate gious by using proposal to gt matching
        giou_loss = torch.gather(
            gious_dist, 2, assignments["per_prop_gt_inds"].unsqueeze(-1)
        ).squeeze(-1)
        # zero-out non-matched proposals
        giou_loss = giou_loss * assignments["proposal_matched_mask"]
        giou_loss = giou_loss.sum()

        if targets["num_boxes"] > 0:
            giou_loss /= targets["num_boxes"]

        return {"loss_giou": giou_loss}

    def loss_size(self, outputs, targets, assignments):
        gt_box_sizes = targets["gt_box_sizes_normalized"]
        pred_box_sizes = outputs["size_normalized"]

        if targets["num_boxes_replica"] > 0:

            # # Non vectorized version
            # p_sizes = []
            # t_sizes = []
            # assign = assignments["assignments"]
            # for b in range(pred_box_sizes.shape[0]):
            #     if len(assign[b]) > 0:
            #         p_sizes.append(pred_box_sizes[b, assign[b][0]])
            #         t_sizes.append(gt_box_sizes[b, assign[b][1]])
            # p_sizes = torch.cat(p_sizes)
            # t_sizes = torch.cat(t_sizes)
            # size_loss = F.l1_loss(p_sizes, t_sizes, reduction="sum")

            # construct gt_box_sizes as [batch x nprop x 3] matrix by using proposal to gt matching
            gt_box_sizes = torch.stack(
                [
                    torch.gather(
                        gt_box_sizes[:, :, x], 1, assignments["per_prop_gt_inds"]
                    )
                    for x in range(gt_box_sizes.shape[-1])
                ],
                dim=-1,
            )
            size_loss = F.l1_loss(pred_box_sizes, gt_box_sizes, reduction="none").sum(
                dim=-1
            )

            # zero-out non-matched proposals
            size_loss *= assignments["proposal_matched_mask"]
            size_loss = size_loss.sum()

            size_loss /= targets["num_boxes"]
        else:
            size_loss = torch.zeros(1, device=pred_box_sizes.device).squeeze()
        return {"loss_size": size_loss}

    def single_output_forward(self, outputs, targets):
        gious = generalized_box3d_iou(
            outputs["box_corners"],
            targets["gt_box_corners"],
            targets["nactual_gt"],
            rotated_boxes=torch.any(targets["gt_box_angles"] > 0).item(),
            needs_grad=(self.loss_weight_dict["loss_giou_weight"] > 0),
        )

        outputs["gious"] = gious
        center_dist = torch.cdist(
            outputs["center_normalized"], targets["gt_box_centers_normalized"], p=1
        )
        outputs["center_dist"] = center_dist
        assignments = self.matcher(outputs, targets)

        losses = {}

        for k in self.loss_functions:
            loss_wt_key = k + "_weight"
            if (
                loss_wt_key in self.loss_weight_dict
                and self.loss_weight_dict[loss_wt_key] > 0
            ) or loss_wt_key not in self.loss_weight_dict:
                # only compute losses with loss_wt > 0
                # certain losses like cardinality are only logged and have no loss weight
                curr_loss = self.loss_functions[k](outputs, targets, assignments)
                losses.update(curr_loss)

        final_loss = 0
        for k in self.loss_weight_dict:
            if self.loss_weight_dict[k] > 0:
                losses[k.replace("_weight", "")] *= self.loss_weight_dict[k]
                final_loss += losses[k.replace("_weight", "")]
        return final_loss, losses

    def forward(self, outputs, targets):
        nactual_gt = targets["gt_box_present"].sum(axis=1).long()
        num_boxes = torch.clamp(all_reduce_average(nactual_gt.sum()), min=1).item()
        targets["nactual_gt"] = nactual_gt
        targets["num_boxes"] = num_boxes
        targets[
            "num_boxes_replica"
        ] = nactual_gt.sum().item()  # number of boxes on this worker for dist training

        loss, loss_dict = self.single_output_forward(outputs["outputs"], targets)

        if "aux_outputs" in outputs:
            for k in range(len(outputs["aux_outputs"])):
                interm_loss, interm_loss_dict = self.single_output_forward(
                    outputs["aux_outputs"][k], targets
                )

                loss += interm_loss
                for interm_key in interm_loss_dict:
                    loss_dict[f"{interm_key}_{k}"] = interm_loss_dict[interm_key]
        return loss, loss_dict


# TODO: add nce softmax loss used by contrastive learning.
class NCESoftmaxLoss(nn.Module):
    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, label):
        bsz = x.shape[0]
        x = x.squeeze()
        loss = self.criterion(x, label)
        return loss


# TODO: add distortion loss used by contrastive learning where the residual error after transformation is computed.
class DistortionLossResidual(nn.Module):
    def __init__(self):
        super(DistortionLossResidual, self).__init__()

    def forward(self, logits, enc_inds_q_sampled, enc_inds_t_sampled, pc_q, pc_t):
        # find the optimial assignment between the query and target points.
        cost_matrix = 1 / (logits.detach().cpu().numpy() + 1e-8)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # take the xyz coordinates of query and target points assigned through the hungarian algorithm.
        enc_inds_t_hungarian = enc_inds_t_sampled[row_ind, col_ind]
        pc_t_hungarian = pc_t[enc_inds_t_hungarian, :3]
        pc_q_hungarian = pc_q[enc_inds_q_sampled, :3]

        # align the query and seed points and return the alignment error.
        R, error = pc_util.svd_rotation(pc_q_hungarian, pc_t_hungarian)

        return error


# TODO: add distortion loss used by contrastive learning where the disparity is computed.
class DistortionLossDisparity(nn.Module):
    def __init__(self, temperature, nce_loss):
        super(DistortionLossDisparity, self).__init__()
        self.temperature = temperature
        self.nce_loss = nce_loss

    def forward(self, q_seed_features_sampled, t_seed_features_sampled, cl_loss_label):
        N, D = q_seed_features_sampled.shape
        disparity = torch.zeros((N, N), dtype=torch.float32, device=q_seed_features_sampled.device)
        pos_pair_idx = 0
        while pos_pair_idx < N:
            # take feature for curr q seed point and its corresponding t.
            q_i = q_seed_features_sampled[pos_pair_idx, :]
            t_i = t_seed_features_sampled[pos_pair_idx, :]

            # randomly select a feature from q points and the feature for its corresponding t.
            indices = [i for i in range(N) if i != pos_pair_idx]
            rand_pos_q_idx = np.random.choice(indices, 1)[0]
            q_j = q_seed_features_sampled[rand_pos_q_idx, :]
            t_j = t_seed_features_sampled[rand_pos_q_idx, :]

            # find feature for negative t points.
            neg_t_indices = [i for i in range(N) if i != rand_pos_q_idx]
            neg_ts = t_seed_features_sampled[neg_t_indices, :]

            # disparity for pos pair.
            disparity[pos_pair_idx, pos_pair_idx] = torch.abs(torch.dot(q_i, q_j) - torch.dot(t_i, t_j))
            disparity[pos_pair_idx, :pos_pair_idx] = torch.abs(torch.dot(q_i, q_j) - torch.mm(t_i.unsqueeze(dim=0),
                                                                                              neg_ts[:pos_pair_idx, :].t()))
            disparity[pos_pair_idx, pos_pair_idx+1:] = torch.abs(torch.dot(q_i, q_j) - torch.mm(t_i.unsqueeze(dim=0),
                                                                                                neg_ts[pos_pair_idx:, :].t()))
            pos_pair_idx += 1

        # compute NCE on the disparity matrix.
        disparity = torch.div(disparity, self.temperature)
        loss = self.nce_loss(disparity, cl_loss_label)

        return loss


# TODO: add the loss for contrastive seed point matching.
class PointContrastiveLoss(nn.Module):
    def __init__(self, npos_pairs, nce_loss, distortion_loss, temperature, distortion_loss_weight,
                 distortion_loss_type=None):
        super().__init__()
        self.npos_pairs = npos_pairs
        self.nce_loss = nce_loss
        self.distortion_loss = distortion_loss
        self.temperature = temperature
        self.distortion_loss_weight = distortion_loss_weight
        self.distortion_loss_type = distortion_loss_type

    def forward(self, q_seed_points_info, t_seed_points_info, cl_loss_label, pc_q, pc_t, evaluation=False):
        # sample positive and negative points for each target seed point
        loss = 0
        batch_size = len(q_seed_points_info)
        batch_logits = torch.zeros((batch_size, self.npos_pairs, self.npos_pairs), dtype=torch.float32,
                                   device=cl_loss_label.device)
        for i in range(batch_size):
            _, q_seed_features, enc_inds_q, q_seed_labels = q_seed_points_info[i]
            _, t_seed_features, enc_inds_t, t_seed_labels = t_seed_points_info[i]
            if len(q_seed_labels) == 0:
                continue

            # normalize the features
            q_seed_features = torch.nn.functional.normalize(q_seed_features, dim=1)
            t_seed_features = torch.nn.functional.normalize(t_seed_features, dim=1)

            if self.distortion_loss_type == 'residual':
                # keep indices of the pos/neg points required for residual distortion loss.
                enc_inds_q_sampled = torch.zeros(self.npos_pairs, dtype=torch.long, device=cl_loss_label.device)
                enc_inds_t_sampled = torch.zeros((self.npos_pairs, self.npos_pairs), dtype=torch.long,
                                                 device=cl_loss_label.device)
            elif self.distortion_loss_type == 'disparity':
                # keep features of the pos/neg points required for disparity distortion loss.
                D = q_seed_features.shape[1]
                q_seed_features_sampled = torch.zeros((self.npos_pairs, D), dtype=torch.float32,
                                                      device=cl_loss_label.device)
                t_seed_features_sampled = torch.zeros((self.npos_pairs, D), dtype=torch.float32,
                                                      device=cl_loss_label.device)

            # find pos/neg examples and compute logits.
            logits = torch.zeros((self.npos_pairs, self.npos_pairs), dtype=torch.float32, device=cl_loss_label.device)
            j = 0
            pos_pair_idx = 0
            bad_example = False
            while pos_pair_idx < self.npos_pairs:
                # find all the target points with the same instance label as the label for the current query point
                curr_label = q_seed_labels[j]
                is_same_instance = (t_seed_labels == curr_label).long()

                # randomly choose one matching query points and n-1 negative examples.
                pos_indices = is_same_instance.nonzero().squeeze(dim=1).detach().cpu()
                neg_indices = (1 - is_same_instance).nonzero().squeeze(dim=1).detach().cpu()

                # skip if no positive or negative found.
                if (pos_indices.dim() < 1) or (neg_indices.dim() < 1) or (len(pos_indices) == 0) or (len(neg_indices) == 0):
                    print('No pos/neg found')
                    bad_example = True
                    break

                if evaluation:
                    np.random.seed(0)
                rand_pos_index = np.random.choice(pos_indices, 1)[0]
                rand_neg_indices = np.random.choice(neg_indices, self.npos_pairs - 1,
                                                    replace=len(neg_indices) < (self.npos_pairs - 1))

                if self.distortion_loss_type == 'residual':
                    # take the indices of the 3d points for the sampled pos and negative points.
                    enc_inds_q_sampled[pos_pair_idx] = enc_inds_q[j]
                    enc_inds_t_sampled[pos_pair_idx, pos_pair_idx] = enc_inds_t[rand_pos_index]
                    enc_inds_t_sampled[pos_pair_idx, :pos_pair_idx] = enc_inds_t[rand_neg_indices[:pos_pair_idx]]
                    enc_inds_t_sampled[pos_pair_idx, pos_pair_idx+1:] = enc_inds_t[rand_neg_indices[pos_pair_idx:]]
                elif self.distortion_loss_type == 'disparity':
                    # take the indices of the 3d points for the sampled pos and negative points.
                    q_seed_features_sampled[pos_pair_idx] = q_seed_features[j]
                    t_seed_features_sampled[pos_pair_idx] = t_seed_features[rand_pos_index]

                # load the positive and negative features.
                t_pos_feature = t_seed_features[rand_pos_index, :]
                t_neg_features = t_seed_features[rand_neg_indices, :]

                # compute the logit given the pos/neg examples.
                logits[pos_pair_idx, pos_pair_idx] = torch.dot(q_seed_features[j, :], t_pos_feature)
                neg_logits = torch.mm(t_neg_features, q_seed_features[j:j + 1, :].t()).squeeze()
                logits[pos_pair_idx, :pos_pair_idx] = neg_logits[:pos_pair_idx]
                logits[pos_pair_idx, pos_pair_idx + 1:] = neg_logits[pos_pair_idx:]

                # update the number of pos pairs constructed.
                pos_pair_idx += 1
                j += 1
                if j == len(q_seed_labels):
                    j = 0

            if not bad_example:
                out = torch.div(logits, self.temperature)
                batch_logits[i, ...] = out
                # compute the distortion loss if needed.
                if self.distortion_loss_type == 'residual':
                    loss += (self.distortion_loss_weight * self.distortion_loss(out,
                                                                                enc_inds_q_sampled,
                                                                                enc_inds_t_sampled,
                                                                                pc_q[i, ...],
                                                                                pc_t[i, ...]))
                elif self.distortion_loss_type == 'disparity':
                    loss += (self.distortion_loss_weight * self.distortion_loss(q_seed_features_sampled,
                                                                                t_seed_features_sampled,
                                                                                cl_loss_label))

                # compute the NCE loss
                loss += self.nce_loss(out, cl_loss_label)

        return loss, batch_logits


def build_criterion(args, dataset_config):
    matcher = Matcher(
        cost_class=args.matcher_cls_cost,
        cost_giou=args.matcher_giou_cost,
        cost_center=args.matcher_center_cost,
        cost_objectness=args.matcher_objectness_cost,
    )

    loss_weight_dict = {
        "loss_giou_weight": args.loss_giou_weight,
        "loss_sem_cls_weight": args.loss_sem_cls_weight,
        "loss_no_object_weight": args.loss_no_object_weight,
        "loss_angle_cls_weight": args.loss_angle_cls_weight,
        "loss_angle_reg_weight": args.loss_angle_reg_weight,
        "loss_center_weight": args.loss_center_weight,
        "loss_size_weight": args.loss_size_weight,
    }
    criterion = SetCriterion(matcher, dataset_config, loss_weight_dict)
    return criterion


def build_criterion_point_contrast(args):
    nce_loss = NCESoftmaxLoss()
    if args.distortion_loss_type == 'residual':
        distortion_loss = DistortionLossResidual()
    else:
        distortion_loss = DistortionLossDisparity(args.temperature, nce_loss)
    criterion = PointContrastiveLoss(args.npos_pairs, nce_loss, distortion_loss, args.temperature,
                                     args.distortion_loss_weight, args.distortion_loss_type)

    return criterion
