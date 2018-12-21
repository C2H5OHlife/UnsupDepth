from __future__ import division
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from inverse_warp import inverse_warp, disp_warp


def compute_ssim(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(3, 1)(x)
    mu_y = nn.AvgPool2d(3, 1)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d

    return torch.clamp((1 - SSIM) / 2, 0, 1).mean()


def reconstruction_loss(tgt_img, ref_imgs, par_img, intrinsics, intrinsics_inv, depth, disps, explainability_mask, pose, rotation_mode='euler', padding_mode='zeros'):
    # depth 自动取第一个channel： depth[:, 0] (line 27) 改成在外面取好第一个channel了
    def one_scale(count, disp, depth, explainability_mask):
        assert (explainability_mask is None or depth.size()[-2:] == explainability_mask.size()[-2:])
        assert (pose.size(1) == len(ref_imgs))

        b, _, h, w = depth.size()
        downscale = tgt_img.size(2) / h

        # 生成图像金字塔
        tgt_img_scaled = F.interpolate(tgt_img, (h, w), mode='area')
        ref_imgs_scaled = [F.interpolate(ref_img, (h, w), mode='area') for ref_img in ref_imgs]
        par_img_scaled = F.interpolate(par_img, (h, w), mode='area')
        intrinsics_scaled = torch.cat((intrinsics[:, 0:2] / downscale, intrinsics[:, 2:]), dim=1)
        intrinsics_scaled_inv = torch.cat((intrinsics_inv[:, :, 0:2] * downscale, intrinsics_inv[:, :, 2:]), dim=2)

        # 1. 计算双目warping loss
        tgt_img_warped, par_img_warped = disp_warp(tgt_img_scaled, par_img_scaled, disp)
        disp_apply_L1 = (tgt_img_scaled - tgt_img_warped).abs().mean() + (par_img_scaled - par_img_warped).abs().mean()
        disp_apply_SSIM = compute_ssim(tgt_img_scaled, tgt_img_warped) + compute_ssim(par_img_scaled, par_img_warped)
        # 双目warping损失由SSIM损失和L1损失组成
        disp_apply_loss = 0.85 * disp_apply_SSIM + 0.15 * disp_apply_L1

        # 2. 计算前后帧warping loss
        reconstruction_loss = 0
        for i, ref_img in enumerate(ref_imgs_scaled):
            current_pose = pose[:, i]

            ref_img_warped = inverse_warp(ref_img, depth[:, 0], current_pose, intrinsics_scaled, intrinsics_scaled_inv, rotation_mode, padding_mode)
            out_of_bound = 1 - (ref_img_warped == 0).prod(1, keepdim=True).type_as(ref_img_warped)
            # 目标图像 - 重建图像 L1损失
            diff = (tgt_img_scaled - ref_img_warped) * out_of_bound

            if explainability_mask is not None:
                diff = diff * explainability_mask[:, i:i + 1].expand_as(diff)

            reconstruction_loss += diff.abs().mean()
            assert ((reconstruction_loss == reconstruction_loss).item() == 1)

            # 3. 计算mask损失
            mask_loss = 0
            if explainability_mask is not None:
                ones_var = Variable(torch.ones(1)).expand_as(explainability_mask).type_as(explainability_mask)
                mask_loss = nn.functional.binary_cross_entropy(explainability_mask, ones_var)

            # 4. 计算smoothness损失
            left_disp_est = disp[:, 0, :, :].unsqueeze(1)
            right_disp_est = disp[:, 1, :, :].unsqueeze(1)

            def gradient_x(img):
                # Pad input to keep output size consistent
                img = F.pad(img, (0, 1, 0, 0), mode="replicate")
                gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
                return gx

            def gradient_y(img):
                # Pad input to keep output size consistent
                img = F.pad(img, (0, 0, 0, 1), mode="replicate")
                gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
                return gy
            smooth_loss = 0
            for d, img in zip([left_disp_est, right_disp_est], [tgt_img_scaled, par_img_scaled]):
                disp_gradients_x = gradient_x(d)
                disp_gradients_y = gradient_y(d)
                img_gradients_x = gradient_x(img).abs()
                img_gradients_y = gradient_y(img).abs()
                weights_x = torch.exp(-torch.mean(img_gradients_x, 1, keepdim=True))
                weights_y = torch.exp(-torch.mean(img_gradients_y, 1, keepdim=True))

                smoothness_x = disp_gradients_x * weights_x
                smoothness_y = disp_gradients_y * weights_y
                smooth_loss += torch.mean(torch.abs(smoothness_x) + torch.abs(smoothness_y)) / (2 ** count)

            # 5. 计算consistency损失
            left_disp_warped, right_disp_warped = disp_warp(left_disp_est, right_disp_est, disp)
            consistency_loss = (left_disp_warped - left_disp_est).abs().mean() + (right_disp_warped - right_disp_est).abs().mean()

            count += 1
            return reconstruction_loss, disp_apply_loss, mask_loss, smooth_loss, consistency_loss

    if type(explainability_mask) not in [tuple, list]:
        explainability_mask = [explainability_mask]
    if type(disps) not in [list, tuple]:
        depth = [depth]
        disps = [disps]

    [loss_1, loss_2, loss_3, loss_4, loss_5] = [0] * 5
    i = 0
    for disp, d, mask in zip(disps, depth, explainability_mask):
        l1, l2, l3, l4, l5 = one_scale(i, disp, d, mask)
        loss_1 += l1
        loss_2 += l2
        loss_3 += l3
        loss_4 += l4
        loss_5 += l5
        i += 1

    return loss_1, loss_2, loss_3, loss_4, loss_5
