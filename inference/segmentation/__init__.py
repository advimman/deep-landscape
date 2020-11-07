from .base import *


def make_still_segm(segm_net, image, target_size=None, crop_params=None, movable_classes=[2, 21],
                    device='cuda', return_full_segm=False):
    with torch.no_grad():
        device = torch.device(device)
        image = image.to(device)
        image = (image + 1) / 2
        in_image_segm = segm_net.predict(image,
                                         imgSizes=[image.shape[-1]])
        in_image_movable_scores = in_image_segm[:, movable_classes].max(1, keepdim=True)[0]
        immovable_classes = [i for i in range(in_image_segm.shape[1]) if i not in movable_classes]
        in_image_immovable_scores = in_image_segm[:, immovable_classes].max(1, keepdim=True)[0]
        in_image_still_segm = (in_image_immovable_scores > in_image_movable_scores).float()

        if target_size is not None:
            in_image_still_segm = F.interpolate(in_image_still_segm,
                                                size=target_size, mode='bilinear', align_corners=False)
            in_image_segm = F.interpolate(in_image_segm, size=target_size, mode='bilinear', align_corners=False)

        if crop_params is not None:
            x1, y1, x2, y2 = crop_params
            in_image_still_segm = in_image_still_segm[:, :, y1:y2, x1:x2]
            in_image_segm = in_image_segm[:, :, y1:y2, x1:x2]

    if return_full_segm:
        return in_image_still_segm, in_image_segm
    else:
        return in_image_still_segm
