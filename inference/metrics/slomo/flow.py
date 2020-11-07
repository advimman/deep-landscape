import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from .model import UNet, backWarp

# MODEL_PATH_RUSSIA = "/Vol1/dbstore/datasets/multimodal/PretrainedModels/SuperSloMo.ckpt"
# MODEL_PATH_KOREA = "/group-volume/orc_srr/multimodal/SuperSloMo.ckpt"
# MODEL_PATH = MODEL_PATH_KOREA if is_korean_cluster() else MODEL_PATH_RUSSIA

# MODEL_PATH = os.path.join(os.environ['DATA_PATH'], 'pretrained_models/SuperSloMo.ckpt')


class SloMoFlow(torch.nn.Module):
    def __init__(self, model_path):
        """
        Basic class initialisation.
        :param model_path: Full path for .ckpt of SuperSloMo.
        Original version could be found at: https://drive.google.com/open?id=1IvobLDbRiBgZr3ryCRrWL8xDbMZ-KnpF
        """
        super().__init__()
        self.mean = [0.429, 0.431, 0.397]
        negmean = [x * -1 for x in self.mean]

        self.std = [1, 1, 1]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.reverse_normalization = transforms.Normalize(mean=negmean, std=self.std)
        self.mean = torch.from_numpy(np.array(self.mean)).float().to(device)
        self.std = torch.from_numpy(np.array(self.std)).float().to(device)

        self.flow_estimator = UNet(6, 4).to(device)
        for param in self.flow_estimator.parameters():
            param.requires_grad = False

        dict1 = torch.load(model_path)
        self.flow_estimator.load_state_dict(dict1['state_dictFC'])
        self.back_warp = backWarp(1024, 1024, device).to(device)
        self.flow_interpolation = UNet(20, 5).to(device)
        for param in self.flow_interpolation.parameters():
            param.requires_grad = False
        self.flow_interpolation.load_state_dict(dict1['state_dictAT'])

    def _convert(self, tensor):
        """

        :param tensor: torch.FloatTensor with values in [0;255]
        :return: Normalised torch.FloatTensor
        """
        for ind in range(len(tensor)):
            tensor[ind] /= 255
            tensor[ind] = tensor[ind].sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        return tensor

    def generate_frames(self, source, target, num_frames):
        """
        Generation of intermediate frames from source to target based on SloMo paper.
        :param source: torch.FloatTensor
        :param target: torch.FloatTensor
        :param num_frames: int
        :return: list of torch.FloatTensor frames
        """
        num_frames += 1
        out_array = []
        forward_flow, backward_flow = self._flow_estimation(source, target)

        for intermediate_index in range(1, num_frames):
            current_time = intermediate_index / num_frames
            temp = -current_time * (1 - current_time)
            flow_coefficient = [temp, current_time * current_time, (1 - current_time) * (1 - current_time), temp]

            backward_intermediate_flow = flow_coefficient[0] * forward_flow + flow_coefficient[1] * backward_flow
            intermediate_forward_flow = flow_coefficient[2] * forward_flow + flow_coefficient[3] * backward_flow

            warped_backward = self.back_warp(self._source, backward_intermediate_flow)
            warped_forward = self.back_warp(self._target, intermediate_forward_flow)

            interpolated_frame = self.flow_interpolation(torch.cat((self._source, self._target,
                                                                    forward_flow,
                                                                    backward_flow,
                                                                    intermediate_forward_flow,
                                                                    backward_intermediate_flow,
                                                                    warped_forward,
                                                                    warped_backward), dim=1))

            interpolated_backward_flow = interpolated_frame[:, :2, :, :] + backward_intermediate_flow
            interpolated_forward_flow = interpolated_frame[:, 2:4, :, :] + intermediate_forward_flow
            final_interpolated_frame = F.sigmoid(interpolated_frame[:, 4:5, :, :])
            inverted_interpolated_frame = 1 - final_interpolated_frame

            warped_backward_interpolated_frame = self.back_warp(self._source, interpolated_backward_flow)
            warped_forward_interpolated_frame = self.back_warp(self._target, interpolated_forward_flow)

            coefficients = [1 - current_time, current_time]

            final_flow = (coefficients[0] * final_interpolated_frame * warped_backward_interpolated_frame +
                          coefficients[1] * inverted_interpolated_frame * warped_forward_interpolated_frame) / (
                                 coefficients[0] * final_interpolated_frame + coefficients[
                             1] * inverted_interpolated_frame)

            final_flow = final_flow.cpu().detach()

            for batchIndex in range(self._source.shape[0]):
                cur_frame = self.reverse_normalization(final_flow[batchIndex])
                cur_frame = np.transpose(cur_frame.data.numpy(), (1, 2, 0))
                cur_frame *= 255
                cur_frame = cur_frame.clip(0, 255).astype(np.uint8)
                out_array.append(cur_frame)
        return out_array

    def _check_input(self, tensor):
        assert torch.all(tensor >= 0), 'All input values should be bigger 0'
        assert torch.any(tensor > 1), 'Values should be in range [0:255]'

    def _flow_estimation(self, source, target):
        """
        Estimates flow between two batches of images.

        :param source: torch.FloatTensor [batch x num_channels x height x weight]
        :param target: torch.FloatTensor [batch x num_channels x height x weight]
        :return: torch.FloatTensor [batch x 2 x height x weight]
        """
        self._check_input(source)
        self._check_input(target)

        self._source = self._convert(source)
        self._target = self._convert(target)

        flow = self.flow_estimator(torch.cat((self._source.clone(),
                                              self._target.clone()), dim=1))

        forward_flow = flow[:, :2, :, :]
        backward_flow = flow[:, 2:, :, :]
        return forward_flow.contiguous(), backward_flow.contiguous()

    def forward(self, source, target=None):
        """Expects values strictly in range [0;255] where maximal value in tensor is 255
        
        Input: torch.FloatTensor [~video_batch x batch x num_channels x height x weight]
        
        Output: torch.FloatTensor [~video_batch x batch x 2 x height x weight]
                with OpticalFlow (Source to target, Target to source)
        
        """
        assert (len(source.size()) == 4 or len(source.size()) == 5), "Bad input shape"
        if len(source.size()) == 5:
            if target is None:
                target = source[:, 1:]
                source = source[:, :-1]

            b, n, c, h, w = source.size()
            source = source.view(-1, c, h, w)
            target = target.view(-1, c, h, w)
            flow, _ = self._flow_estimation(source, target)
            return flow.view(b, 2, n, h, w)
        else:
            if target is None:
                target = source[1:]
                source = source[:-1]

            flow, _ = self._flow_estimation(source, target)
            return flow
