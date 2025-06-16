import numpy as np
import cv2
import os
import torch

from basicsr.archs.rrdbnet_arch import RRDBNet


class Resizer(object):

    def apply(self, img, resolution):
        output_width, output_height = resolution
        resized = cv2.resize(img, (output_width, output_height))
        return resized


class SuperResEngine(object):

    def __init__(self, model_path):
        '''
        Initialize the Super Resolution Engine with a pre-trained model.
        :param model_path: Path to the pre-trained model file.
        '''
        torch.cuda.empty_cache()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device = torch.device('cpu')
        # set up model
        model = RRDBNet(num_in_ch=3,
                        num_out_ch=3,
                        num_feat=64,
                        num_block=23,
                        num_grow_ch=32)
        model.load_state_dict(torch.load(model_path)['params'], strict=True)
        model.eval()
        self._model = model.to(device)
        self._device = device
        self._resizer = Resizer()

    def apply_once(self, img):
        '''
        x4 original size
        '''

        img = img.astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]],
                                            (2, 0, 1))).float()
        img = img.unsqueeze(0).to(self._device)
        # inference
        try:
            with torch.no_grad():
                output = self._model(img)
        except Exception as error:
            print('Error', error)
            return None
        else:

            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)
            return output

    def apply(self, img, resolution):
        '''
        Apply super-resolution to the input image.
        :param img: Input image.
        :param resolution: Tuple (width, height) for the output resolution.
        :return: Super-resolved image.
        '''
        H, W, C = img.shape
        output_width, output_height = resolution
        if H > 0.3 * output_height or W > 0.3 * output_width:
            print('cropped image is big, skipping super-resolution.')
            return self._resizer.apply(img, resolution)

        # Apply super-resolution
        sr_img = self.apply_once(img)
        print("applied super res:", sr_img.shape)
        return self._resizer.apply(sr_img, resolution)
