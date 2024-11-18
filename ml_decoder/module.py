import torch
from torch import nn
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed


import numpy as np
from dotted_dict import DottedDict


from src_files.helper_functions.bn_fusion import fuse_bn_recursively
from src_files.models import create_model
from src_files.models.tresnet.tresnet import InplacABN_to_ABN


class inference_ml_decoder(nn.Module):
    def __init__(
        self,
        model_path="./ML_Decoder/tresnet_xl_COCO_640_91_4.pth",
        model_name="tresnet_xl",
        image_size=640,
    ):
        super(inference_ml_decoder, self).__init__()
        args = DottedDict()
        args.num_classes = 80  # COCO
        args.model_path = model_path
        args.model_name = model_name
        args.image_size = image_size
        args.th = 0.75
        args.top_k = 20
        args.use_ml_decoder = 1
        args.num_of_groups = -1
        args.decoder_embedding = 768
        args.zsl = 0
        self.model = create_model(args, load_head=True).cuda()
        state = torch.load(args.model_path, map_location="cpu")
        self.model.load_state_dict(state["model"], strict=False)
        self.model = self.model.cpu()
        self.model = InplacABN_to_ABN(self.model)
        self.model = fuse_bn_recursively(self.model)
        self.model = self.model.cuda().half().eval()
        print(f"MLDecoder Model Loading Done: {args.model_path}")
        self.classes_list = np.array(list(state["idx_to_class"].values()))
        self.args = args

    def __call__(self, pil_img):
        im = pil_img
        im_resize = im.resize((self.args.image_size, self.args.image_size))
        np_img = np.array(im_resize, dtype=np.uint8)
        tensor_img = (
            torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0
        )  # HWC to CHW
        tensor_batch = torch.unsqueeze(tensor_img, 0).cuda().half()  # float16 inference
        output = torch.squeeze(torch.sigmoid(self.model(tensor_batch)))
        np_output = output.cpu().detach().numpy()

        idx_sort = np.argsort(-np_output)
        detected_classes = np.array(self.classes_list)[idx_sort][: self.args.top_k]
        scores = np_output[idx_sort][: self.args.top_k]
        idx_th = scores > self.args.th
        detected_classes = detected_classes[idx_th]
        return detected_classes
