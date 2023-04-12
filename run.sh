#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/4/10 14:46
# @Author   : DrizzleSh
# @Usage    : run the SAM

# CUDA_VISIBLE_DEVICES=2,3 python scripts/amg.py --checkpoint ./default.pth --model-type vit_h --input ~/fewimages/ --output ~/segout/ --box-nms-thresh 0.1 --min-mask-region-area 8000
CUDA_VISIBLE_DEVICES=0,1 python scripts/mySAM.py --images_dir ~/fewimages/ --result_dir ~/segout/