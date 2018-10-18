# ShuffleNetV2_vs_MnasNet.PyTorch

Non-official implement of Paper：[MnasNet: Platform-Aware Neural Architecture Search for Mobile](https://arxiv.org/abs/1807.11626?context=cs.LG)

Non-official implement of Paper：[ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164v1)

## Requirements

- Python3
- PyTorch 0.4.1
- tensorboardX (optional)
- torchnet
- pretrainedmodels (optional)

## Results

We just test four models in ImageNet-1K, both train set and val set are scaled to 256(minimal side), only use **Mirror** and **RandomResizeCrop** as training data augmentation, during validation, we use center crop to get 224x224 patch.

CPU Info: Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz

### ImageNet-1K

Models         | validation(Top-1) | validation(Top-5) | CPU Cost(ms) |
-------------  | ----------------- | ----------------- | ------------ |
MnasNet        | 64.91             | 86.28             | ~300         |

## Note

Maybe the implement of these network have some different from origin method, we can not achieve the best performance as said in the paper. 