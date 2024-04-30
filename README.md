# DINOv2 Fine-Tuning and Production Benchmarking Pipeline

## Overview

This project is dedicated to implementing fine-tuning for the DINOv2 model to adapt the model to satellite imagery. The aim is to enhance the model's performance on performing binary classification for satellite imagery, and set up the model for further domain-related adaptation.

## Model Details

The DINOv2 model, based on the Vision Transformer (ViT) architecture, is enhanced in this project to better adapt to specific domains or tasks through fine-tuning on targeted datasets.

## Q1: Do you expect to see good results from fine-tuning just the linear classification head on satellite imagery?

> Due to the drastic differences in the imagery from the model's training data, we can expect that the results initially may not be very accurate, and that additional adaptations may be necessary. And, given that only the head is being trained, there's less risk of overfitting compared to training the entire network--but that may also lead to slower performance gains.

## DINOv2 Architecture

![image](https://github.com/jnieus01/dinov2-finetune/blob/main/dinov2-arch.png)

### DINOv2 Layers with Linear Classification Head

\_LinearClassifierWrapper(
(backbone): DinoVisionTransformer(
(patch_embed): PatchEmbed(
(proj): Conv2d(3, 1024, kernel_size=(14, 14), stride=(14, 14))
(norm): Identity()
)
(blocks): ModuleList(
(0-23): 24 x NestedTensorBlock(
(norm1): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
(attn): MemEffAttention(
(qkv): Linear(in_features=1024, out_features=3072, bias=True)
(attn_drop): Dropout(p=0.0, inplace=False)
(proj): Linear(in_features=1024, out_features=1024, bias=True)
(proj_drop): Dropout(p=0.0, inplace=False)
)
(ls1): LayerScale()
(drop_path1): Identity()
(norm2): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
(mlp): Mlp(
(fc1): Linear(in_features=1024, out_features=4096, bias=True)
(act): GELU(approximate='none')
(fc2): Linear(in_features=4096, out_features=1024, bias=True)
(drop): Dropout(p=0.0, inplace=False)
)
(ls2): LayerScale()
(drop_path2): Identity()
)
)
(norm): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
(head): Identity()
)
(linear_head): Linear(in_features=5120, out_features=1000, bias=True)
)

## LoRA

### LoRA Weight Matrices

![image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/lora_diagram.png)

### Q2: Do you expect LoRA parameter-efficient fine-tuning to improve performance? Why/why not?

> Vision Transformers generally need even more data than CNNs because the ViT needs to learn by itself 2D positional information and other image properties that are inherent in the structure of a CNN. Because the training dataset is small, we might not expect an improvement in results, generally. LoRA modifies the attention and feed-forward layers of transformer models by adding low-rank matrices to the weight matrices in these layers. These matrices are smaller in size and specifically designed to capture the most significant updates needed for adaptation to new tasks or domains. LoRA's targeted approach to modifying model parameters allows the model to learn new features or adjust its existing features without the risk of catastrophic forgetting or overfitting that might occur with full model retraining.

### DINOv2 + LoRA Architecture

PeftModel(
(base_model): LoraModel(
(model): \_LinearClassifierWrapper(
(backbone): DinoVisionTransformer(
(patch_embed): PatchEmbed(
(proj): Conv2d(3, 1024, kernel_size=(14, 14), stride=(14, 14))
(norm): Identity()
)
(blocks): ModuleList(
(0-23): 24 x NestedTensorBlock(
(norm1): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
(attn): MemEffAttention(
(qkv): lora.Linear(
(base_layer): Linear(in_features=1024, out_features=3072, bias=True)
(lora_dropout): ModuleDict(
(default): Dropout(p=0.1, inplace=False)
)
(lora_A): ModuleDict(
(default): Linear(in_features=1024, out_features=4, bias=False)
)
(lora_B): ModuleDict(
(default): Linear(in_features=4, out_features=3072, bias=False)
)
(lora_embedding_A): ParameterDict()
(lora_embedding_B): ParameterDict()
)
(attn_drop): Dropout(p=0.0, inplace=False)
(proj): Linear(in_features=1024, out_features=1024, bias=True)
(proj_drop): Dropout(p=0.0, inplace=False)
)
(ls1): LayerScale()
(drop_path1): Identity()
(norm2): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
(mlp): Mlp(
(fc1): Linear(in_features=1024, out_features=4096, bias=True)
(act): GELU(approximate='none')
(fc2): Linear(in_features=4096, out_features=1024, bias=True)
(drop): Dropout(p=0.0, inplace=False)
)
(ls2): LayerScale()
(drop_path2): Identity()
)
)
(norm): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
(head): Identity()
)
(linear_head): ModulesToSaveWrapper(
(original_module): Linear(in_features=1024, out_features=2, bias=True)
(modules_to_save): ModuleDict(
(default): Linear(in_features=1024, out_features=2, bias=True)
)
)
)
)
)

## Critical Analysis

The results here suggest that LoRA has great potential for obtaining high-accuracy in the domain of satellite imagery using classification tasks. LoRA is directly aimed at fine-tuning pre-trained models by adapting their parameters in a low-rank format. However, LoRA actually increases the model size, which can pose a problem in the future for training on a more robust remote sensor dataset and including additional channels to the input. DoRA is a technique proposed soon after LoRA concerned with optimizing neural network quantization. Further tests should be conducted to evaluate the efficiency and efficacy of LoRA on larger, more complex satellite imagery data and comparisons can and should be made with DoRA to evaluate whether eiether can be realistically used on a "full" dataset.

## Project Structure

```
├── setup_finetune.ipynb          # Notebook for fine-tuning DINOv2 on new datasets
├── finetune_lora.ipynb           # Notebook for fine-tuning DINOv2 with LoRA
├── training_history/        # Folder containing fine-tuning history
└── model.onnx.png        # Diagram of DINOv2 architecture
```

## References

- [DINOv2 Research Resources](https://dinov2.metademolab.com/)

- [Building a Vision Transformer from Scratch with PyTorch ](https://www.akshaymakes.com/blogs/vision-transformer)

- [Vision Transformers 1: Low Earth Orbit Satellites](https://myrtle.ai/resources/leo-1-low-earth-orbit-satellites/)

- [Huggingface LoRA Guide](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)

- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models (arXiv:2106.09685). arXiv. http://arxiv.org/abs/2106.09685

- Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba, W., Howes, R., Huang, P.-Y., Li, S.-W., Misra, I., Rabbat, M., Sharma, V., … Bojanowski, P. (2024). DINOv2: Learning Robust Visual Features without Supervision (arXiv:2304.07193). arXiv. http://arxiv.org/abs/2304.07193

## Contact

For any queries or further assistance, please contact Jordan Nieusma (jordan.m.nieusma@vanderbilt.edu)
