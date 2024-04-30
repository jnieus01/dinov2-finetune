# DINOv2 Fine-Tuning

## Overview

This project is dedicated to implementing fine-tuning for the DINOv2 model to adapt the model to satellite imagery.

The dataset I am using for training is from the Vanderbilt Spatial Analysis and Research Lab's [GeoPACHA project](https://github.com/geopacha). The project strives to implement AI-based archaeological survey to reduce the time required to identify new archaeological sites from satellite imagery and to improve the precision at which these sites (true positives in the dataset) are identified. My work on this project has been to experiment with fine-tuning a pre-trained DINOv2 model to explore whether the learning from DINOv2 can be retained while adapting the model to a very different domain of imagery.  Specifically, the aim of this effort is to enhance the model's performance on performing binary classification for satellite images, and set up the model pipeline for further domain-related adaptation.

I said in my previous presentation that DINOv2 boasts high-performance out-of-the-box, especially on linear classification tasks, presumably because the pre-training approach employed by Meta enables the model to have learned visual features that are highly generalizable. However, on a subset of the GeoPACHA dataset, it does no better than guessing at random. This makes sense when we consider the types of images DINOv2 was trained on - photos of animals, people, objects, not bird's eye views of the Earth. 

By way of example, here are randomly selected images from the GeoPACHA dataset: 
![image](https://github.com/jnieus01/dinov2-finetune/blob/main/image_examples.jpg)


## Q1: Given what we know about fine-tuning, is it reasonable to believe that training just the classification head on satellite imagery will result in notable improvements in the model's ability to detect archaeological features? Why or why not? 

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

Total Parameters: 309,489,640 | Trainable Parameters: 5,121,000 | Trainable%: 1.65

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

Total Parameters: 315,003,856 | Trainable Parameters: 5,514,216 | Trainable%: 1.75
+ Total Parameters: 5,514,216 | Trainable Parameters: 393,216 | Trainable%: 0.10


## Critical Analysis and Future Work

The results here suggest the potential that LoRA has for obtaining high-accuracy in this domain (satellite imagery). However, LoRA does still increase the number of parameters in the model, and training on a more robust remote sensor dataset and including at least 5 additional channel matrices to the input layer will further increase the parameters required for domain-adaptation training. The level of experimentation I conducted for this project was largely possible because it was conducted on DGX A100 machinery, which delivers outstanding, industry-leading training performance--for reference, Meta AI used A100 GPUs to train DINOv2 from scratch. 

I would be remiss if I did not also mention DoRA, a PEFT technique proposed shortly after LoRA. With DoRA, matrix decomposition is applied to hidden layers to create magnitude and directional components, and LoRA is applied only to the directional component and letting the magnitude component be developed separately. DoRA would add a small percentage (as in, 0.01%) of parameters, but research found that DoRA can outperform LoRA even with half the parameter usage of LoRA. 

![image](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F27c269b8-e0cc-4be0-a5d3-bd2a79ac7f29_1600x368.png) (Liu et. al.)

The research on DoRA suggests that even more efficient and accurate training is possible, which bodes well for future training of satellite imagery-based models.  

I mention above that implementing LoRA (and by extension, DoRA) inherently increases the number of parameters to the model, thereby increasing the memory requirments. On the other hand, the training time with LoRA is a fraction of the time used to train DINOv2 from scratch (Meta reported that a single pre-training of a ViT-g model takes approximately 22k GPU-hours), which, in the context of the environmental impact of transformer model development, AI developers can signficantly reduce their carbon footprint by leveraging PEFT techniques to develop their models and obtain--potentially--outstanding performance all the same. 

My future work in this area will also involve implementing the configurations to (1) implement the same experimental parameters from above using DoRA, (2) include the 5 other spectral channel matrices to the input layer of DINOv2 and observe the effects on model performance, and (2) conducting these experiments on a semantic segmentation head. Because objectives (2) and (3) will significantly increase the parameters and memory required to train a domain-adapted model, the results from my experiments with LoRA and DoRA will be useful for efficient experimentation. 

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

- [Improving LoRA: Implementing Weight-Decomposed Low-Rank Adaptation (DoRA) from Scratch](https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch)

- [Vision Transformers 1: Low Earth Orbit Satellites](https://myrtle.ai/resources/leo-1-low-earth-orbit-satellites/)

- [Huggingface LoRA Guide](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)

- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models (arXiv:2106.09685). arXiv. http://arxiv.org/abs/2106.09685
  
- Liu, S.-Y., Wang, C.-Y., Yin, H., Molchanov, P., Wang, Y.-C. F., Cheng, K.-T., & Chen, M.-H. (2024). DoRA: Weight-Decomposed Low-Rank Adaptation (arXiv:2402.09353). arXiv. https://doi.org/10.48550/arXiv.2402.09353

- Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba, W., Howes, R., Huang, P.-Y., Li, S.-W., Misra, I., Rabbat, M., Sharma, V., … Bojanowski, P. (2024). DINOv2: Learning Robust Visual Features without Supervision (arXiv:2304.07193). arXiv. http://arxiv.org/abs/2304.07193

## Contact

For any queries or further assistance, please contact Jordan Nieusma (jordan.m.nieusma@vanderbilt.edu)
