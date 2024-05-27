# UGATIT-PyTorch



### Credit

#### U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation

_Junho Kim, Minjae Kim, Hyeonwoo Kang, Kwanghee Lee_ <br>

**Abstract** <br>
We propose a novel method for unsupervised image-to-image translation, which incorporates a new attention module 
and a new learnable normalization function in an end-to-end manner. The attention module guides our model to focus 
on more important regions distinguishing between source and target domains based on the attention map obtained 
by the auxiliary classifier. Unlike previous attention-based methods which cannot handle the geometric changes 
between domains, our model can translate both images requiring holistic changes and images requiring large shape 
changes. Moreover, our new AdaLIN (Adaptive Layer-Instance Normalization) function helps our attention-guided 
model to flexibly control the amount of change in shape and texture by learned parameters depending on datasets. 
Experimental results show the superiority of the proposed method compared to the existing state-of-the-art 
models with a fixed network architecture and hyper-parameters.

[[Paper]](https://arxiv.org/pdf/1907.10830) [[Authors' Implementation (TensorFlow)]](https://github.com/taki0112/UGATIT) [[Authors' Implementation (PyTorch)]](https://github.com/znxlwm/UGATIT-pytorch) 

```
@inproceedings{
    Kim2020U-GAT-IT:,
    title={U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation},
    author={Junho Kim and Minjae Kim and Hyeonwoo Kang and Kwang Hee Lee},
    booktitle={International Conference on Learning Representations},
    year={2020},
    url={https://openreview.net/forum?id=BJlZ5ySKPH}
}
```
