---
layout:     post
title:      Transformer论文笔记
subtitle:   Attention Is All You Need
date:       2021-11-1
author:     fennel
header-img: /my_img/resnest50.jpg
catalog: true
tags:
    - 论文笔记
---

## 论文pdf

- [Attention Is All You Need](/paper/Transformer.pdf)

## 前言

![transformert2](/my_img/transformert2.png)

**Purpose：** 循环模型通常沿输入输出的序列顺序来计算，产生一系列的隐藏状态，通过先前的隐藏状态与当前位置的输入来输出，这种方式使得计算过程中无法并行计算。
减少顺序计算也构成了扩展模型的基础，在如ByteNet、ConvS2S等模型中关联两个输入或输出位置的信号所需的计算量随着位置的增加而增加，这使得学习远距离位置之间的依赖关系更加困难。<br>
**Method：** 注意力机制允许对依赖关系建模，无需考虑它们在输入与输出序列中的距离。作者提出了Transformer结构，用多头自注意力取代了编码器-解码器架构中的循环层，实现了一个完全基于注意力的序列转换模型。 <br>
**Results：** 在翻译任务中因为Transformer的高并行计算，训练速度明显快于基于循环或卷积层的架构。在 WMT 2014 English-to-German 和 WMT 2014 English-to-French 翻译任务中取得了最优水平。 <br>


## Architecture

![transformerf1](/my_img/transformerf1.png)

#### Encoder-Decoder

- Encoder：每个编码器层中均有两个子层，多头注意力(multi-head self-attention mechanism)与全连接前馈网络(Feed Forward Neural Network)。并且两个子层中均有残差连接与**层归一化**，即每个子层的输出为 LayerNorm(x + Sublayer(x))。模型中所有子层输出的维度为512(每个词元为一个长度为512的向量)。*batchNorm：对同一个batch下不同样本的同一特征进行归一化，；layerNorm：对同一样本的不同特征进行归一化*
- Decoder：除了编码器中的两个子层以外解码器还加入了一个encoder-decoder attention层，在这一层中Q来自解码器的输出，K、V来自编码器的输出。由于在翻译当前位置时不应该关注后续位置的信息，所以使用masked multi-head attention去掩盖住后续的输出，保证位置i的预测只依赖于位置i之前的输出。解码器中同样加入了参擦连接与层归一化。

#### Attention

在注意力机制中有Query(Q)、Key(K)、Value(V)，attention的计算过程是计算我要查询的Q与每个K的相似程度，再将相似度乘到V上得到最后的值。在self-attention中Q、K、V都从输入的embedding向量中得到，长度均为64。通过三个512×64的权重矩阵学习得到，如下图所示。
![transformerp1](/my_img/transformerp1.png)
attention的计算过程分为7部：
1. 将输入词元转换为embedding向量。
2. 由embedding向量得到Q、K、V三个向量。
3. 每个词元计算一个值score=Q×K^T。
4. 当Q、K向量长度较长时，点积值会过大，再经softmax函数会使梯度过小收敛变慢。所以对score除以[latex]\sqrt{d_k}[/latex]
![transformerp2](/my_img/transformerp2.png)



#### Feed-Forward Networks

#### Positional Encoding
