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



#### Encoder-Decoder

#### Attention

#### Feed-Forward Networks

#### Positional Encoding
