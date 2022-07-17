---
layout: distill
title: Notes About Books
description: Notes about libraries for deep learning, especially with graphs.
date: 2021-11-08

authors:
  - name: Lukas Gosch
    url: "https://saper0.github.io/"
    affiliations:
      name: Data Mining and Machine Learning, TU Munich

bibliography: 2021-11-08_libraries.bib

## PyTorch

Allows to easily implement deep learning models by providing preimplementations of common layers, optimization methods or whole deep learning models. I assume you are quite familiar with PyTorch and hence, won't go into details about it here. Below you find an example implementation of a convolutional neural network for references of the other sections.

{% highlight python linenos %}

from torch.nn import Conv2d

class CNN(torch.nn.Module):
  def __init__(self):
    self.conv1 = Conv2d(3, 64)
    self.conv2 = Conv2d(64, 64)

    def forward(self, input):
      h = self.conv1(input)
      h = h.relu()
      h = self.conv2(h)
      return h

{% endhighlight %}

## PyTorch Geometric (PyG)

Allows to easily implement graph neural networks in a PyTorch fashion. See this example taken from a talk of the creator of PyG (Twitter: @rusty1s) implementing a convolutional graph neural network and compare it with the example in the PyTorch section.

{% highlight python linenos %}

from torch_geometric.nn import GCNConv

class GNN(torch.nn.Module):
  def __init__(self):
    self.conv1 = GCNConv(3, 64)
    self.conv2 = GCNConv(64, 64)

    def forward(self, input, edge_index):
      h = self.conv1(input, edge_index)
      h = h.relu()
      h = self.conv2(h, edge_index)
      return h

{% endhighlight %}

## PyTorch Ligthning

Library to simplify research code generation and scaling models, "automates" the engineering parts of deep learning research.

ToDo: Show how above code (extended with GPU usages) can be simplified and scaled with PyTorch Lighning

## Captum

Library built on PyTorch for interpretability research. Allows to better debug deep learning models and implements methods such as integrated gradients.

## BackPACK

Library built on top of PyTorch to extract more information from backpropagation to compute approximations of the Hessian or variance of the gradients to implement higher order optimization schemes. https://backpack.pt/