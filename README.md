# Mooon: Graph Data Augmentation Library

<p align="left">
  <img width = "230" height = "55" src="./imgs/favicon.png" alt="banner"/>
  <br/>
</p>
<p align="center"><strong></strong></p>

<p align=left>
  <a href="https://www.python.org/downloads/release/python-370/">
    <img src="https://img.shields.io/badge/Python->=3.7-3776AB?logo=python" alt="Python">
  </a>
  <a href="https://github.com/pytorch/pytorch">
    <img src="https://img.shields.io/badge/PyTorch->=1.8-FF6F00?logo=pytorch" alt="pytorch">
  </a>
  <a href="https://github.com/EdisonLeeeee/Mooon/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/EdisonLeeeee/Mooon" alt="license">
  </a>
</p>

> 人有*悲欢离合*，月有*阴晴圆缺*。         ———— 苏轼《水调歌头》

# Why "Mooon"?

*Graph* with data augmentations, is like the *moon*, now dark, now full.

# Quick Tour
+ Functional API
```python
from mooon import drop_edge

edge_index, edge_weight = drop_edge(edge_index, p=0.5)
edge_index, edge_weight = drop_edge(edge_index, edge_weight, p=0.5)
```
+ Module Layer
```python
from mooon import DropEdge

drop_edge = DropEdge(p=0.5)
edge_index, edge_weight = drop_edge(edge_index)
edge_index, edge_weight = drop_edge(edge_index, edge_weight)
```
# 🚀 Installation

Please make sure you have installed [PyTorch](https://pytorch.org) and [PyTorch Geometric (PyG)](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

```bash
# Coming soon
pip install -U mooon
```

or

```bash
# Recommended
git clone https://github.com/EdisonLeeeee/Mooon.git && cd Mooon
pip install -e . --verbose
```

where `-e` means "editable" mode so you don't have to reinstall every time you make changes.


# Roadmap

**Note:** this is an ongoing project, please feel free to contact me for collaboration.

- [ ] Based on PyTorch
- [ ] Support only PyG
- [ ] High-level class and low-level functional API
- [ ] Seamlessly integrated into existing code written by PyG
