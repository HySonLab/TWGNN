# Fast Temporal Wavelet Graph Neural Networks (NeurIPS 2023)

![Wavelet](Wavelet.png)

Proceedings: https://proceedings.mlr.press/v228/nguyen24a.html

Workshop on Symmetry and Geometry in Neural Representations: https://openreview.net/pdf?id=Mo5qZaBl8v

Temporal Graph Learning Workshop: https://openreview.net/pdf?id=hh3salTr27

Contributors:
* Nguyen Duc Thien
* Nguyen Manh Duc Tuan
* Hy Truong Son (Correspondent / PI)

Implementation of Multiresolution Matrix Factorization (MMF) and graph wavelet computation from (Hy and Kondor, 2021) https://proceedings.mlr.press/v196/hy22a.html is publicly available at: https://github.com/risilab/Learnable_MMF

* Experiments for brain networks: ```brain-networks/```
* Experiments for traffic prediction: ```traffic-prediction/```

## Please cite our work

```bibtex

@InProceedings{pmlr-v228-nguyen24a,
  title = 	 {Fast Temporal Wavelet Graph Neural Networks},
  author =       {Nguyen, Duc Thien and Nguyen, Manh Duc Tuan and Hy, Truong Son and Kondor, Risi},
  booktitle = 	 {Proceedings of the 2nd NeurIPS Workshop on Symmetry and Geometry in Neural Representations},
  pages = 	 {35--54},
  year = 	 {2024},
  editor = 	 {Sanborn, Sophia and Shewmake, Christian and Azeglio, Simone and Miolane, Nina},
  volume = 	 {228},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {16 Dec},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v228/main/assets/nguyen24a/nguyen24a.pdf},
  url = 	 {https://proceedings.mlr.press/v228/nguyen24a.html},
  abstract = 	 {Spatio-temporal signals forecasting plays an important role in numerous domains, especially in neuroscience and transportation. The task is challenging due to the highly intricate spatial structure, as well as the non-linear temporal dynamics of the network. To facilitate reliable and timely forecast for the human brain and traffic networks, we propose the Fast Temporal Wavelet Graph Neural Networks (FTWGNN) that is both time- and memory-efficient for learning tasks on timeseries data with the underlying graph structure, thanks to the theories of Multiresolution analysis and Wavelet theory on discrete spaces. We employ Multiresolution Matrix Factorization (MMF) (Kondor et al., 2014) to factorize the highly dense graph structure and compute the corresponding sparse wavelet basis that allows us to construct fast wavelet convolution as the backbone of our novel architecture. Experimental results on real-world PEMS-BAY, METR-LA traffic datasets and AJILE12 ECoG dataset show that FTWGNN is competitive with the state-of-the-arts while maintaining a low computational footprint. Our PyTorch implementation is publicly available at https://github.com/HySonLab/TWGNN}
}
```
