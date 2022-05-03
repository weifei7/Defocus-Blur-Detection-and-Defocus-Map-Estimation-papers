# Defocus-Blur-Detection-and-Defocus-Map-Estimation-papers
Deep Learning based defocus blur detection and defocus map estimation papers in recent 5 years.

## Defocus Blur Detection
##### 1. [Global context guided hierarchically residual feature refinement network for defocus blur detection](https://www.sciencedirect.com/science/article/abs/pii/S0165168421000359)
Published in: Elsevier
Published year: 2021
`@article{ZHAI2021107996,
title = {Global context guided hierarchically residual feature refinement network for defocus blur detection},
journal = {Signal Processing},
volume = {183},
pages = {107996},
year = {2021},
issn = {0165-1684},
doi = {https://doi.org/10.1016/j.sigpro.2021.107996},
url = {https://www.sciencedirect.com/science/article/pii/S0165168421000359},
author = {Yongping Zhai and Junhua Wang and Jinsheng Deng and Guanghui Yue and Wei Zhang and Chang Tang},
keywords = {Defocus blur detection, Feature aggregation, Global context information, Feature fusion},
abstract = {As an important pre-processing step, defocus blur detection makes critical role in various computer vision tasks. However, previous methods cannot obtain satisfactory results due to the complex image background clutter, scale sensitivity and miss of region boundary details. In this paper, for addressing these issues, we introduce a global context guided hierarchically residual feature refinement network (HRFRNet) for defocus blur detection from a natural image. In our network, the low-level fine detail features, high-level semantic and global context information are aggregated in a hierarchical manner to boost the final detection performance. In order to reduce the affect of complex background clutter and smooth regions without enough textures on the final results, we design a multi-scale dilation convolution based global context pooling module to capture the global context information from the most deep feature layer of the backbone feature extraction network. Then, a global context guiding module is introduced to add the global context information into different feature refining stages for guiding the feature refining process. In addition, by considering that the defocus blur is sensitive to image scales, we add a deep features guided fusion module to integrate the outputs of different stages for generating the final score map. Extensive experiments with ablation studies on two commonly used datasets are carried out to validate the superiority of our proposed network when compared with other 11 state-of-the-art methods in terms of both efficiency and accuracy.}
}`
