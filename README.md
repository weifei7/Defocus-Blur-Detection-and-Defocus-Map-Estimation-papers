# Defocus-Blur-Detection-and-Defocus-Map-Estimation-papers

A list of deep learning based defocus blur detection and defocus map estimation papers.

I would appreciate it if you have any suggestions, and please contact me ( [*@Email*](fwei_mail@163.com) ).

****
## Defocus Blur Detection
### 1. A Local Metric for Defocus Blur Detection Based on CNN Feature Learning. [[paper]](https://ieeexplore.ieee.org/document/8537933)

Published in: **IEEE Transactions on Image Processing**

Date of Publication: **2018**

```
@ARTICLE{8537933,
  author={Zeng, Kai and Wang, Yaonan and Mao, Jianxu and Liu, Junyang and Peng, Weixing and Chen, Nankai},  
  journal={IEEE Transactions on Image Processing},   
  title={A Local Metric for Defocus Blur Detection Based on CNN Feature Learning},   
  year={2019},  
  volume={28},  
  number={5},  
  pages={2107-2115},  
  abstract={Defocus blur detection is an important and challenging task in computer vision and digital imaging fields. Previous work on defocus blur detection has put a lot of effort into designing local sharpness metric maps. This paper presents a simple yet effective method to automatically obtain the local metric map for defocus blur detection, which based on the feature learning of multiple convolutional neural networks (ConvNets). The ConvNets automatically learn the most locally relevant features at the super-pixel level of the image in a supervised manner. By extracting convolution kernels from the trained neural network structures and processing it with principal component analysis, we can automatically obtain the local sharpness metric by reshaping the principal component vector. Meanwhile, an effective iterative updating mechanism is proposed to refine the defocus blur detection result from coarse to fine by exploiting the intrinsic peculiarity of the hyperbolic tangent function. The experimental results demonstrate that our proposed method consistently performed better than the previous state-of-the-art methods.},  
  keywords={},  
  doi={10.1109/TIP.2018.2881830},  
  ISSN={1941-0042},  
  month={May},
  }
```
### 2. Defocus Blur Detection via Multi-stream Bottom-Top-Bottom Fully Convolutional Network. [[paper]](https://ieeexplore.ieee.org/document/8578423)

Published in: **IEEE Transactions on Image Processing**

Date of Publication: **2018**

```
@INPROCEEDINGS{8578423,
  author={Zhao, Wenda and Zhao, Fan and Wang, Dong and Lu, Huchuan},
  booktitle={2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition},title={Defocus Blur Detection via Multi-stream Bottom-Top-Bottom Fully Convolutional Network},
  year={2018},  
  volume={},  
  number={},  
  pages={3080-3088},  
  abstract={Defocus blur detection (DBD) is the separation of in-focus and out-of-focus regions in an image. This process has been paid considerable attention because of its remarkable potential applications. Accurate differentiation of homogeneous regions and detection of low-contrast focal regions, as well as suppression of background clutter, are challenges associated with DBD. To address these issues, we propose a multi-stream bottom-top-bottom fully convolutional network (BTBNet), which is the first attempt to develop an end-to-end deep network for DBD. First, we develop a fully convolutional BTBNet to integrate low-level cues and high-level semantic information. Then, considering that the degree of defocus blur is sensitive to scales, we propose multi-stream BTBNets that handle input images with different scales to improve the performance of DBD. Finally, we design a fusion and recurrent reconstruction network to recurrently refine the preceding blur detection maps. To promote further study and evaluation of the DBD models, we construct a new database of 500 challenging images and their pixel-wise defocus blur annotations. Experimental results on the existing and our new datasets demonstrate that the proposed method achieves significantly better performance than other state-of-the-art algorithms.},  
  keywords={},  
  doi={10.1109/CVPR.2018.00325},  
  ISSN={2575-7075},  
  month={June}
}
```
### 3. Segmentation of Defocus Blur using Local Triplicate Co-Occurrence Patterns (LTCoP). [[paper]](https://ieeexplore.ieee.org/document/9024808)

Published in: **2019 13th International Conference on Mathematics, Actuarial Science, Computer Science and Statistics (MACS)Processing**

Date of Publication: **2019**

```
@INPROCEEDINGS{9024808,
  author={Khan, Awais and Irtaza, Syed Aun and Javed, Ali and Khan, Muhammad Ammar},  
  booktitle={2019 13th International Conference on Mathematics, Actuarial Science, Computer Science and Statistics (MACS)},   
  title={Segmentation of Defocus Blur using Local Triplicate Co-Occurrence Patterns (LTCoP)},   
  year={2019},  
  volume={},  
  number={},  
  pages={1-6},  
  abstract={Many digital images contain blurred regions which are caused by motion or defocus. The defocus blur reduces the contrast and sharpness detail of the image. Automatic blur detection and segmentation is an important and challenging task in the field of Computer vision “e.g. object recognition and scene interpretation” that requires the extraction and processing of large amounts of data from sharp areas of the image. Therefore, the sharp and blur areas must be segmented separately to assure that the information is extracted from the sharp regions. The existing techniques on blur detection and segmentation have taken a lot of effort and time to design metric maps of local clarity. Furthermore, these methods have various limitations “i.e. low accuracy rate in noisy images, detecting blurred smooth and sharp smooth regions, and high execution cost”. Therefore, there is a dire necessity to propose a method for the detection and segmentation of defocus blur robust to the aforementioned limitations. In this paper, we present a novel defocus blur detection and segmentation algorithm, “Local Triplicate Co-occurrence Patterns” (LTCoP) for the separation of in-focus and out-of-focus regions. It is observed that the fusion of extracted higher and lower patterns of LTCoP produces far better results than the others. To test the effectiveness of our algorithm, the proposed method is compared with several state-of-the-art techniques over a large number of sample images. The experimental results show that the proposed technique obtains comparative results with state-of-the-art methods and offers a significant high-speed advantage over them. Therefore, we argue that the proposed method can reliably be used for defocus blur detection and segmentation in high-density images.},  
  keywords={},  
  doi={10.1109/MACS48846.2019.9024808},  
  ISSN={},  
  month={Dec},
}
```
### 4. DeFusionNET: Defocus Blur Detection via Recurrently Fusing and Refining Multi-Scale Deep Features. [[paper]](https://ieeexplore.ieee.org/document/8953847)

Published in: **2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)**

Date of Publication: **2019**

```
@INPROCEEDINGS{8953847,
  author={Tang, Chang and Zhu, Xinzhong and Liu, Xinwang and Wang, Lizhe and Zomaya, Albert},  
  booktitle={2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},   
  title={DeFusionNET: Defocus Blur Detection via Recurrently Fusing and Refining Multi-Scale Deep Features},   
  year={2019},  
  volume={},  
  number={},  
  pages={2695-2704},  
  abstract={Defocus blur detection aims to detect out-of-focus regions from an image. Although attracting more and more attention due to its widespread applications, defocus blur detection still confronts several challenges such as the interference of background clutter, sensitivity to scales and missing boundary details of defocus blur regions. To deal with these issues, we propose a deep neural network which recurrently fuses and refines multi-scale deep features (DeFusionNet) for defocus blur detection. We firstly utilize a fully convolutional network to extract multi-scale deep features. The features from bottom layers are able to capture rich low-level features for details preservation, while the features from top layers can characterize the semantic information to locate blur regions. These features from different layers are fused as shallow features and semantic features, respectively. After that, the fused shallow features are propagated to top layers for refining the fine details of detected defocus blur regions, and the fused semantic features are propagated to bottom layers to assist in better locating the defocus regions. The feature fusing and refining are carried out in a recurrent manner. Also, we finally fuse the output of each layer at the last recurrent step to obtain the final defocus blur map by considering the sensitivity to scales of the defocus degree. Experiments on two commonly used defocus blur detection benchmark datasets are conducted to demonstrate the superority of DeFusionNet when compared with other 10 competitors. Code and more results can be found at: http://tangchang.net.},  
  keywords={},  
  doi={10.1109/CVPR.2019.00281},  
  ISSN={2575-7075},  
  month={June},
}
```
### 5. Defocus Blur Detection via Multi-Stream Bottom-Top-Bottom Network. [[paper]](https://ieeexplore.ieee.org/document/8673588)

Published in: **IEEE Transactions on Pattern Analysis and Machine Intelligence**

Date of Publication: **2020**

```
@ARTICLE{8673588,
  author={Zhao, Wenda and Zhao, Fan and Wang, Dong and Lu, Huchuan},  
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},   
  title={Defocus Blur Detection via Multi-Stream Bottom-Top-Bottom Network},   
  year={2020},  
  volume={42},  
  number={8},  
  pages={1884-1897},  
  abstract={Defocus blur detection (DBD) is aimed to estimate the probability of each pixel being in-focus or out-of-focus. This process has been paid considerable attention due to its remarkable potential applications. Accurate differentiation of homogeneous regions and detection of low-contrast focal regions, as well as suppression of background clutter, are challenges associated with DBD. To address these issues, we propose a multi-stream bottom-top-bottom fully convolutional network (BTBNet), which is the first attempt to develop an end-to-end deep network to solve the DBD problems. First, we develop a fully convolutional BTBNet to gradually integrate nearby feature levels of bottom to top and top to bottom. Then, considering that the degree of defocus blur is sensitive to scales, we propose multi-stream BTBNets that handle input images with different scales to improve the performance of DBD. Finally, a cascaded DBD map residual learning architecture is designed to gradually restore finer structures from the small scale to the large scale. To promote further study and evaluation of the DBD models, we construct a new database of 1100 challenging images and their pixel-wise defocus blur annotations. Experimental results on the existing and our new datasets demonstrate that the proposed method achieves significantly better performance than other state-of-the-art algorithms.},  
  keywords={},  
  doi={10.1109/TPAMI.2019.2906588},  
  ISSN={1939-3539},  
  month={Aug},
}
```
### 6. Deep Direction-Context-Inspiration Network for Defocus Region Detection in Natural Images. [[paper]](https://ieeexplore.ieee.org/document/8713543)

Published in: **IEEE Access**

Date of Publication: **2019**

```
@ARTICLE{8713543,
  author={Zhao, Fan and Wang, Haipeng and Zhao, Wenda},  
  journal={IEEE Access},   
  title={Deep Direction-Context-Inspiration Network for Defocus Region Detection in Natural Images},   
  year={2019},  
  volume={7},  
  number={},  
  pages={64737-64743},  
  abstract={Defocus region detection (DRD) problem aims to assign per-pixel predictions of focus clear areas and defocus blur areas. One of the challenges in this problem is to accurately detect the boundary of the transition region between the focus and defocus regions. To address this issue, the paper proposes a direction-context-inspiration network (DCINet), which can take advantage of the directional context effectively. First, we extract directional context by recurrent neural networks initialized with the identity matrix (IRNN) to weight the feature maps and integrate them in the two-group integration method, which can produce the coarse DRD maps. Second, the maps are level-integrated with the source image guiding and the coarse maps are refined gradually. The overall DCINet can integrate low-level details and high-level semantics efficiently. The Experimental results demonstrate that the network can detect the boundary of the transition region precisely, achieving the state-of-the-art performance.},  
  keywords={},  
  doi={10.1109/ACCESS.2019.2916332},  
  ISSN={2169-3536},  
  month={},
}
```
### 7. Enhancing Diversity of Defocus Blur Detectors via Cross-Ensemble Network. [[paper]](https://ieeexplore.ieee.org/document/8953856)

Published in: **2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)**

Date of Publication: **2019**

```
@INPROCEEDINGS{8953856,
  author={Zhao, Wenda and Zheng, Bowen and Lin, Qiuhua and Lu, Huchuan},  
  booktitle={2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},   
  title={Enhancing Diversity of Defocus Blur Detectors via Cross-Ensemble Network},   
  year={2019},  
  volume={},  
  number={},  
  pages={8897-8905},  
  abstract={Defocus blur detection (DBD) is a fundamental yet challenging topic, since the homogeneous region is obscure and the transition from the focused area to the unfocused region is gradual. Recent DBD methods make progress through exploring deeper or wider networks with the expense of high memory and computation. In this paper, we propose a novel learning strategy by breaking DBD problem into multiple smaller defocus blur detectors and thus estimate errors can cancel out each other. Our focus is the diversity enhancement via cross-ensemble network. Specifically, we design an end-to-end network composed of two logical parts: feature extractor network (FENet) and defocus blur detector cross-ensemble network (DBD-CENet). FENet is constructed to extract low-level features. Then the features are fed into DBD-CENet containing two parallel-branches for learning two groups of defocus blur detectors. For each individual, we design cross-negative and self-negative correlations and an error function to enhance ensemble diversity and balance individual accuracy. Finally, the multiple defocus blur detectors are combined with a uniformly weighted average to obtain the final DBD map. Experimental results indicate the superiority of our method in terms of accuracy and speed when compared with several state-of-the-art methods.},  
  keywords={},  
  doi={10.1109/CVPR.2019.00911},  
  ISSN={2575-7075},  
  month={June},
}
```
### 8. Accurate and Fast Blur Detection Using a Pyramid M-Shaped Deep Neural Network. [[paper]](https://ieeexplore.ieee.org/document/8755854)

Published in: **IEEE Access**

Date of Publication: **2019**

```
@ARTICLE{8755854,
  author={Wang, Xuewei and Zhang, Shulin and Liang, Xiao and Zhou, Hongjun and Zheng, Jinjin and Sun, Mingzhai},  
  journal={IEEE Access},   
  title={Accurate and Fast Blur Detection Using a Pyramid M-Shaped Deep Neural Network},   
  year={2019},  
  volume={7},  
  number={},  
  pages={86611-86624},  
  abstract={Blur detection is aimed at estimating the probability of each pixel being blurred or non-blurred in an image affected by motion or defocus blur. This task has gained considerable attention due to its promising application fields in computer vision. Accurate differentiation of anomalous regions (including the sharp but homogeneous regions and pseudo-sharp backgrounds) and motion-blurred regions are main challenges in blur detection, in which both conventional and recently developed blur detection methods have limited performance and low time efficiency. To address these issues, this paper develops an accurate and fast blur detection method for both motion and defocus blur using a new end-to-end deep neural network. First, a novel multi-input multi-loss encoder-decoder network (M-shaped) is proposed to learn rich hierarchical representations related to blur. Then, to resolve the problem shows that blur degree is susceptible to scales, we construct a pyramid ensemble model (PM-Net) consisting of different scales of M-shaped subnets and a unified fusion layer. The experiments demonstrate that the proposed PM-Net can accurately handle those challenging scenarios with anomalous regions for both defocus and motion blur. Our method performs better than previous state-of-the-art methods. It achieves the $F_{1}$ -score of 0.893 for only defocus blur and 0.884 for joint motion and defocus blur, both of which significantly surpass previous methods on the benchmark BDD dataset. We also test our PM-Net on another public CDD dataset composed of challenging defocused images. The proposed method also outperforms other published methods with an $F_{1}$ -score of 0.885. In addition, our proposed method is hundreds of times faster (millisecond) than other state-of-the-art methods (second). Moreover, our experiments also demonstrate that the PM-Net is robust to noise and has a good generalization property.},  
  keywords={},  
  doi={10.1109/ACCESS.2019.2926747},  
  ISSN={2169-3536},  
  month={},
}
```
### 9. Defocus blur detection via edge pixel DCT feature of local patches. [[paper]](https://www.sciencedirect.com/science/article/abs/pii/S0165168420302139)

Published in: **Signal Processing**

Date of Publication: **2020**

```
@article{MA2020107670,
  title = {Defocus blur detection via edge pixel DCT feature of local patches},
  journal = {Signal Processing},
  volume = {176},
  pages = {107670},
  year = {2020},
  issn = {0165-1684},
  doi = {https://doi.org/10.1016/j.sigpro.2020.107670},
  url = {https://www.sciencedirect.com/science/article/pii/S0165168420302139},
  author = {Ming Ma and Wei Lu and Wenjing Lyu},
  keywords = {Defocus blur region segementation, Sub-band fusion, Multi-scale fusion, Image matting},
  abstract = {In common natural image blur, objects that not lie in the focal length of a digital camera generate defocus areas in the photographed image. In this paper, we propose a novel edge-based method for spatially varying defocus blur detection based on reblurred DCT coefficients ratios of the corresponding local patches. This method selects appropriate local reblur scales while detecting the edge points to deal with the problem of different blur degree and texture richness of the local image blocks. A sub-band fusion method of DCT coefficients is proposed to expand the difference between DCT features of in-focus and out-of-focus regions. Edge points blur maps are computed in multi-scale and multi-orientation image windows and more blur points are added to initialize sparse blur maps, finally Matting Laplacian method is used along with multi-scale fusion algorithm to obtain a more accurate blur segmentation. Experimental results present the proposed method has strong advantages in image detail processing and outperforms state-of-the-art methods for blur detection.}
}
```
### 10. BR2Net: Defocus Blur Detection Via a Bidirectional Channel Attention Residual Refining Network. [[paper]](https://ieeexplore.ieee.org/document/9057632)

Published in: **IEEE Transactions on Multimedia**

Date of Publication: **2020**

```
@ARTICLE{9057632,
  author={Tang, Chang and Liu, Xinwang and An, Shan and Wang, Pichao},  
  journal={IEEE Transactions on Multimedia},   
  title={BR$^2$Net: Defocus Blur Detection Via a Bidirectional Channel Attention Residual Refining Network},   
  year={2021},  
  volume={23},  
  number={},  
  pages={624-635},  
  abstract={Due to the remarkable potential applications, defocus blur detection, which aims to separate blurry regions from an image, has attracted much attention. Although significant progress has been made by many methods, there are still various challenges that hinder the results, e.g., confusing background areas, sensitivity to the scale and missing the boundary details of the defocus blur regions. To solve these issues, in this paper, we propose a deep convolutional neural network (CNN) for defocus blur detection via a Bi-directional Residual Refining network (BR2Net). Specifically, a residual learning and refining module (RLRM) is designed to correct the prediction errors in the intermediate defocus blur map. Then, we develop a bidirectional residual feature refining network with two branches by embedding multiple RLRMs into it for recurrently combining and refining the residual features. One branch of the network refines the residual features from the shallow layers to the deep layers, and the other branch refines the residual features from the deep layers to the shallow layers. In such a manner, both the low-level spatial details and highlevel semantic information can be encoded step by step in two directions to suppress background clutter and enhance the detected region details. The outputs of the two branches are fused to generate the final results. In addition, with the observation that different feature channels have different extents of discrimination for detecting blurred regions, we add a channel attention module to each feature extraction layer to select more discriminative features for residual learning. To promote further research on defocus blur detection, we create a new dataset with various challenging images and manually annotate their corresponding pixelwise ground truths. The proposed network is validated on two commonly used defocus blur detection datasets and our newly collected dataset by comparing it with 10 other state-of-the-art methods. Extensive experiments with ablation studies demonstrate that BR2Net consistently and significantly outperforms the competitors in terms of both the efficiency and accuracy.},  
  keywords={},  
  doi={10.1109/TMM.2020.2985541},  
  ISSN={1941-0077},  
  month={},
}
```
### 11. DeFusionNET: Defocus Blur Detection via Recurrently Fusing and Refining Discriminative Multi-Scale Deep Features. [[paper]](https://ieeexplore.ieee.org/document/9161280)

Published in: **IEEE Transactions on Pattern Analysis and Machine Intelligence**

Date of Publication: **2020**

```
@ARTICLE{9161280,
  author={Tang, Chang and Liu, Xinwang and Zheng, Xiao and Li, Wanqing and Xiong, Jian and Wang, Lizhe and Zomaya, Albert Y. and Longo, Antonella},  
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},   
  title={DeFusionNET: Defocus Blur Detection via Recurrently Fusing and Refining Discriminative Multi-Scale Deep Features},   
  year={2022},  
  volume={44},  
  number={2},  
  pages={955-968},  
  abstract={Albeit great success has been achieved in image defocus blur detection, there are still several unsolved challenges, e.g., interference of background clutter, scale sensitivity and missing boundary details of blur regions. To deal with these issues, we propose a deep neural network which recurrently fuses and refines multi-scale deep features (DeFusionNet) for defocus blur detection. We first fuse the features from different layers of FCN as shallow features and semantic features, respectively. Then, the fused shallow features are propagated to deep layers for refining the details of detected defocus blur regions, and the fused semantic features are propagated to shallow layers to assist in better locating blur regions. The fusion and refinement are carried out recurrently. In order to narrow the gap between low-level and high-level features, we embed a feature adaptation module before feature propagating to exploit the complementary information as well as reduce the contradictory response of different feature layers. Since different feature channels are with different extents of discrimination for detecting blur regions, we design a channel attention module to select discriminative features for feature refinement. Finally, the output of each layer at last recurrent step are fused to obtain the final result. We collect a new dataset consists of various challenging images and their pixel-wise annotations for promoting further study. Extensive experiments on two commonly used datasets and our newly collected one are conducted to demonstrate both the efficacy and efficiency of DeFusionNet.},  
  keywords={},  
  doi={10.1109/TPAMI.2020.3014629},  
  ISSN={1939-3539},  
  month={Feb},
}
```
### 12. Defocus Blur Detection by Fusing Multiscale Deep Features With Conv-LSTM. [[paper]](https://ieeexplore.ieee.org/document/9097895)

Published in: **IEEE Access**

Date of Publication: **2020**

```
@ARTICLE{9097895,
  author={Heng, Hongjun and Ye, Hebin and Huang, Rui},  
  journal={IEEE Access},   
  title={Defocus Blur Detection by Fusing Multiscale Deep Features With Conv-LSTM},   
  year={2020},  
  volume={8},  
  number={},  
  pages={97279-97288},  
  abstract={Defocus blur detection aiming at distinguishing out-of-focus blur and sharpness has attracted considerable attention in computer vision. The present blur detectors suffer from scale ambiguity, which results in blur boundaries and low accuracy in blur detection. In this paper, we propose a defocus blur detector to address these problems by integrating multiscale deep features with Conv-LSTM. There are two strategies to extract multiscale features. The first one is extracting features from images with different sizes. The second one is extracting features from multiple convolutional layers by a single image. Our method employs both strategies, i.e., extracting multiscale convolutional features from same image with different sizes. The features extracted from different sized images at the corresponding convolutional layers are fused to generate more robust representations. We use Conv-LSTMs to integrate the fused features gradually from top-to-bottom layers, and to generate multiscale blur estimations. The experiments on CUHK and DUT datasets demonstrate that our method is superior to the state-of-the-art blur detectors.},  
  keywords={},  
  doi={10.1109/ACCESS.2020.2996200},  
  ISSN={2169-3536},  
  month={},
}
```
### 13. MultiANet: a Multi-Attention Network for Defocus Blur Detection. [[paper]](https://ieeexplore.ieee.org/document/9287072)

Published in: **2020 IEEE 22nd International Workshop on Multimedia Signal Processing (MMSP)**

Date of Publication: **2020**

```
@INPROCEEDINGS{9287072,
  author={Jiang, Zeyu and Xu, Xun and Zhang, Chao and Zhu, Ce},  
  booktitle={2020 IEEE 22nd International Workshop on Multimedia Signal Processing (MMSP)},   
  title={MultiANet: a Multi-Attention Network for Defocus Blur Detection},   
  year={2020},  
  volume={},  
  number={},  
  pages={1-6},  
  abstract={Defocus blur detection is a challenging task because of obscure homogenous regions and interferences of background clutter. Most existing deep learning-based methods mainly focus on building wider or deeper network to capture multi-level features, neglecting to extract the feature relationships of intermediate layers, thus hindering the discriminative ability of network. Moreover, fusing features at different levels have been demonstrated to be effective. However, direct integrating without distinction is not optimal because low-level features focus on fine details only and could be distracted by background clutters. To address these issues, we propose the Multi-Attention Network for stronger discriminative learning and spatial guided low-level feature learning. Specifically, a channel-wise attention module is applied to both high-level and low-level feature maps to capture channel-wise global dependencies. In addition, a spatial attention module is employed to low-level features maps to emphasize effective detailed information. Experimental results show the performance of our network is superior to the state-of-the-art algorithms.},  
  keywords={},  
  doi={10.1109/MMSP48831.2020.9287072},  
  ISSN={2473-3628},  
  month={Sep.},
}
```
### 14. [Global context guided hierarchically residual feature refinement network for defocus blur detection](https://www.sciencedirect.com/science/article/abs/pii/S0165168421000359)

Published in: **Signal Processing**

Date of Publication: **2021**

```
@article{ZHAI2021107996,
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
}
```
### 15. Defocus Blur Detection via Boosting Diversity of Deep Ensemble Networks. [[paper]](https://ieeexplore.ieee.org/document/9447919)

Published in: **IEEE Transactions on Image Processing**

Date of Publication: **2021**

```
@ARTICLE{9447919,
  author={Zhao, Wenda and Hou, Xueqing and He, You and Lu, Huchuan},  
  journal={IEEE Transactions on Image Processing},   
  title={Defocus Blur Detection via Boosting Diversity of Deep Ensemble Networks},   
  year={2021},  
  volume={30},  
  number={},  
  pages={5426-5438},  
  abstract={Existing defocus blur detection (DBD) methods usually explore multi-scale and multi-level features to improve performance. However, defocus blur regions normally have incomplete semantic information, which will reduce DBD's performance if it can't be used properly. In this paper, we address the above problem by exploring deep ensemble networks, where we boost diversity of defocus blur detectors to force the network to generate diverse results that some rely more on high-level semantic information while some ones rely more on low-level information. Then, diverse result ensemble makes detection errors cancel out each other. Specifically, we propose two deep ensemble networks (e.g., adaptive ensemble network (AENet) and encoder-feature ensemble network (EFENet)), which focus on boosting diversity while costing less computation. AENet constructs different light-weight sequential adapters for one backbone network to generate diverse results without introducing too many parameters and computation. AENet is optimized only by the self- negative correlation loss. On the other hand, we propose EFENet by exploring the diversity of multiple encoded features and ensemble strategies of features (e.g., group-channel uniformly weighted average ensemble and self-gate weighted ensemble). Diversity is represented by encoded features with less parameters, and a simple mean squared error loss can achieve the superior performance. Experimental results demonstrate the superiority over the state-of-the-arts in terms of accuracy and speed. Codes and models are available at: https://github.com/wdzhao123/DENets.},  
  keywords={},  
  doi={10.1109/TIP.2021.3084101},  
  ISSN={1941-0042},  
  month={},
}
```
### 16. A Novel Defocused Image Segmentation Method Based on PCNN and LBP. [[paper]](https://ieeexplore.ieee.org/document/9444355)

Published in: **IEEE Access**

Date of Publication: **2021**

```
@ARTICLE{9444355,
  author={Basar, Sadia and Ali, Mushtaq and Ochoa-Ruiz, Gilberto and Waheed, Abdul and Rodriguez-Hernandez, Gerardo and Zareei, Mahdi},  
  journal={IEEE Access},   
  title={A Novel Defocused Image Segmentation Method Based on PCNN and LBP},   
  year={2021},  
  volume={9},  
  number={},  
  pages={87219-87240},  
  abstract={The defocus blur concept adds an artistic effect and enables an enhancement in the visualization of image scenery. Moreover, some specialized computer vision fields, such as object recognition or scene restoration enhancement, might need to perform segmentation to separate the blurred and non-blurred regions in partially blurred images. This study proposes a sharpness measure comprised of a Local Binary Pattern (LBP) descriptor and Pulse Coupled Neural Network (PCNN) component used to implement a robust approach for segmenting in-focus regions from out of focus sections in the scene. The proposed approach is very robust in the sense that the parameters of the model can be modified to accommodate different settings. The presented metric exploits the fact that, in general, local patches of the image in blurry regions have less prominent LBP descriptors than non-blurry regions. The proposed approach combines this sharpness measure with the PCNN algorithm; the images are segmented along with clear regions and edges of segmented objects. The proposed approach has been tested on a dataset comprised of 1000 defocused images with eight state-of-the-art methods. Based on a set of evaluation metrics, i.e., precision, recall, and F1-Measure, the results show that the proposed algorithm outperforms previous works in terms of prominent accuracy and efficiency improvement. The proposed approach also uses other evaluation parameters, i.e., Accuracy, Matthews Correlation Coefficient (MCC), Dice Similarity Coefficient (DSC), and Specificity, to assess better the results obtained by our proposal. Moreover, we adopted a fuzzy logic ranking scheme inspired by the Evaluation Based on Distance from Average Solution (EDAS) technique to interpret the defocus segmentation integrity. The experimental outputs illustrate that the proposed approach outperforms the referenced methods by optimizing the segmentation quality and reducing the computational complexity.},  
  keywords={},  
  doi={10.1109/ACCESS.2021.3084905},  
  ISSN={2169-3536},  
  month={},
}
```
### 17. Layer-Output Guided Complementary Attention Learning for Image Defocus Blur Detection. [[paper]](https://ieeexplore.ieee.org/document/9380693)

Published in: **IEEE Transactions on Image Processing**

Date of Publication: **2021**

```
@ARTICLE{9380693,
  author={Li, Jinxing and Fan, Dandan and Yang, Lingxiao and Gu, Shuhang and Lu, Guangming and Xu, Yong and Zhang, David},  
  journal={IEEE Transactions on Image Processing},   
  title={Layer-Output Guided Complementary Attention Learning for Image Defocus Blur Detection},   
  year={2021},  
  volume={30},  
  number={},  
  pages={3748-3763},  
  abstract={Defocus blur detection (DBD), which has been widely applied to various fields, aims to detect the out-of-focus or in-focus pixels from a single image. Despite the fact that the deep learning based methods applied to DBD have outperformed the hand-crafted feature based methods, the performance cannot still meet our requirement. In this paper, a novel network is established for DBD. Unlike existing methods which only learn the projection from the in-focus part to the ground-truth, both in-focus and out-of-focus pixels, which are completely and symmetrically complementary, are taken into account. Specifically, two symmetric branches are designed to jointly estimate the probability of focus and defocus pixels, respectively. Due to their complementary constraint, each layer in a branch is affected by an attention obtained from another branch, effectively learning the detailed information which may be ignored in one branch. The feature maps from these two branches are then passed through a unique fusion block to simultaneously get the two-channel output measured by a complementary loss. Additionally, instead of estimating only one binary map from a specific layer, each layer is encouraged to estimate the ground truth to guide the binary map estimation in its linked shallower layer followed by a top-to-bottom combination strategy, gradually exploiting the global and local information. Experimental results on released datasets demonstrate that our proposed method remarkably outperforms state-of-the-art algorithms.},  
  keywords={},  
  doi={10.1109/TIP.2021.3065171},  
  ISSN={1941-0042},  
  month={},
}
```
### 18. Self-generated Defocus Blur Detection via Dual Adversarial Discriminators. [[paper]](https://ieeexplore.ieee.org/document/9578013)

Published in: **021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)**

Date of Publication: **2021**

```
@INPROCEEDINGS{9578013,
  author={Zhao, Wenda and Shang, Cai and Lu, Huchuan},  
  booktitle={2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},   
  title={Self-generated Defocus Blur Detection via Dual Adversarial Discriminators},   
  year={2021},  
  volume={},  
  number={},  
  pages={6929-6938},  
  abstract={Although existing fully-supervised defocus blur detection (DBD) models significantly improve performance, training such deep models requires abundant pixel-level manual annotation, which is highly time-consuming and error-prone. Addressing this issue, this paper makes an effort to train a deep DBD model without using any pixel-level annotation. The core insight is that a defocus blur region/focused clear area can be arbitrarily pasted to a given realistic full blurred image/full clear image without affecting the judgment of the full blurred image/full clear image. Specifically, we train a generator G in an adversarial manner against dual discriminators Dc and Db. G learns to produce a DBD mask that generates a composite clear image and a composite blurred image through copying the focused area and unfocused region from corresponding source image to another full clear image and full blurred image. Then, Dc and Db can not distinguish them from realistic full clear image and full blurred image simultaneously, achieving a self-generated DBD by an implicit manner to define what a defocus blur area is. Besides, we propose a bilateral triplet-excavating constraint to avoid the degenerate problem caused by the case one discriminator defeats the other one. Comprehensive experiments on two widely-used DBD datasets demonstrate the superiority of the proposed approach. Source codes are available at: https://github.com/shangcai1/SG.},  
  keywords={},  
  doi={10.1109/CVPR46437.2021.00686},  
  ISSN={2575-7075},  
  month={June},
}
```
### 19. Image-Scale-Symmetric Cooperative Network for Defocus Blur Detection. [[paper]](https://ieeexplore.ieee.org/document/9476010)

Published in: **IEEE Transactions on Circuits and Systems for Video Technology**

Date of Publication: **2021**

```
@ARTICLE{9476010,
  author={Zhao, Fan and Lu, Huimin and Zhao, Wenda and Yao, Libo},  
  journal={IEEE Transactions on Circuits and Systems for Video Technology},   
  title={Image-Scale-Symmetric Cooperative Network for Defocus Blur Detection},   
  year={2021},  
  volume={},  
  number={},  
  pages={1-1},  
  abstract={Defocus blur detection (DBD) for natural images is a challenging vision task especially in the presence of homogeneous regions and gradual boundaries. In this paper, we propose a novel image-scale-symmetric cooperative network (IS2CNet) for DBD. On one hand, in the process of image scales from large to small, IS2CNet gradually spreads the recept of image content. Thus, the homogeneous region detection map can be optimized gradually. On the other hand, in the process of image scales from small to large, IS2CNet gradually feels the high-resolution image content, thereby gradually refining transition region detection. In addition, we propose a hierarchical feature integration and bi-directional delivering mechanism to transfer the hierarchical feature of previous image scale network to the input and tail of the current image scale network for guiding the current image scale network to better learn the residual. The proposed approach achieves state-of-the-art performance on existing datasets. Codes and results are available at: https://github.com/wdzhao123/IS2CNet.},  
  keywords={},  
  doi={10.1109/TCSVT.2021.3095347},  
  ISSN={1558-2205},  
  month={},
}
```
****
## Defocus Map Estimation
### 1. Spatially variant defocus blur map estimation and deblurring from a single image. [[paper]](https://www.sciencedirect.com/science/article/abs/pii/S1047320316000031)

Published in: **Journal of Visual Communication and Image Representation**

Date of Publication: **2016**

```
@article{ZHANG2016257,
title = {Spatially variant defocus blur map estimation and deblurring from a single image},
journal = {Journal of Visual Communication and Image Representation},
volume = {35},
pages = {257-264},
year = {2016},
issn = {1047-3203},
doi = {https://doi.org/10.1016/j.jvcir.2016.01.002},
url = {https://www.sciencedirect.com/science/article/pii/S1047320316000031},
author = {Xinxin Zhang and Ronggang Wang and Xiubao Jiang and Wenmin Wang and Wen Gao},
keywords = {Spatially variant blur, Edge information, Defocus image deblurring, Image deblurring, Blur map estimation, Ringing artifacts removal, Image restoration, Non-blind deconvolution},
abstract = {In this paper, we propose a single image deblurring algorithm to remove spatially variant defocus blur based on the estimated blur map. Firstly, we estimate the blur map from a single image by utilizing the edge information and K nearest neighbors (KNN) matting interpolation. Secondly, the local kernels are derived by segmenting the blur map according to the blur amount of local regions and image contours. Thirdly, we adopt a BM3D-based non-blind deconvolution algorithm to restore the latent image. Finally, ringing artifacts and noise are detected and removed, to obtain a high quality in-focus image. Experimental results on real defocus blurred images demonstrate that our proposed algorithm outperforms some state-of-the-art approaches.}
}
```
### 2. Defocus Map Estimation From a Single Image Based on Two-Parameter Defocus Model. [[paper]](https://ieeexplore.ieee.org/document/7589980)

Published in: **IEEE Transactions on Image Processing**

Date of Publication: **2016**

```
@ARTICLE{7589980,
  author={Liu, Shaojun and Zhou, Fei and Liao, Qingmin},
  journal={IEEE Transactions on Image Processing}, 
  title={Defocus Map Estimation From a Single Image Based on Two-Parameter Defocus Model}, 
  year={2016},
  volume={25},
  number={12},
  pages={5943-5956},
  abstract={Defocus map estimation (DME) is highly important in many computer vision applications. Nearly, all existing approaches for DME from a single image are based on a one-parameter defocus model, which does not allow for the variation of depth over edges. In this paper, a novel two-parameter model of defocused edges is proposed for DME from a single image. We can estimate the defocus amounts for each side of the edges through this proposed model, and the confidence that the edge is a pattern edge, where the depth remains the same over the edge, can be generated. Then, we modify the TV-L1 algorithm for structure-texture decomposition by taking advantage of this confidence to eliminate pattern edges while preserving structural ones. Finally, the defocus amounts estimated at the edge positions are used as initial values, and the structure component is employed as a guidance in the following Laplacian matting procedure to avoid the influence of pattern edges on the final defocus map. Experiment results show that the proposed method can effectively eliminate the influence of pattern edges compared with the state-of-art method. Furthermore, the estimated defocus map is feasible in applications of depth estimation and foreground/background segmentation.},
  keywords={},
  doi={10.1109/TIP.2016.2617460},
  ISSN={1941-0042},
  month={Dec},}
```
### 3. Defocus Map Detection Using a Single Image. [[paper]](https://ieeexplore.ieee.org/document/7881444)

Published in: **2016 International Conference on Computational Science and Computational Intelligence (CSCI)**

Date of Publication: **2016**

```
@INPROCEEDINGS{7881444,
  author={Andrade, Juan},
  booktitle={2016 International Conference on Computational Science and Computational Intelligence (CSCI)}, 
  title={Defocus Map Detection Using a Single Image}, 
  year={2016},
  volume={},
  number={},
  pages={777-780},
  abstract={The estimation of blurred regions is an important stage in several computer vision applications. In this paper an efficient training-free detector of local blurriness based on edge features is presented. Due to the intrinsic sparsity of edges in natural images a blur map is creating by using an approach based on the heat diffusion principle. A 2D point discrete Poisson solver is concatenated with a guided filter stage in order to create the blurring map. Experiments with images from two publicly available datasets validate the proposed method.},
  keywords={},
  doi={10.1109/CSCI.2016.0151},
  ISSN={},
  month={Dec},
}
```
### 4. Estimating Defocus Blur via Rank of Local Patches. [[paper]](https://ieeexplore.ieee.org/document/8237836)

Published in: **2017 IEEE International Conference on Computer Vision (ICCV)**

Date of Publication: **2017**

```
@INPROCEEDINGS{8237836,
  author={Xu, Guodong and Quan, Yuhui and Ji, Hui},
  booktitle={2017 IEEE International Conference on Computer Vision (ICCV)}, 
  title={Estimating Defocus Blur via Rank of Local Patches}, 
  year={2017},
  volume={},
  number={},
  pages={5381-5389},
  abstract={This paper addresses the problem of defocus map estimation from a single image. We present a fast yet effective approach to estimate the spatially varying amounts of defocus blur at edge locations, which is based on the maximum ranks of the corresponding local patches with different orientations in gradient domain. Such an approach is motivated by the theoretical analysis which reveals the connection between the rank of a local patch blurred by a defocus-blur kernel and the blur amount by the kernel. After the amounts of defocus blur at edge locations are obtained, a complete defocus map is generated by a standard propagation procedure. The proposed method is extensively evaluated on real image datasets, and the experimental results show its superior performance to existing approaches.},
  keywords={},
  doi={10.1109/ICCV.2017.574},
  ISSN={2380-7504},
  month={Oct},
}
```
### 5. A Unified Approach of Multi-scale Deep and Hand-Crafted Features for Defocus Estimation. [[paper]](https://ieeexplore.ieee.org/document/8099778)

Published in: **2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)**

Date of Publication: **2017**

```
@INPROCEEDINGS{8099778,
  author={Park, Jinsun and Tai, Yu-Wing and Cho, Donghyeon and Kweon, In So},
  booktitle={2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  title={A Unified Approach of Multi-scale Deep and Hand-Crafted Features for Defocus Estimation}, 
  year={2017},
  volume={},
  number={},
  pages={2760-2769},
  abstract={In this paper, we introduce robust and synergetic hand-crafted features and a simple but efficient deep feature from a convolutional neural network (CNN) architecture for defocus estimation. This paper systematically analyzes the effectiveness of different features, and shows how each feature can compensate for the weaknesses of other features when they are concatenated. For a full defocus map estimation, we extract image patches on strong edges sparsely, after which we use them for deep and hand-crafted feature extraction. In order to reduce the degree of patch-scale dependency, we also propose a multi-scale patch extraction strategy. A sparse defocus map is generated using a neural network classifier followed by a probability-joint bilateral filter. The final defocus map is obtained from the sparse defocus map with guidance from an edge-preserving filtered input image. Experimental results show that our algorithm is superior to state-of-the-art algorithms in terms of defocus estimation. Our work can be used for applications such as segmentation, blur magnification, all-in-focus image generation, and 3-D estimation.},
  keywords={},
  doi={10.1109/CVPR.2017.295},
  ISSN={1063-6919},
  month={July},
}
```
### 6. Edge-Based Defocus Blur Estimation With Adaptive Scale Selection. [[paper]](https://ieeexplore.ieee.org/document/8101511)

Published in: **IEEE Transactions on Image Processing**

Date of Publication: **2017**

```
@ARTICLE{8101511,
  author={Karaali, Ali and Jung, Claudio Rosito},
  journal={IEEE Transactions on Image Processing}, 
  title={Edge-Based Defocus Blur Estimation With Adaptive Scale Selection}, 
  year={2018},
  volume={27},
  number={3},
  pages={1126-1137},
  abstract={Objects that do not lie at the focal distance of a digital camera generate defocused regions in the captured image. This paper presents a new edge-based method for spatially varying defocus blur estimation using a single image based on reblurred gradient magnitudes. The proposed approach initially computes a scale-consistent edge map of the input image and selects a local reblurring scale aiming to cope with noise, edge mis-localization, and interfering edges. An initial blur estimate is computed at the detected scale-consistent edge points and a novel connected edge filter is proposed to smooth the sparse blur map based on pixel connectivity within detected edge contours. Finally, a fast guided filter is used to propagate the sparse blur map through the whole image. Experimental results show that the proposed approach presents a very good compromise between estimation error and running time when compared with the state-of-the-art methods. We also explore our blur estimation method in the context of image deblurring, and show that metrics typically used to evaluate blur estimation may not correlate as expected with the visual quality of the deblurred image.},
  keywords={},
  doi={10.1109/TIP.2017.2771563},
  ISSN={1941-0042},
  month={March},
}
```
### 7. Simultaneous blur map estimation and deblurring of a single space-variantly defocused image. [[paper]](https://ieeexplore.ieee.org/document/8077133)

Published in: **2017 Twenty-third National Conference on Communications (NCC)**

Date of Publication: **2017**

```
@INPROCEEDINGS{8077133,
  author={Narayan, Latha H. and Parida, Kranti K. and Sahay, Rajiv R.},
  booktitle={2017 Twenty-third National Conference on Communications (NCC)}, 
  title={Simultaneous blur map estimation and deblurring of a single space-variantly defocused image}, 
  year={2017},
  volume={},
  number={},
  pages={1-6},
  abstract={In this work we address the problem of blind deblurring using a single space-variantly defocused image containing text. We estimate both the all-in-focus image and the blur map corresponding to the space-variant point spread function of the finite aperture camera. Since this problem is highly ill-posed we exploit a recently proposed technique [1] to obtain an initial estimate of the space-variant blur map which is used in an MAP-MRF alternating minimization framework. We obtain analytically the gradients with respect to the unknowns and show that the proposed objective function can be successfully optimized with the steepest descent technique. Initially, we show results using the Gauss-Markov random field (GMRF) prior and then contrast its performance with the discontinuity adaptive Markov random field (DAMRF) prior. We show that details such as edges and fine details are preserved by the DAMRF regularizer. We compare the results of our algorithm with state-of-the-art techniques and provide both qualitative and quantitative evaluation.},
  keywords={},
  doi={10.1109/NCC.2017.8077133},
  ISSN={},
  month={March},
}
```
### 8. Defocus blur detection based on multiscale SVD fusion in gradient domain. [[paper]](https://www.sciencedirect.com/science/article/abs/pii/S1047320318303742)

Published in: **Journal of Visual Communication and Image Representation**

Date of Publication: **2019**

```
@article{XIAO201952,
  title = {Defocus blur detection based on multiscale SVD fusion in gradient domain},
  journal = {Journal of Visual Communication and Image Representation},
  volume = {59},
  pages = {52-61},
  year = {2019},
  issn = {1047-3203},
  doi = {https://doi.org/10.1016/j.jvcir.2018.12.048},
  url = {https://www.sciencedirect.com/science/article/pii/S1047320318303742},
  author = {Huimei Xiao and Wei Lu and Ruipeng Li and Nan Zhong and Yuileong Yeung and Junjia Chen and Fei Xue and Wei Sun},
  keywords = {Defocus blur detection, Multiscale singular value decomposition, Sub-bands, Meanshift},
  abstract = {Recently, defocus blur detection has been an extensive study, but it is still full of challenges in the blur estimation without having any prior knowledge of test image such as blur kernel, degree, or camera parameters. Inspired by the observation that the degree of defocus blur depth could be distinguished by different frequencies, a novel blur metric based on Multiscale SVD fusion (M-SVD) is proposed. The blur metric fuses different sub-bands of the selected singular values (SVs) in multiscale image windows, which could drastically reduce the chances of false positives for blur detection and overcome the difficulty that the sharp region is misjudged for a blur region because of its smooth texture. Finally, a blur map is applied on the test image combined with post-processing operation meanshift cluster to segment the blur region. Experimental results demonstrate that the proposed method can detect the defocus blur regions of test images with a satisfactory performance and outperforms the state-of-the-art methods.}
}
```
### 9. Deep Defocus Map Estimation Using Domain Adaptation. [[paper]](https://ieeexplore.ieee.org/document/8953220)

Published in: **2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)**

Date of Publication: **2019**

```
@INPROCEEDINGS{8953220,
  author={Lee, Junyong and Lee, Sungkil and Cho, Sunghyun and Lee, Seungyong},
  booktitle={2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  title={Deep Defocus Map Estimation Using Domain Adaptation}, 
  year={2019},
  volume={},
  number={},
  pages={12214-12222},
  abstract={In this paper, we propose the first end-to-end convolutional neural network (CNN) architecture, Defocus Map Estimation Network (DMENet), for spatially varying defocus map estimation. To train the network, we produce a novel depth-of-field (DOF) dataset, SYNDOF, where each image is synthetically blurred with a ground-truth depth map. Due to the synthetic nature of SYNDOF, the feature characteristics of images in SYNDOF can differ from those of real defocused photos. To address this gap, we use domain adaptation that transfers the features of real defocused photos into those of synthetically blurred ones. Our DMENet consists of four subnetworks: blur estimation, domain adaptation, content preservation, and sharpness calibration networks. The subnetworks are connected to each other and jointly trained with their corresponding supervisions in an end-to-end manner. Our method is evaluated on publicly available blur detection and blur estimation datasets and the results show the state-of-the-art performance.In this paper, we propose the first end-to-end convolutional neural network (CNN) architecture, Defocus Map Estimation Network (DMENet), for spatially varying defocus map estimation. To train the network, we produce a novel depth-of-field (DOF) dataset, SYNDOF, where each image is synthetically blurred with a ground-truth depth map. Due to the synthetic nature of SYNDOF, the feature characteristics of images in SYNDOF can differ from those of real defocused photos. To address this gap, we use domain adaptation that transfers the features of real defocused photos into those of synthetically blurred ones. Our DMENet consists of four subnetworks: blur estimation, domain adaptation, content preservation, and sharpness calibration networks. The subnetworks are connected to each other and jointly trained with their corresponding supervisions in an end-to-end manner. Our method is evaluated on publicly available blur detection and blur estimation datasets and the results show the state-of-the-art performance.},
  keywords={},
  doi={10.1109/CVPR.2019.01250},
  ISSN={2575-7075},
  month={June},
}
```
### 10. Defocus map estimation from a single image using improved likelihood feature and edge-based basis. [[paper]](https://www.sciencedirect.com/science/article/pii/S0031320320302880)

Published in: **Pattern Recognition**

Date of Publication: **2020**

```
@article{LIU2020107485,
  title = {Defocus map estimation from a single image using improved likelihood feature and edge-based basis},
  journal = {Pattern Recognition},
  volume = {107},
  pages = {107485},
  year = {2020},
  issn = {0031-3203},
  doi = {https://doi.org/10.1016/j.patcog.2020.107485},
  url = {https://www.sciencedirect.com/science/article/pii/S0031320320302880},
  author = {Shaojun Liu and Qingmin Liao and Jing-Hao Xue and Fei Zhou},
  keywords = {Defocus map estimation, Regression tree fields, Localized 2D frequency analysis},
  abstract = {Defocus map estimation (DME) is very useful in many computer vision applications and has drawn much attention in recent years. Edge-based DME methods can generate sharp defocus discontinuities but usually suffer from textures of the input image. Region-based methods are free of textures but cannot catch the defocus discontinuities very well. In this paper, we propose a DME method combining edge-based and region-based methods together to keep their respective advantages while eliminating the shortcomings. The combination is achieved via regression tree fields (RTF). In an RTF, the input feature and the linear basis are of vital importance. For our RTF, they are obtained as follows. (i) Two orthogonal gradient operators with the corresponding subsets of Gabor filters are employed in localized 2D frequency analysis to generate accurate likelihood, and the first K highest local maximums of likelihood are sent to an RTF as input feature. (ii) At the same time, the input image is processed by three edge-based methods and the results serve as the linear basis of RTF. The experiments demonstrate that the proposed method outperforms state-of-the-art DME methods. Moreover, the proposed method can be readily applied to defocused image deblurring and defocus blur detection.}
}
```
### 11. Defocus Image Deblurring Network With Defocus Map Estimation as Auxiliary Task. [[paper]](https://ieeexplore.ieee.org/document/9619959)

Published in: **IEEE Transactions on Image Processing**

Date of Publication: **2021**

```
@ARTICLE{9619959,
  author={Ma, Haoyu and Liu, Shaojun and Liao, Qingmin and Zhang, Juncheng and Xue, Jing-Hao},
  journal={IEEE Transactions on Image Processing}, 
  title={Defocus Image Deblurring Network With Defocus Map Estimation as Auxiliary Task}, 
  year={2022},
  volume={31},
  number={},
  pages={216-226},
  abstract={Different from the object motion blur, the defocus blur is caused by the limitation of the cameras’ depth of field. The defocus amount can be characterized by the parameter of point spread function and thus forms a defocus map. In this paper, we propose a new network architecture called Defocus Image Deblurring Auxiliary Learning Net (DID-ANet), which is specifically designed for single image defocus deblurring by using defocus map estimation as auxiliary task to improve the deblurring result. To facilitate the training of the network, we build a novel and large-scale dataset for single image defocus deblurring, which contains the defocus images, the defocus maps and the all-sharp images. To the best of our knowledge, the new dataset is the first large-scale defocus deblurring dataset for training deep networks. Moreover, the experimental results demonstrate that the proposed DID-ANet outperforms the state-of-the-art methods for both tasks of defocus image deblurring and defocus map estimation, both quantitatively and qualitatively. The dataset, code, and model is available on GitHub: <uri>https://github.com/xytmhy/DID-ANet-Defocus-Deblurring</uri>.},
  keywords={},
  doi={10.1109/TIP.2021.3127850},
  ISSN={1941-0042},
  month={},
}
```
### 12. Joint Depth and Defocus Estimation From a Single Image Using Physical Consistency. [[paper]](https://ieeexplore.ieee.org/document/9366926)

Published in: **IEEE Transactions on Image Processing**

Date of Publication: **2021**

```
@ARTICLE{9366926,
  author={Zhang, Anmei and Sun, Jian},
  journal={IEEE Transactions on Image Processing}, 
  title={Joint Depth and Defocus Estimation From a Single Image Using Physical Consistency}, 
  year={2021},
  volume={30},
  number={},
  pages={3419-3433},
  abstract={Estimating depth and defocus maps are two fundamental tasks in computer vision. Recently, many methods explore these two tasks separately with the help of the powerful feature learning ability of deep learning and these methods have achieved impressive progress. However, due to the difficulty in densely labeling depth and defocus on real images, these methods are mostly based on synthetic training dataset, and the performance of learned network degrades significantly on real images. In this paper, we tackle a new task that jointly estimates depth and defocus from a single image. We design a dual network with two subnets respectively for estimating depth and defocus. The network is jointly trained on synthetic dataset with a physical constraint to enforce the physical consistency between depth and defocus. Moreover, we design a simple method to label depth and defocus order on real image dataset, and design two novel metrics to measure accuracies of depth and defocus estimation on real images. Comprehensive experiments demonstrate that joint training for depth and defocus estimation using physical consistency constraint enables these two subnets to guide each other, and effectively improves their depth and defocus estimation performance on real defocused image dataset.},
  keywords={},
  doi={10.1109/TIP.2021.3061901},
  ISSN={1941-0042},
  month={},
}
```
### 13. Learning to Estimate Kernel Scale and Orientation of Defocus Blur with Asymmetric Coded Aperture. [[paper]](https://ieeexplore.ieee.org/document/9413920)

Published in: **ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)**

Date of Publication: **2021**

```
@INPROCEEDINGS{9413920,
  author={Li, Jisheng and Dai, Qi and Wen, Jiangtao},
  booktitle={ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Learning to Estimate Kernel Scale and Orientation of Defocus Blur with Asymmetric Coded Aperture}, 
  year={2021},
  volume={},
  number={},
  pages={1465-1469},
  abstract={Consistent in-focus input imagery is an essential precondition for machine vision systems to perceive the dynamic environment. A de-focus blur severely degrades the performance of vision systems. To tackle this problem, we propose a deep-learning-based framework estimating the kernel scale and orientation of the defocus blur to ad-just lens focus rapidly. Our pipeline utilizes 3D ConvNet for a variable number of input hypotheses to select the optimal slice from the input stack. We use random shuffle and Gumbel-softmax to improve network performance. We also propose to generate synthetic defocused images with various asymmetric coded apertures to facilitate training. Experiments are conducted to demonstrate the effectiveness of our framework.},
  keywords={},
  doi={10.1109/ICASSP39728.2021.9413920},
  ISSN={2379-190X},
  month={June},
}
```
### 14. Defocus Map Estimation and Deblurring from a Single Dual-Pixel Image. [[paper]](https://ieeexplore.ieee.org/document/9710489)

Published in: **2021 IEEE/CVF International Conference on Computer Vision (ICCV)**

Date of Publication: **2021**

```
@INPROCEEDINGS{9710489,
  author={Xin, Shumian and Wadhwa, Neal and Xue, Tianfan and Barron, Jonathan T. and Srinivasan, Pratul P. and Chen, Jiawen and Gkioulekas, Ioannis and Garg, Rahul},
  booktitle={2021 IEEE/CVF International Conference on Computer Vision (ICCV)}, 
  title={Defocus Map Estimation and Deblurring from a Single Dual-Pixel Image}, 
  year={2021},
  volume={},
  number={},
  pages={2208-2218},
  doi={10.1109/ICCV48922.2021.00223}
}
```
### 15. Deep Multi-Scale Feature Learning for Defocus Blur Estimation. [[paper]](https://ieeexplore.ieee.org/document/9673106)

Published in: **IEEE Transactions on Image Processing**

Date of Publication: **2022**

```
@ARTICLE{9673106,
  author={Karaali, Ali and Harte, Naomi and Jung, Claudio R.},
  journal={IEEE Transactions on Image Processing}, 
  title={Deep Multi-Scale Feature Learning for Defocus Blur Estimation}, 
  year={2022},
  volume={31},
  number={},
  pages={1097-1106},
  abstract={This paper presents an edge-based defocus blur estimation method from a single defocused image. We first distinguish edges that lie at depth discontinuities (called <i>depth</i> edges, for which the blur estimate is ambiguous) from edges that lie at approximately constant depth regions (called <i>pattern</i> edges, for which the blur estimate is well-defined). Then, we estimate the defocus blur amount at <i>pattern</i> edges only, and explore an interpolation scheme based on guided filters that prevents data propagation across the detected <i>depth</i> edges to obtain a dense blur map with well-defined object boundaries. Both tasks (edge classification and blur estimation) are performed by deep convolutional neural networks (CNNs) that share weights to learn meaningful local features from multi-scale patches centered at edge locations. Experiments on naturally defocused images show that the proposed method presents qualitative and quantitative results that outperform state-of-the-art (SOTA) methods, with a good compromise between running time and accuracy.},
  keywords={},
  doi={10.1109/TIP.2021.3139243},
  ISSN={1941-0042},
  month={},
}
```
