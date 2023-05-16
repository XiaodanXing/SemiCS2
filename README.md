# SemiCS2
This is the source code for our TMI paper:

Less is More: Unsupervised Mask-guided Annotated CT Image Synthesis with Minimum Manual Segmentations

![teaser](https://github.com/XiaodanXing/SemiCS2/assets/30890745/e0b66e09-1f8c-41e3-b6bd-64563b5f18d7)

## Highlight
In this work, we propose a novel strategy for medical image synthesis, namely Unsupervised Mask (UM)-guided synthesis, to obtain both synthetic images and segmentations using limited manual segmentation labels. 

1. We first develop a **superpixel based** algorithm to generate unsupervised structural guidance and then design a conditional generative model to synthesize images and annotations simultaneously from those unsupervised masks in a semi-supervised multi-task setting. 
2. In addition, we devise a **multi-scale multi-task Fréchet Inception Distance (MM-FID)** and **multi-scale multi-task standard deviation (MM-STD)** to harness both fidelity and variety evaluations of synthetic CT images. With multiple analyses on different scales, we could produce stable image quality measurements with high reproducibility. 

Compared with the segmentation mask guided synthesis, our UM-guided synthesis provided high-quality synthetic images with significantly higher fidelity, variety, and utility (p < 0.05 by Wilcoxon Signed Ranked test).

![flowchart](https://github.com/XiaodanXing/SemiCS2/assets/30890745/1e1b7d2a-402d-436f-b99c-967698847e2a)


## Showcases
According to our experiments, the loss of variety is inevitable during image synthesis. However, we claimed that introducing intensity guidance can address and improve the loss
of variety in synthetic images. 

To visualize the data distributions, we used features derived from the pretrained model $P_4$ under the resolution of $2^{10}\times2^{10}$ because the target of $P_4$ is to discriminate different image types, and t-distributed Stochastic Neighbor Embedding (T-SNE) \cite{hinton2002stochastic} was used to reduce the feature dimensions. 

We discovered two clusters in the image space that the V2M2I model failed to cover, as shown in the black boxes in Fig. \ref{fig:layout} (1). One cluster is located between $x=[-80,-70]$ and $y= [0, 20]$, and the other is located between $x=[20, 40]$ and $y=[-80,-60]$. We randomly selected two V2UM2I synthesized images from each cluster. This indicates that the V2M2I model failed to synthesize CT montages with different FOVs, but our V2UM2I was able to perform these syntheses. Our real dataset contains images reconstructed with different kernels, thus producing images with limited FOVs and circular FOVs. 

![layout](https://github.com/XiaodanXing/SemiCS2/assets/30890745/67e9054c-36ff-4258-96e1-52524bbc1428)


## Citation
This repository partially based on:

- pix2pixHD: High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs ([code](https://github.com/NVIDIA/pix2pixHD) and 
[paper](https://arxiv.org/abs/1711.11585));

If you find the proposed super-pixel guided synthesis, MM-FID and MM-STD useful, please cite our paper:

- Xing, X., Papanastasiou, G., Walsh, S., & Yang, G. (2023). Less is More: Unsupervised Mask-guided Annotated CT Image Synthesis with Minimum Manual Segmentations. IEEE Transactions on Medical Imaging.
