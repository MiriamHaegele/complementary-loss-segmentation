# Leveraging weak complementary labels for semantic segmentation

[![DOI](https://zenodo.org/badge/858722226.svg)](https://zenodo.org/doi/10.5281/zenodo.13772873)

Pre-print: https://arxiv.org/abs/2302.01813

In the paper, we present a deep learning segmentation approach to classify and quantify the two most prevalent primary liver cancers – hepatocellular carcinoma and intrahepatic cholangiocarcinoma – from hematoxylin and eosin (H&E) stained whole slide images. While semantic segmentation of medical images typically requires costly pixel-level annotations by domain experts, there often exists additional information which is routinely obtained in clinical diagnostics but rarely utilized for model training. We propose to leverage such weak information from patient diagnoses by deriving complementary labels that indicate to which class a sample cannot belong to. To integrate these labels, we formulate a complementary loss for segmentation. Motivated by the medical application, we demonstrate for general segmentation tasks that including additional patches with solely weak complementary labels during model training can significantly improve the predictive performance and robustness of a model. On the task of diagnostic differentiation between hepatocellular carcinoma and intrahepatic cholangiocarcinoma, we achieve a balanced accuracy of 0.91 (CI 95%: 0.86 − 0.95) at case level for 165 hold-out patients. Furthermore, we also show that leveraging complementary labels improves the robustness of segmentation and increases performance at case level.

![Figure1](https://github.com/user-attachments/assets/0c9648b3-1044-40fc-a54d-cdf4c714177d)
