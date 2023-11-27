# Intra-class Correlation Coefficient (ICC Regularizer)

Codebase for the NeurIPS 2023 paper - J. Zhang, S. Jayasuriya, V. Berisha, "Learning Repeatable Speech Embeddings Using An Intra-class Correlation Regularizer."

### Abstract

A good supervised embedding for a specific machine learning task is only sensitive to changes in the label of interest and is invariant to other confounding factors. We leverage the concept of repeatability from measurement theory to describe this property and propose to use the intra-class correlation coefficient (ICC) to evaluate the repeatability of embeddings. We then propose a novel regularizer, the ICC regularizer, as a complementary component for contrastive losses to guide deep neural networks to produce embeddings with higher repeatability. We use simulated data to explain why the ICC regularizer works better on minimizing the intra-class variance than the contrastive loss alone. We implement the ICC regularizer and apply it to three speech tasks: speaker verification, voice style conversion, and a clinical application for detecting dysphonic voice. The experimental results demonstrate that adding an ICC regularizer can improve the repeatability of learned embeddings compared to only using the contrastive loss; further, these embeddings lead to improved performance in these downstream tasks.

## ICC Regularizer implementation

**ICCRegularizer.py** - PyTorch implementation for ICC regularizer. This regularizer, which is described in Section 3, focuses on balanced training data, i.e., each class has the same number of samples.

**ICCRegularizer_ext.py** - PyTorch implementation for ICC regularizer extended version. This extended version of the ICC regularizer, which is presented in Appendix A, focuses on imbalanced training data, i.e., each class has a different number of samples.

## Citation

```
@article{zhang2023learning,
  title={Learning Repeatable Speech Embeddings Using An Intra-class Correlation Regularizer},
  author={Zhang, Jianwei and Jayasuriya, Suren and Berisha, Visar},
  journal={arXiv preprint arXiv:2310.17049},
  year={2023}
}
```
