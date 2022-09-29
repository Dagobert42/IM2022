# A Segmentation-Aware Generative Model for Sequential 3-Dimensional Design

Humans are able to compartmentalize construction processes on different levels of abstraction, implying that we are capable of understanding complex structures both holistically and as a complex sequence of sub-parts. In this work we propose a transformer-based Generative Adversarial Network ("GAN") to answer the question: How can machine learning models leverage sequentially annotated, three-dimensional data in a generative process? We compare our proposed model with a convolutional baseline model to see how transformer-based GAN and convolutional GAN differ in performance. Both models are assessed in terms of training speed and data requirements as well as output quality. Furthermore, we make an effort to analyse the verbosity of the transformer-based approach on a scale of block-by-block construction (player approach) to whole-scale generation (convolutional model). We show that the transformer-based approach opens up interface points in the sequence generation which are not present in the convolutional model. Finally, we aim to demonstrate cooperative potential in the sequential model by locking peripheral frames of an output sequence in the penultimate layer of the generator, then completing the building with suggestions from the learned option space.

This repository was created as part of the individual research module in the M. Sc. Cognitive Systems program at the University of Potsdam.

Author: Arthur Hilbert