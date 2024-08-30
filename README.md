# Multivariate Time Series Anomaly Detection with Idempotent Reconstruction
Idempotent reconstruction for MTS anomaly detection.

# Introduction of this work
Reconstruction-based methods are commonly choices for multivariate time series (MTS) anomaly detection. However, one challenge these methods may suffer is over generalization, where abnormal inputs are also well reconstructed. In addition, balancing robustness and sensitivity is also important for final performance, as robustness ensures accurate detection in potentially noisy data, while sensitivity enables early detection of subtle anomalies. To address these problems, inspired by idempotent generative network, we take the view from the manifold and propose a light-weight module **I**dempotent **G**eneration for **A**nomaly **D**etection (IGAD) which can be flexibly combined with a reconstruction-based method. We modify the manifold to make sure that normal time points can be mapped onto it while tightening it so that abnormal time points are dropped out with three objectives optimized simultaneously. We evaluate the proposed IGAD on methods with different structures by four real-world datasets, and they achieve a visible improvement on the F1-score, outperforming their predecessors, demonstrating the effective potential of IGAD for further improvement in MTS anomaly detection tasks.

![Architecture](pic/Architecture.png)

# Usage
We have uploaded codes in module IGAD and share **[Anomaly Transformer, ICLR 2022](https://github.com/thuml/Anomaly-Transformer) plus IGAD** as the case. You can modify the codes to make sure that variables _fz_, _f_z_, _ff_z_ and _f_fz_ are in the correct position to obtain reconstructed time series during training.

## ign_simplified.py: Simple Standard Deviation Calculation
This version is suitable for most data processing scenarios where standard deviation is directly computed from frequency domain data. It is straightforward and simple. This approach generally works well when the standard deviation of the input data is not too small.

## ign_stabilized.py: Improved Numerical Stability
This version is more concerned with numerical stability and is suitable for handling cases where the data distribution may result in very small variances. It is more robust when dealing with extreme data values or distributions.

You can choose one of them to adjust module IGAD to your own MTS anomaly detection model.
