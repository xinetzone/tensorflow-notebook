# TensorFlow 学习笔记

TensorFlow 中默认为**动态图**方式执行，便于调试；同时可以轻松使用 `tf.function` 把静态图转换为动态图，还可打开 XLA 编译优化功能，提高性能。API 一致性和丰富的文档非常重要，TensorFlow 对于保持 API 的一致性做了大量工作。TF2 中可轻松使用 Distribute Strategy，一行代码就可以实现从单机多卡到多机多卡的切换；提供了 `tf.data` 提供高性能可扩展性的数据流水线；也在 TensorBoard 中提供了丰富的功能来帮助调试和优化性能。

TensorFlow Lite 加速了端侧机器学习 (On-device ML，ODML) 的快速发展，让机器学习无处不在。它支持安卓、iOS、嵌入式设备，以及极小的 MCU 平台。TensorFlow Lite 支持多种量化和压缩技巧，支持各种硬件加速器（比如 NNAPI、GPU、DSP、CoreML 等)，持续发布前沿模型（比如 EfficientNet-Lite，MobileBERT）和完整参考应用，并提供丰富的工具降低门槛（比如 TFLite Model Maker 和 Android Studio ML model binding）。最近的一些突破，比如基于强大的 BERT 模型的问题回答系统也可以运行在低端 CPU 上（利用压缩的 MobileBERT），在 MCU 上的简单语音识别模型只需要 20KB，这些给端侧机器学习带来了广阔前景，真正让机器学习无处不在成为可能。

TensorFlow 生态系统还有着丰富的工具链。TFX 支持端到端的复杂的机器学习流程，而其中 TF Serving 是广泛使用的高性能的服务器端部署平台。TF.js 支持使用 Javascript 在浏览器端部署，也与微信小程序有很好的集成，是最广大 Javascript 爱好者提供了便利。TensorFlow Hub 提供了丰富的即开即用的上千个预训练模型，覆盖语言、语音、文本等多种应用，方便使用迁移学习，进一步降低机器学习的门槛。众多团队基于 TensorFlow 构建了多元的工具，比如 TF Probability （TF 和概率模型结合）、TF Federated (联邦学习）、TF Graphics（TF 和图形学），甚至 TF Quantum（TF 和量子计算）。

TensorFlow 开源的目标是促进人人可用的负责任的 AI (Responsible AI)，为此提供了一系列的工具加速此过程。我们推荐了最佳实践（如 People+AI Guidebook），促进公正性（比 Fairness Indicators），推动模型可解释性（促进相关研究，提供相关工具），关注隐私（比如 TF Privacy，TF Federated），以及关注安全。

资源清单见：[resources](resources.md)。

@import "basic.md"