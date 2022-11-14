# Run Multiple Models on the Same GPU with Amazon SageMaker Multi-Model Endpoints Powered by NVIDIA Triton Inference Server with a Java Client App

Amazon SageMaker multi-model endpoints (MMEs) provide a scalable and cost-effective way to deploy a large number of deep learning models. MMEs are a popular hosting choice to host hundreds of CPU-based models among customers like Zendesk, Veeva, and AT&T. Previously, you had limited options to deploy hundreds of deep learning models that needed accelerated compute with GPUs. On Oct 25, 2022, we announce MME support for GPU. Now you can deploy thousands of deep learning models behind one SageMaker endpoint. MMEs can now run multiple models on a GPU core, share GPU instances behind an endpoint across multiple models, and dynamically load and unload models based on the incoming traffic. With this, you can significantly save cost and achieve the best price performance.

For further reading:
 - [Amazon SageMaker now enables customers to cost effectively host 1000s of GPU models using Multi Model Endpoint](https://aws.amazon.com/about-aws/whats-new/2022/10/amazon-sagemaker-cost-effectively-host-1000s-gpu-multi-model-endpoint/)
 - [NVIDIA Triton™ Inference Server announcement](https://developer.nvidia.com/blog/run-multiple-ai-models-on-same-gpu-with-sagemaker-mme-powered-by-triton/)
 - [Host multiple models in one container behind one endpoint](https://docs.aws.amazon.com/sagemaker/latest/dg/multi-model-endpoints.html#multi-model-support)
 - [Run multiple deep learning models on GPU with Amazon SageMaker multi-model endpoints](https://aws.amazon.com/blogs/machine-learning/run-multiple-deep-learning-models-on-gpu-with-amazon-sagemaker-multi-model-endpoints/)
 - [Triton Java API](https://github.com/triton-inference-server/client/tree/main/src/java)

