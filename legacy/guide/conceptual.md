<!-- LEGACY: This guide captures the earlier single-layer / probe-gated steering
workflow. It is preserved for historical context, but `guide/how-to-abliterate.md`
is now the core reference for the current extraction strategy. -->
The foundation of this project lies in manipulating the geometry of a Transformer's internal representations. The core premise is the **Linear Representation Hypothesis**, which states that large language models encode high-level semantic concepts (such as "harmfulness," "politeness," or "truthfulness") as linear directions—specific vectors—within their high-dimensional activation space. 

To execute conditional steering, you must understand how to locate these vectors, measure their presence, and mathematically inject them into the model's forward pass.

### 1. The Architecture and Activation Caching

A Transformer processes an input sequence of tokens by passing them through $L$ consecutive layers. The backbone of this architecture is the **residual stream**. At any given layer $l$, the residual stream is a matrix $\mathbf{H}^{(l)} \in \mathbb{R}^{T \times d}$, where $T$ is the number of tokens in the sequence and $d$ is the model's hidden dimension.

The update rule for the residual stream at layer $l+1$ is:
$$\mathbf{H}^{(l+1)} = \mathbf{H}^{(l)} + \text{Attention}(\text{LN}(\mathbf{H}^{(l)})) + \text{FFN}(\text{LN}(\mathbf{H}^{(l)}))$$

Where $\text{LN}$ is Layer Normalization. 

**Activation Caching** is the process of intercepting the forward pass and saving the state of $\mathbf{H}^{(l)}$ to memory before it moves to layer $l+1$. Typically, when steering behavior, we care most about the activation corresponding to the *last token* in the sequence, as this token contains the aggregated context used to predict the next word. Let’s define this specific $d$-dimensional vector as $\mathbf{h}^{(l)} \in \mathbb{R}^d$.

### 2. Linear Probing (Feature Localization)

A linear probe is a logistic regression classifier trained directly on the cached activations $\mathbf{h}^{(l)}$ to detect the presence of a specific concept. 

To train a probe, you construct a dataset of prompts $D = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$. For a "safety" guardrail, $y_i = 1$ if the prompt elicits harmful intent, and $y_i = 0$ if it is benign. You run all prompts through the model, caching the activations $\{\mathbf{h}_i^{(l)}\}_{i=1}^N$.

The probe consists of a weight vector $\mathbf{w} \in \mathbb{R}^d$ and a bias $b \in \mathbb{R}$. The probability that the concept is present in a given activation is computed as:
$$p_i = \sigma(\mathbf{w}^T \mathbf{h}_i^{(l)} + b)$$
Where $\sigma$ is the sigmoid function. You optimize $\mathbf{w}$ and $b$ using standard Gradient Descent to minimize Binary Cross-Entropy loss. 

Mathematically, the trained weight vector $\mathbf{w}$ defines a hyperplane in the $d$-dimensional space that separates the "harmful" activations from the "benign" ones. The vector $\mathbf{w}$ itself is orthogonal to this hyperplane, pointing in the direction of the concept.

### 3. Steering Vector Extraction (Difference-in-Means)

While the probe weight $\mathbf{w}$ can technically be used as a steering direction, a much more robust empirical method for extracting the pure concept vector is **Contrastive Activation Addition** (or Difference-in-Means).

You separate your cached activations into two sets: $S_{pos}$ (concept present) and $S_{neg}$ (concept absent). You then compute the centroid (mean vector) for each set:
$$\bm{\mu}_{pos} = \frac{1}{|S_{pos}|} \sum_{\mathbf{h} \in S_{pos}} \mathbf{h} \quad \text{and} \quad \bm{\mu}_{neg} = \frac{1}{|S_{neg}|} \sum_{\mathbf{h} \in S_{neg}} \mathbf{h}$$

The steering vector $\mathbf{v}_{concept} \in \mathbb{R}^d$ is the vector pointing from the negative centroid to the positive centroid:
$$\mathbf{v}_{concept} = \bm{\mu}_{pos} - \bm{\mu}_{neg}$$

**Matrix Example:**
Assume a miniature Transformer with a hidden dimension $d=3$. 
You cache the activations for two "unsafe" prompts and two "safe" prompts at layer 15.
Unsafe activations: $\mathbf{h}_{p1} = \begin{bmatrix} 1.2 \\ -0.5 \\ 3.1 \end{bmatrix}$, $\mathbf{h}_{p2} = \begin{bmatrix} 1.0 \\ -0.1 \\ 2.9 \end{bmatrix} \implies \bm{\mu}_{pos} = \begin{bmatrix} 1.1 \\ -0.3 \\ 3.0 \end{bmatrix}$
Safe activations: $\mathbf{h}_{n1} = \begin{bmatrix} -0.1 \\ 0.4 \\ 0.2 \end{bmatrix}$, $\mathbf{h}_{n2} = \begin{bmatrix} 0.3 \\ 0.2 \\ 0.0 \end{bmatrix} \implies \bm{\mu}_{neg} = \begin{bmatrix} 0.1 \\ 0.3 \\ 0.1 \end{bmatrix}$

Your steering vector is:
$$\mathbf{v}_{concept} = \begin{bmatrix} 1.1 \\ -0.3 \\ 3.0 \end{bmatrix} - \begin{bmatrix} 0.1 \\ 0.3 \\ 0.1 \end{bmatrix} = \begin{bmatrix} 1.0 \\ -0.6 \\ 2.9 \end{bmatrix}$$

We typically normalize this vector to a unit length $\mathbf{\hat{v}} = \frac{\mathbf{v}_{concept}}{||\mathbf{v}_{concept}||}$ so that we can control the magnitude of our intervention systematically.

### 4. Activation Addition / Patching (The Intervention)

Activation addition (or steering) is the process of casually intervening in the network's computation during inference. 

When a user submits a new prompt, the model begins generating tokens. During the forward pass for a new token, the model computes the residual stream at layer $l$, resulting in vector $\mathbf{h}^{(l)}$. 

Before this vector is passed to layer $l+1$, you mathematically inject your normalized steering vector, scaled by an intervention strength scalar $\alpha$:
$$\tilde{\mathbf{h}}^{(l)} = \mathbf{h}^{(l)} + \alpha \mathbf{\hat{v}}$$

This modified vector $\tilde{\mathbf{h}}^{(l)}$ is then passed through the LayerNorm and into the Attention and FFN blocks of the next layer. By physically translating the activation state in the $d$-dimensional space along the concept direction, you force the downstream layers to calculate attention patterns and feed-forward updates that align with that concept. If you use a negative $\alpha$ (e.g., subtracting the "harmful" vector), you achieve **Concept Erasure**, effectively blinding the downstream layers to that semantic feature.

*Architectural Note:* Because Transformers utilize Layer Normalization extensively, the scale of $\alpha$ is highly sensitive. If $\alpha$ is too large, the added vector dominates the variance of the residual stream. When $\text{LN}(\tilde{\mathbf{h}}^{(l)})$ is computed, the normalization process will crush the original contextual information in $\mathbf{h}^{(l)}$, resulting in gibberish text (a collapse in perplexity).

### 5. Tying it Together: Conditional Steering

Your project combines these elements into a dynamic feedback loop. Instead of permanently adding or subtracting the steering vector at every token generation step, the intervention is conditional.

During the generation of token $t$, the forward pass reaches layer $l$, yielding $\mathbf{h}_t^{(l)}$.
1. **Probe Evaluation:** The cached vector is immediately fed into your pre-trained linear probe: $p_t = \sigma(\mathbf{w}^T \mathbf{h}_t^{(l)} + b)$.
2. **Conditional Logic:** If $p_t > \tau$ (where $\tau$ is your predefined confidence threshold, e.g., 0.85):
   $$\tilde{\mathbf{h}}_t^{(l)} = \mathbf{h}_t^{(l)} - \alpha \mathbf{\hat{v}}_{harmful}$$
   If $p_t \le \tau$:
   $$\tilde{\mathbf{h}}_t^{(l)} = \mathbf{h}_t^{(l)}$$

This architecture allows the LLM to traverse its latent space normally, maintaining its full distribution of knowledge and reasoning capabilities. The guardrail only activates precisely at the computational moment the probe detects the residual stream drifting into the targeted concept subspace, immediately applying a negative geometric translation to steer it back to safety.


# GPU requirements estimates
Substep,Description,Primary Resource,Est. GPU Memory (8B Model)
1. Model Loading,Loading weights into VRAM (FP16 or Quantized).,GPU,16GB (FP16) or 5-8GB (4-bit)
2. Dataset Prep,"Cleaning 500–1000 pairs of ""safe"" vs. ""unsafe"" text.",CPU,N/A
3. Activation Collection,Running the model to save residual stream tensors.,GPU,8-12GB (VRAM used for KV cache)
4. Probe Training,Training a simple Logistic Regression on saved tensors.,CPU,N/A (Sklearn is fine)
5. Vector Calculation,Computing the difference-in-means of the activations.,CPU,N/A
6. Intervention Loop,Hooking the model to add the vector during generation.,GPU,8-16GB (Standard Inference)