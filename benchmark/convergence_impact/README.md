# Convergence impact
_Fig. 2. Impact of GPU change on training convergence (Changing GPUs from 2 to 4 with GPT-3 and MNIST)_

Fig. 2a shows how model convergence, plotted as the loss value, is affected after adding a GPU (vertical orange line) under data parallelism. The solid black line shows regular model convergence with a static GPU allocation; the dashed red line shows convergence after the scale-out event when the dataset is processed inconsistently after re-partitioning: when resuming the training in the middle of the epoch, the first half of the training data is used twice, which overfits the model and reduces the loss value unreasonably.

In Fig. 2b, we show how the global batch size must be kept constant after adding a GPU (vertical orange line) under data parallelism. The solid black line shows model convergence (measured as loss) without the GPU change. The dashed red line shows the divergence when the GPU allocation changes but the device batch size remains constant.
