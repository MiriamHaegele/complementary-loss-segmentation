import tensorflow as tf


def weak_complementary_biased_label_loss(y_compl: int,
                                         y_pred: tf.Tensor,
                                         Q: tf.Tensor,
                                         gamma: int = 2,
                                         eps: float = 1e-5) -> tf.Tensor:
    """
    Weak complementary label loss for segmentation.

    Args:
      y_compl: Complementary label.
      y_pred: Predictions. A 4-tensor of shape (B, H, W, C).
      Q: Transition matrix.
      gamma: Focal factor.
      eps: Value used to ensure numerical stability.
      sample_weight: Sample weights. A 4-tensor of shape (B, H, W, 1).
    Returns:
      A scalar tensor of the mean loss score.
    """

    # Clip probabilities at `eps`
    y_pred = tf.clip_by_value(y_pred, eps, 1 - eps)

    Q_t = tf.gather(tf.transpose(Q), y_compl)  # get rows of transposed Q, (B,num_classes)
    prob = tf.einsum('bn, bhwn -> bhw', Q_t, y_pred)

    focal_factor = tf.math.pow(1 - prob, gamma)
    loss_compl = - focal_factor * tf.math.log(prob)
    loss_compl = tf.reduce_mean(loss_compl, axis=(1, 2))
    return loss_compl
