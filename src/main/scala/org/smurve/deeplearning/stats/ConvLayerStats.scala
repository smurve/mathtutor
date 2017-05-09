package org.smurve.deeplearning.stats

import org.smurve.deeplearning.DV

/**
  * Performance statistics for a convolutional layer
  */
class ConvLayerStats ( val name:String = "Some Conv Layer", val n_features: Int) extends LayerStats {

  /** the history of the weights */
  var w: Array[List[DV]] = Array.fill[List[DV]](n_features){ Nil }

  /** the history of the biases */
  var b: Array[List[Double]] = Array.fill[List[Double]](n_features){ Nil }

  /** the history of the weight gradients nabla_w */
  var nw: Array[List[DV]] = Array.fill[List[DV]](n_features){ Nil }

  /** the history of the bias gradients nabla_b */
  var nb: Array[List[Double]] = Array.fill[List[Double]](n_features){ Nil }
}
