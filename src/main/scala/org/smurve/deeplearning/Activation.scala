package org.smurve.deeplearning

import breeze.linalg.DenseVector

/**
  * container for an activation function and its derivative
  * @param fn the activation function
  * @param deriv its derivative
  */
case class Activation
(
  fn: DenseVector[Double] => DenseVector[Double],
  deriv: DenseVector[Double] => DenseVector[Double]) {
}

