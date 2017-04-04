package org.smurve.mnist

import org.smurve.deeplearning.{CostFunction, DV}
import org.smurve.deeplearning.EUCLIDEAN


/**
  * A final layer doesn't actually learn. It provides the cost function and its derivative to the previous layers
  *
  * @param outputSize output size of this layer = number of neurons on the right side
  * @param inputSize  the input size of this layer = number of neurons on the left side
  */
class OutputLayer(inputSize: Int, val outputSize: Int,
                  costFunction: CostFunction = EUCLIDEAN
                 ) extends MNISTLayer(inputSize) {

  private var n: Double = 0

  // containers for averaging over the sample batches
  var avg_cost_by_time: List[Double] = List()
  private var sum_cost: Double = 0.0

  private def reset(): Unit = {
    sum_cost = 0
    n = 0
  }

  /**
    * update all weights and biases with the average of the most recently finished sample batch
    *
    * @param eta the learning factor
    */
  def update(eta: Double): Unit = {
    avg_cost_by_time = sum_cost / n :: avg_cost_by_time
    reset()
  }

  def costByTime: List[Double] = avg_cost_by_time.reverse

  /**
    * nothing to do here because the previous layer already computed the end result. This is a little bit awkward
    * but maintains a consistent interface of layers in general
    */
  def feedForward(x: DV): DV = x

  override def toString: String = {
    "Final Layer"
  }

  /**
    * Feed forward and store the results for back propagation
    *
    * @param x the input vector
    * @param y the desired output                        T
    * @return the weighted delta for back propagation: W  *  d
    */
  def feedForwardAndPropBack(x: DV, y: DV): DV = {
    n += 1

    sum_cost += costFunction.fn(x, y)
    costFunction.deriv(x, y)
  }

}
