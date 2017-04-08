package org.smurve.mnist

import breeze.linalg.{DenseMatrix, DenseVector}
import org.smurve.deeplearning._

/**
  * A neural network layer. Here, a layer shall be understood as the mapping of an input vector to an output vector
  * or, in biological terms, it's like a layer of dendrites, delimited by neurons on each side
  * @param outputSize output size of this layer = number of neurons on the right side
  * @param inputSize the input size of this layer = number of neurons on the left side
  * @param next an optional next layer. if None, this is the output layer of the network
  * @param initWith choose random or constant = 0.5 for all initial values for bias and weight
  */
class FullyConnectedLayer(inputSize: Int, val outputSize: Int,
                          val next: Option[MNISTLayer] = None,
                          initWith: InitWith = INIT_WITH_RANDOM,
                          costFunction: CostFunction = EUCLIDEAN,
                          activation: Activation = a_sigmoid
             ) extends MNISTLayer(inputSize) {

  private var b: DV = newBias(outputSize, initWith)
  private var w: DM = newWeight(outputSize, inputSize, initWith)
  private var n: Double = 0

  // containers for averaging over the sample batches
  private var avg_nabla_b: DV = DenseVector.zeros(outputSize)
  private var avg_nabla_w: DM = DenseMatrix.zeros(outputSize, inputSize)
  var avg_cost_by_time: List[Double] = List()
  private var sum_cost: Double = 0.0

  private def reset() : Unit = {
    avg_nabla_b = DenseVector.zeros(outputSize)
    avg_nabla_w = DenseMatrix.zeros(outputSize, inputSize)
    sum_cost = 0
    n = 0
  }

  def weights: DM = w

  /**
    * update all weights and biases with the average of the most recently finished sample batch
    * @param eta the learning factor
    */
  def update ( eta: Double ): Unit = {
    w :-= avg_nabla_w * ( eta / n)
    b :-= avg_nabla_b * ( eta / n)
    avg_cost_by_time = sum_cost / n :: avg_cost_by_time
    reset()
    next.foreach(_.update(eta))
  }

  def costByTime: List[Double] = avg_cost_by_time.reverse

  private def newBias ( size: Int, initWith: InitWith ) : DV =
    if ( initWith == INIT_WITH_RANDOM )
      DenseVector.rand(size)-DenseVector.fill(size){0.5}
    else
      DenseVector.fill(size){0.001}

  private def newWeight ( outputSize: Int, inputSize: Int, initWith: InitWith): DM =
    if ( initWith == INIT_WITH_RANDOM )
      DenseMatrix.rand(outputSize, inputSize)-DenseMatrix.fill(outputSize, inputSize){0.5}
    else
      DenseMatrix.fill(outputSize, inputSize){0.001}

  /**
    * calculate the output, and feed it into the next layer, if there is one.
    * return this layer's output, or the next layer's output, if there is one.
    * @param x the input, possibly from the previous layer
    * @return the output of the last layer, which may be this
    */
  def feedForward ( x: DV) : DV = {
    val z = w * x + b
    val a = activation.fn ( z )
    if ( next.isEmpty) a else next.get.feedForward(a)
  }

  override def toString : String = {
    w.toString + "\n\n" + b.toString + "\n"
  }

  /**
    * Feed forward and store the results for back propagation
    * @param x the input vector
    * @param y the desired output                        T
    * @return the weighted delta for back propagation: W  *  d
    */
  def feedForwardAndPropBack(x: DV, y: DV): DV = {
    n += 1

    val z = w * x + b
    val a = activation.fn ( z )


    val d = if (next.isEmpty) {
      sum_cost += costFunction.fn(a, y)
      costFunction.deriv(a, y) :* activation.deriv(z)
    } else {
      val wd = next.get.feedForwardAndPropBack( a, y)
      wd :* activation.deriv(z)
    }

    avg_nabla_b += d
    avg_nabla_w += d * x.t

    w.t * d
  }

}
