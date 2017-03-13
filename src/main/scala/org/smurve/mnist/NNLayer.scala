package org.smurve.mnist

import breeze.linalg.{DenseMatrix, DenseVector}

/**
  * A neural network layer. Here, a layer shall be understood as the mapping of an input vector to an output vector
  * or, in biological terms, it's like a layer of dendrites, delimited by neurons on each side
  * @param outputSize output size of this layer = number of neurons on the right side
  * @param inputSize the input size of this layer = number of neurons on the left side
  * @param next an optional next layer. if None, this is the output layer of the network
  * @param initWith choose random or constant = 0.5 for all initial values for bias and weight
  */
class NNLayer(val inputSize: Int, val outputSize: Int,
              val next: Option[NNLayer] = None,
              initWith: InitWith = INIT_WITH_RANDOM,
              val costDerivative: (DV, DV)=>DV
             )(implicit network: NeuralNetwork) {

  private var b: DV = newBias(outputSize, initWith)
  private var w: DM = newWeight(outputSize, inputSize, initWith)
  private var n: Double = 0

  private var nabla_b: DV = DenseVector.zeros(outputSize)
  private var nabla_w: DM = DenseMatrix.zeros(outputSize, inputSize)

  private def reset() : Unit = {
    nabla_b = DenseVector.zeros(outputSize)
    nabla_w = DenseMatrix.zeros(outputSize, inputSize)
    n = 0
  }

  def update ( eta: Double ): Unit = {
    w = w - nabla_w * ( eta / n)
    b = b - nabla_b * ( eta / n)
    reset()
    next.foreach(_.update(eta))
  }

  private def newBias ( size: Int, initWith: InitWith ) : DV =
    if ( initWith == INIT_WITH_RANDOM ) DenseVector.rand(size) else DenseVector.fill(size){0.5}

  private def newWeight ( outputSize: Int, inputSize: Int, initWith: InitWith): DM =
    if ( initWith == INIT_WITH_RANDOM )
      DenseMatrix.rand(outputSize, inputSize)
    else
      DenseMatrix.fill(outputSize, inputSize){0.5}

  def feedForward ( x: DV) : DV = {
    val z = w * x + b
    val a = sigmoid ( z )
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
    val a = sigmoid ( z )

    val d = if (next.isEmpty) {
      costDerivative(a, y)
    } else {
      val wd = next.get.feedForwardAndPropBack( a, y)
      adamard (wd,  sigmoid_prime(z))
    }

    nabla_b += d
    nabla_w += d * x.t

    w.t * d
  }

}
