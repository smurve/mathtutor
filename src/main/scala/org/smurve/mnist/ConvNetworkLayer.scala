package org.smurve.mnist

import breeze.linalg.{DenseMatrix, DenseVector}

/**
  */
class ConvNetworkLayer(inputSize: Int, outputSize: Int,
                       next: Option[ConvNetworkLayer] = None,
                       initWith: InitWith = INIT_WITH_RANDOM,
                       costDerivative: (DV, DV)=>DV,
                       activation: Activation,
                       features: DM
             ) extends NNLayer ( inputSize, outputSize, next, initWith, costDerivative, activation) {

  private val b: DV = newBias(outputSize)
  private var w: DM = features


  private var n: Int = 0

  // containers for averaging over the sample batches
  private var avg_nabla_b: DV = DenseVector.zeros(outputSize)
  private var avg_nabla_w: DM = DenseMatrix.zeros(outputSize, inputSize)

  private def reset() : Unit = {
    avg_nabla_b = DenseVector.zeros(outputSize)
    avg_nabla_w = DenseMatrix.zeros(outputSize, inputSize)
    n = 0
  }

  /**
    * update all weights and biases with the average of the most recently finished sample batch
    * @param eta the learning factor
    */
  override def update ( eta: Double ): Unit = {
    super.update(eta)
  }

  /**
    * For the time being: no bias.
    * @param size the size of the output layer
    * @return
    */
  private def newBias ( size: Int ) : DV =
      DenseVector.fill(size){0.0}


  /**
    * calculate the output, and feed it into the next layer, if there is one.
    * return this layer's output, or the next layer's output, if there is one.
    * @param x the input, possibly from the previous layer
    * @return the output of the last layer, which may be this
    */
  override def feedForward ( x: DV) : DV = {
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
  override def feedForwardAndPropBack(x: DV, y: DV): DV = {
    n += 1

    val z = w * x + b
    val a = activation.fn ( z )

    val d = if (next.isEmpty) {
      costDerivative(a, y)
    } else {
      val wd = next.get.feedForwardAndPropBack( a, y)
      wd :* activation.deriv(z)
    }

    avg_nabla_b += d
    avg_nabla_w += d * x.t

    w.t * d
  }

}
