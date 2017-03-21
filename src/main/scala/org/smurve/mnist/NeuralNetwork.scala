package org.smurve.mnist


/**
  * A neural network. Note that we use the term layer for the function sigmoid(mx+b), not for the io gates.
  * Thus, a network of sizes 4,8,3 actually has two layers, one of [4,8] and the other of [8,3]
  *
  * @param ioSizes     List of sizes of the input/output vectors
  * @param activations array of activation functions for the layers
  * @param initWith    initial value strategy: INIT_WITH_RANDOM or INIT_WITH_CONSTANT
  */
class NeuralNetwork(val ioSizes: Array[Int], val activations: Array[Activation], initWith: InitWith) {

  assert(ioSizes.length - 1 == activations.length,"Number of activation functions does not match!")

  val layers = new Array[NNLayer](ioSizes.length - 1)

  var prev: Option[NNLayer] = None
  for (i <- layers.length - 1 to 0 by -1) {
    layers(i) = new NNLayer(ioSizes(i), ioSizes(i + 1), prev, initWith,
      euclideanCostDerivative, activations(i))
    prev = Some(layers(i))
  }

  /**
    * Once the network is trained the weights should produce accurate predictions
    * @param x the input vector to classify
    * @return
    */
  def classify(x: DV ) : DV = {
    layers(0).feedForward ( x )
  }

  def update ( eta: Double ): Unit = layers(0).update(eta)

  override def toString : String = {
    "\n" + layers.map(_.toString).mkString("\n\n") +
    "\n--------------------------------------------------------------------\n"
  }

  def train(x: DV, y: DV): DV = {
    layers(0).feedForwardAndPropBack(x, y)
  }

  def train(sample: (DV, DV)): DV = {
    layers(0).feedForwardAndPropBack(sample._1, sample._2)
  }

  def euclideanCost(finalActivation: DV, desired: DV): Double = {
    val diff = finalActivation - desired
    (.5 * diff.t * diff).toArray.apply(0)
  }

  def euclideanCostDerivative(finalActivation: DV, desired: DV): DV = finalActivation - desired

}
