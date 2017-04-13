package org.smurve.deeplearning.layers

import org.smurve.deeplearning._

/**
  * Output Layer
  * responsible for the loss function calculus and stats
  */
class OutputLayer(_inputSize: Int, costFunction: CostFunction = EUCLIDEAN) extends Layer {

  var recentLoss: Double = _
  private var batchCost = 0.0
  private var avgCostByTime = List[Double]()
  private var batchCounter = 0

  /**
    * update the statistics per batch
    *
    * @param eta : the learning factor
    */
  override def update(eta: Double): Double = {
    avgCostByTime = batchCost / batchCounter :: avgCostByTime
    batchCounter = 0
    batchCost = 0
    avgCostByTime.head
  }

  /**
    * just the forward feed, returns the final activations as a result
    *
    * @param x the vector to classify
    * @return the classification for the given vector according to the recent model
    */
  override def feedForward(x: DV): DV = x

  /**
    *
    * @param a_in the input vector to learn from
    * @param y the desired classification for the given input
    * @return the back-propagated deltas
    */
  override def feedForwardAndPropBack(a_in: DV, y: DV): DV = {

    batchCounter += 1
    recentLoss = costFunction.fn(a_in, y)
    batchCost += recentLoss

    // back propagation
    val delta = costFunction.deriv(a_in, y)
    delta
  }


  /**
    * not supported. This is the last layer by definition
    * @param next unused
    * @return Nothing. Throws an exception
    */
  override def ||(next: Layer): Layer = {
    throw new NetworkActivityException("You can't connect the output layer to a subsequent layer")
  }

  /**
    * Every Layer has a well defined input size that may, however, only be determined once the previous layer is known
    *
    * @return the size of the expected input vector
    */
  override def inputSize: Int = _inputSize

  /**
    * initialize weights, to be called by the next layer, should continue until the input layer
    */
  override def initialize(): Unit = previousLayer.get.initialize()
}
