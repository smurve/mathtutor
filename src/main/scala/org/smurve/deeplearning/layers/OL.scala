package org.smurve.deeplearning.layers

import org.smurve.deeplearning._

/**
  * Output Layer
  * responsible for the loss function calculus and stats
  */
class OL(inputSize: Int, activation: Activation, costFunction: CostFunction) extends Layer(inputSize) {

  var recentLoss: Double = _
  private var batchCost = 0.0
  private var avgCostByTime = List[Double]()
  private var batchCounter = 0

  /**
    * just to fulfill the interface. Do nothing.
    *
    * @param eta : the learning factor
    */
  override def update(eta: Double): Unit = {
    avgCostByTime = batchCost / batchCounter :: avgCostByTime
    batchCounter = 0
    batchCost = 0
  }

  /**
    * just the forward feed, returns the final activations as a result
    *
    * @param x the vector to classify
    * @return the classification for the given vector according to the recent model
    */
  override def feedForward(x: DV): DV = {
    activation.fn(x)
  }

  /**
    *
    * @param z the input vector to learn from
    * @param y the desired classification for the given input
    * @return the back-propagated deltas
    */
  override def feedForwardAndPropBack(z: DV, y: DV): DV = {

    val a = activation.fn(z)
    recentLoss = costFunction.fn(a, y)
    batchCost += recentLoss

    // back propagation
    costFunction.deriv(a, y) :* activation.deriv(z)
  }


  /**
    * not supported. This is the last layer by definition
    * @param next unused
    * @return Nothing. Throws an exception
    */
  override def º(next: Layer): NeuralNetwork = {
    throw new NetworkActivityException("You can't connect the output layer to a subsequent layer")
  }

  override private[deeplearning] def stack(next: Layer) =
    throw new NetworkActivityException("You can't connect the output layer to a subsequent layer")

}
