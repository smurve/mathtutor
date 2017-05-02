package org.smurve.deeplearning.stats

import org.smurve.deeplearning._
import org.smurve.deeplearning.layers.Layer

/**
  * Output Layer
  * responsible for the loss function calculus and stats
  */
class OutputLayer(size: Int, costFunction: CostFunction = EUCLIDEAN) extends Layer {

  var recentLoss: Double = _
  val name = "Output"
  private var batchCost = 0.0
  private var avgCostByTime = List[Double]()
  private var batchCounter = 0
  private var outputLayerStats = new OutputLayerStats()

  /**
    * update the statistics per batch
    */
  override def update( nNStats: NNStats): NNStats = {
    outputLayerStats.registerCost (batchCost / batchCounter)
    batchCounter = 0
    batchCost = 0
    nNStats.registerStats(outputLayerStats)
    nNStats
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
  override def inputSize: Int = size

  /**
    * initialize weights, to be called by the next layer, should continue until the input layer
    */
  override def initialize(): Unit = previousLayer.get.initialize()
}
