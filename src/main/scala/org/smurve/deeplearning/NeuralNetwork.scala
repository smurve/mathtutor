package org.smurve.deeplearning

import org.smurve.deeplearning.layers.Layer
import org.smurve.deeplearning.stats.{NNStats, OutputLayer}

/**
  * A neural network is a stack of layers, which again is a layer by itself, with an output layer at the end
  */
class NeuralNetwork(input: Layer, output: OutputLayer) extends Layer  {

  /**
    * Need to implement for this to become a layer, but here, we're actually providing the stats ourselves
    */
  override def update( otherNNStats: NNStats ): NNStats =
    throw new NotImplementedError("Use update withouth parameter instead")

  /**
    * tell the input layer to update, input layer will call the subsequent layers
    */
  def update( ): NNStats = input.update( new NNStats())

  /**
    * redirect to the input layer to update, input layer will call the subsequent layers
    *
    * @param x the vector to classify
    * @return the classification for the given vector according to the recent model
    */
  override def feedForward(x: DV): DV = input.feedForward(x)

  /**
    * redirect to the input layer to update, input layer will call the subsequent layers
    *
    * @param x the input vector to learn from
    * @param y the desired classification for the given input
    * @return the back-propagated deltas
    */
  override def feedForwardAndPropBack(x: DV, y: DV): DV = input.feedForwardAndPropBack(x, y)

  override def entry: Layer = input

  override def exit: OutputLayer = output

  /**
    * Recent accumulated loss
    * @return
    */
  def recentLoss: Double = {
    output.recentLoss
  }

  /**
    * Every Layer has a well defined input size that may, however, only be determined once the previous layer is known
    *
    * @return the size of the expected input vector
    */
  override def inputSize: Int = input.inputSize

  /**
    * initialize weights, to be called by the next layer, should continue until the input layer
    */
  override def initialize(): Unit = output.initialize()

  /**
    * a readable name for diagnostic purposes
    *
    * @return
    */
  override def name: String = "The Network"
}
