package org.smurve.deeplearning

import org.smurve.deeplearning.layers.OL

import scala.collection.mutable

/**
  */
class NeuralNetwork(initialLayers: Layer*) extends Layer(initialLayers.head.inputSize) {

  private val layers: mutable.ArrayBuffer[Layer] = mutable.ArrayBuffer()

  initialLayers.foreach(layers += _)

  def º(next: Layer): NeuralNetwork = {
    stack(next)
    this
  }

  def stack(next: Layer): Unit = {
    layers.last.stack(next)
    layers += next
  }

  /**
    * tell the input layer to update, input layer will call the subsequent layers
    *
    * @param eta : the learning factor
    */
  override def update(eta: Double): Unit = layers.head.update(eta)

  /**
    * redirect to the input layer to update, input layer will call the subsequent layers
    *
    * @param x the vector to classify
    * @return the classification for the given vector according to the recent model
    */
  override def feedForward(x: DV): DV = layers.head.feedForward(x)

  /**
    * redirect to the input layer to update, input layer will call the subsequent layers
    *
    * @param x the input vector to learn from
    * @param y the desired classification for the given input
    * @return the back-propagated deltas
    */
  override def feedForwardAndPropBack(x: DV, y: DV): DV = layers.head.feedForwardAndPropBack(x, y)

  def recentLoss: Double = {
    layers.last.asInstanceOf[OL].recentLoss
  }
}
