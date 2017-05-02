package org.smurve.deeplearning.layers

import org.smurve.deeplearning._
import org.smurve.deeplearning.stats.NNStats

/**
  * A mere activation layer. Applies the activation function to the each component of the input vector
  */
class ActivationLayer(val name: String, activation: Activation = a_scale(1)) extends Layer {

  override def inputSize: Int = nextLayer.map(_.inputSize)
    .getOrElse(throw new IllegalStateException("not initialized yet."))

  /**
    * just the forward feed, returns the final activations as a result
    *
    * @param z_in the vector to classify
    * @return the classification for the given vector according to the recent model
    */
  override def feedForward(z_in: DV): DV = {
    nextLayer.get.feedForward(activation.fn(z_in))
  }

  /**
    * @param z_in the input vector to learn from
    * @param y    the desired classification for the given input
    * @return the back-propagated deltas
    */
  override def feedForwardAndPropBack(z_in: DV, y: DV): DV = {
    val a = activation.fn(z_in)
    val delta = nextLayer.get.feedForwardAndPropBack(a, y)
    delta :* activation.deriv(z_in)
  }

  /**
    * no parameters to update here, just delegate downstream
    */
  def update(nNStats: NNStats): NNStats = {
    nextLayer.get.update(nNStats)
  }

  /**
    * Do nothing but continue backwards
    */
  override def initialize(): Unit = previousLayer.foreach(_.initialize())


  override def toString: String = "Activation: " + activation.name
}
