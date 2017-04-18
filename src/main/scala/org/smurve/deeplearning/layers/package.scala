package org.smurve.deeplearning

package object layers {

  def SIGMOID = new ActivationLayer( a_sigmoid )
  def RELU = new ActivationLayer( a_relu )
  def SCALE(scale: Double) = new ActivationLayer( a_scale(scale) )

  /**
    * A layer that activates by sigmoid for negative values and linearly for posive inputs
    */
  def TAU = new ActivationLayer(a_tau)

  def MAX_POOLING = new PoolingFunction()
}
