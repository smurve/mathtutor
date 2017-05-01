package org.smurve.deeplearning

package object layers {

  def SIGMOID(name: String = "sigmoid" ) = new ActivationLayer( name, a_sigmoid )
  def RELU(name: String = "relu") = new ActivationLayer( name, a_relu )
  def SCALE(name: String = "scale", scale: Double = 1.0) = new ActivationLayer( name, a_scale(scale) )

  /**
    * A layer that activates by sigmoid for negative values and linearly for posive inputs
    */
  def TAU(name: String = "tau") = new ActivationLayer(name, a_tau)

  def MAX_POOLING = new PoolingFunction()
}
