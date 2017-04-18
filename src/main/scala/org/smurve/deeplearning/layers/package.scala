package org.smurve.deeplearning

package object layers {

  def SIGMOID = new ActivationLayer( a_sigmoid )
  def RELU = new ActivationLayer( a_relu )
  def SCALE(scale: Double) = new ActivationLayer( a_scale(scale) )

}
