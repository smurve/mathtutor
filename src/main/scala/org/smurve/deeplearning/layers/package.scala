package org.smurve.deeplearning

package object layers {

  def SIGMOID = new ActivationLayer( a_sigmoid )
  def RELU = new ActivationLayer( a_relu )
  def IDENTITY = new ActivationLayer( a_identity )

}
