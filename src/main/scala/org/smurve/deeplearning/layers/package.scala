package org.smurve.deeplearning

package object layers {

  def SIGMOID_LAYER = new ActivationLayer( SIGMOID )
  def RELU_LAYER = new ActivationLayer( RELU )
  def IDENTITY_LAYER = new ActivationLayer( IDENTITY )

}
