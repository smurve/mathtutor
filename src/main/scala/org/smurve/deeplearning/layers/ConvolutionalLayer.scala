package org.smurve.deeplearning.layers

import breeze.linalg.DenseVector
import org.smurve.deeplearning._

/**
  * A convolutional layer
  */
class ConvolutionalLayer(n_features: Int,
                         lrf: LocalReceptiveFieldSpec, // the model of the local receptive field
                         initWith: InitWith = INIT_WITH_CONST,
                         initialValue: Option[Array[DV]] = None,
                         inputActivation: Activation = a_identity)

  extends Layer {

  var w: Array[DV] = _

  assert ( initialValue.isDefined || initWith == INIT_WITH_RANDOM )

  w = initialValue.getOrElse(Array.tabulate(n_features)(_=>DenseVector.rand[Double](lrf.size)))

  /**
    * update the weights from the average corrections collected in previous learnings
    *
    * @param eta : the learning factor
    */
  override def update(eta: Double): Unit = ???

  /**
    * just the forward feed, returns the final activations as a result
    *
    * @param x the vector to classify
    * @return the classification for the given vector according to the recent model
    */
  override def feedForward(x: DV): DV = {
    val a = inputActivation.fn(x)
    val z = DenseVector((0 until n_features).flatMap(convolute(_,a)).toArray)
    nextLayer.get.feedForward(z)
  }


  /**
    * This is where the actual convolution happens
    *
    * @param f the index of the feature to be convoluted
    * @param input the layer's input vector
    * @return the resulting feature map
    */
  def convolute(f: Int, input: DV): Array[Double] = {
    (0 until lrf.size_featureMap).map(k => {
      (0 until lrf.size).map(j => {
        w(f)(j) * input(lrf.tau(k, j))
      }).sum
    }).toArray
  }


  /**
    *
    * @param x the input vector to learn from
    * @param y the desired classification for the given input
    * @return the back-propagated deltas
    */
  override def feedForwardAndPropBack(x: DV, y: DV): DV = ???


  override def *(next: Layer): Layer = {
    super.*(next)
  }

  /**
    * Every Layer has a well defined input size that may, however, only be determined once the previous layer is known
    *
    * @return the size of the expected input vector
    */
  override def inputSize: Int = lrf.input_size

  /**
    * initialize weights, to be called by the next layer, should continue until the input layer
    */
  override def initialize(): Unit = {
    val requiredOutputSize = n_features * lrf.size_featureMap
    assert(requiredOutputSize == nextLayer.get.inputSize, "Can't connect layers. Sizes don't match")
    previousLayer.foreach(_.initialize())
  }
}
