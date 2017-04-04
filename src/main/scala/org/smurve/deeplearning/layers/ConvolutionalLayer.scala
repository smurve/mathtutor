package org.smurve.deeplearning.layers

import breeze.linalg.DenseVector
import org.smurve.deeplearning._

/**
  * A convolutional layer
  */
class ConvolutionalLayer(n_features: Int,
                         lrf: LocalReceptiveField, // the model of the local receptive field
                         initWith: InitWith = INIT_WITH_CONST,
                         initialValue: Option[Array[DV]] = None,
                         inputActivation: Activation = IDENTITY)

  extends Layer(lrf.input_size) {

  var w: Array[DV] = _

  assert ( initialValue.isDefined || initWith == INIT_WITH_RANDOM )

  w = initialValue.getOrElse(Array.tabulate(n_features)(_=>DenseVector.rand[Double](lrf.size)))

  // the next layer - there must be one before we can start.
  private var nextLayer: Layer = _

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
    nextLayer.feedForward(z)
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


  override def *(next: Layer): NeuralNetwork = {
    val requiredOutputSize = n_features * lrf.size_featureMap

    assert(requiredOutputSize == next.inputSize, "Can't connect layers. Sizes don't match")

    nextLayer = next
    new NeuralNetwork(this, next)
  }

  override private[deeplearning] def stack(next: Layer) = ???
}
