package org.smurve.deeplearning.layers

import breeze.linalg.{DenseMatrix, DenseVector}
import org.smurve.deeplearning._
import org.smurve.deeplearning.stats.NNStats

/**
  * A fully connected Layer
  * Note that in my model, the activation function is applied to the input vector
  */
class FullyConnectedLayer( val name: String = "fc",
                           _inputSize: Int,
                          initWith: InitWith = INIT_WITH_CONST,
                          initialValue: Double = .5,
                          inputActivation: Activation = a_scale(1),
                         eta: Double )
  extends Layer {

  // the weights
  private var w: DM = _

  // the bias
  private var b: DV = _

  private var avg_nabla_b: DV = _ // DenseVector.zeros(outputSize)
  private var avg_nabla_w: DM = _ // DenseMatrix.zeros(outputSize, inputSize)

  private var sum_cost: Double = 0.0
  private var batchCounter = 0


  override def initialize(): Unit = {

    if (initWith == INIT_WITH_CONST) {
      w = DenseMatrix.fill(nextLayer.get.inputSize, _inputSize) {
        initialValue
      }
      b = DenseVector.fill(nextLayer.get.inputSize) {
        initialValue
      }
    } else if ( initWith == INIT_WITH_RANDOM) {
      w = DenseMatrix.rand[Double](nextLayer.get.inputSize, _inputSize) -
        DenseMatrix.fill(nextLayer.get.inputSize, _inputSize){0.5}
      b = DenseVector.rand[Double](nextLayer.get.inputSize) -
        DenseVector.fill(nextLayer.get.inputSize) {0.5}
    }
    resetBatch()
    previousLayer.foreach(_.initialize())
  }


  private def resetBatch() : Unit = {
    avg_nabla_b = DenseVector.zeros[Double](nextLayer.get.inputSize)
    avg_nabla_w = DenseMatrix.zeros[Double](nextLayer.get.inputSize, _inputSize)
    sum_cost = 0
    batchCounter = 0
  }

  /**
    * just the forward feed, returns the final activations as a result
    *
    * @param z_in the vector to classify
    * @return the classification for the given vector according to the recent model
    */
  override def feedForward(z_in: DV): DV = {
    assertReady()
    nextLayer.get.feedForward(w * inputActivation.fn(z_in) + b)
  }


  /**
    * @param z_in the input vector to learn from
    * @param y the desired classification for the given input
    * @return the back-propagated deltas
    */
  override def feedForwardAndPropBack(z_in: DV, y: DV): DV = {
    assertReady()
    batchCounter += 1
    val a_in = inputActivation.fn(z_in)
    val z_out = w * a_in + b
    val delta = nextLayer.get.feedForwardAndPropBack(z_out, y)


    avg_nabla_b += delta
    avg_nabla_w += delta * a_in.t

    (w.t * delta) :* inputActivation.deriv(z_in)
  }

  /**
    * update all weights and biases with the average of the most recently finished sample batch
    */
  def update (nNStats: NNStats): NNStats = {
    assertReady()

    w :-= avg_nabla_w * ( eta / batchCounter)
    b :-= avg_nabla_b * ( eta / batchCounter)

    resetBatch()
    nextLayer.get.update(nNStats)
  }


  private def assertReady () : Unit =
    if ( w == null )
      throw new NetworkDesignException("Layer has not been forward-connected yet, and is not an output layer")

  /**
    * Every Layer has a well defined input size that may, however, only be determined once the previous layer is known
    *
    * @return the size of the expected input vector
    */
  override def inputSize: Int = _inputSize

}
