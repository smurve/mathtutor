package org.smurve.deeplearning.layers

import breeze.linalg.{DenseMatrix, DenseVector}
import org.smurve.deeplearning._
import org.smurve.deeplearning.optimizers.Optimizer
import org.smurve.deeplearning.stats.NNStats

/**
  * A fully connected or dense layer
  * @param name: A name for diagnostics
  * @param _inputSize the size of the input vector
  * @param initWith initialization strategy
  * @param initialValue a number to initialize all weights with - typically for diagnostic purposes
  * @param opt_w the optimizer for the weights
  * @param opt_b the optimizer for the bias
  */
class DenseLayer(val name: String = "Some Affine Layer",
                 _inputSize: Int,
                 initWith: InitWith = INIT_WITH_CONST,
                 initialValue: Double = .5,
                 opt_w: Optimizer, opt_b: Optimizer )  extends Layer {

  // the weights
  private var w: DM = _

  // the bias
  private var b: DV = _

  def dump : (DM, DV) = (w,b)

  private var avg_nabla_b: DV = _ // DenseVector.zeros(outputSize)
  private var avg_nabla_w: DM = _ // DenseMatrix.zeros(outputSize, inputSize)
  private var prev_avg_nabla_b: DV = _ // DenseVector.zeros(outputSize)
  private var prev_avg_nabla_w: DM = _ // DenseMatrix.zeros(outputSize, inputSize)

  private var batchCounter = 0.0


  override def initialize(): Unit = {

    val outputSize: Int = nextLayer.get.inputSize
    if (initWith == INIT_WITH_CONST) {
      w = DenseMatrix.fill(outputSize, _inputSize) { initialValue }
      b = DenseVector.fill(outputSize) { initialValue }
    } else if ( initWith == INIT_WITH_RANDOM) {
      w = DenseMatrix.rand(outputSize, _inputSize) - DenseMatrix.fill(outputSize, _inputSize){0.5}
      b = DenseVector.rand(outputSize) - DenseVector.fill(outputSize) {0.5}
    }

    prev_avg_nabla_w = DenseMatrix.fill(outputSize, _inputSize){0.0}
    prev_avg_nabla_b = DenseVector.fill(outputSize){0.0}

    resetBatch()
    previousLayer.foreach(_.initialize())
  }


  private def resetBatch() : Unit = {
    avg_nabla_b = DenseVector.zeros[Double](nextLayer.get.inputSize)
    avg_nabla_w = DenseMatrix.zeros[Double](nextLayer.get.inputSize, _inputSize)
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
    nextLayer.get.feedForward(w * z_in + b)
  }


  /**
    * @param a_in the input vector, possibly from the previous layer
    * @param y the desired classification for the given input
    * @return the back-propagated deltas
    */
  override def feedForwardAndPropBack(a_in: DV, y: DV): DV = {
    assertReady()
    batchCounter += 1
    val z_out = w * a_in + b
    val delta = nextLayer.get.feedForwardAndPropBack(z_out, y)

    avg_nabla_b += delta
    avg_nabla_w += delta * a_in.t

    w.t * delta
  }

  /**
    * update all weights and biases with the average of the most recently finished sample batch
    */
  def update( nNStats: NNStats ): NNStats = {
    assertReady()

    val stats = nextLayer.get.update(nNStats)

    val step_w = opt_w.nextStep(avg_nabla_w * (1.0 / batchCounter))
    val step_b = opt_b.nextStep(avg_nabla_b * (1.0 / batchCounter))

    w :-= step_w
    b :-= step_b

    resetBatch()

    stats
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

  override def toString: String = {
    name + ": " + inputSize + "x" + "?"
  }
}

