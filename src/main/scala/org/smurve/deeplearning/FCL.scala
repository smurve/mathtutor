package org.smurve.deeplearning

import breeze.linalg.{DenseMatrix, DenseVector}

/**
  * A fully connected Layer
  * Note that in my model, the activation function is applied to the input vector
  */
class FCL (inputSize: Int,
           initWith: InitWith = INIT_WITH_CONST,
           initialValue: Double = .5,
           inputActivation: Activation = IDENTITY )

  extends Layer( inputSize ) {

  // the weights
  private var w: DM = _

  // the bias (Caro told me, everyone has a bias of some sort...;-)
  private var b: DV = _

  // the next layer - there must be one before we can start.
  private var nextLayer: Layer = _

  private var avg_nabla_b: DV = _ // DenseVector.zeros(outputSize)
  private var avg_nabla_w: DM = _ // DenseMatrix.zeros(outputSize, inputSize)

  private var sum_cost: Double = 0.0
  private var batchCounter = 0

  /**
    * stack the next layer on top of this
    * @param next the next layer
    * @return the resulting network
    */
  def º ( next: Layer ): NeuralNetwork = {
    stack( next )
    new NeuralNetwork(this, next)
  }

  /**
    * stack the next layer on top of this and wrap it into a neural network instance
    * @param next the next layer
    * @return the resulting network
    */
  override private[deeplearning] def stack ( next: Layer ): Unit = {
    nextLayer = next
    initParams()
  }

  private def initParams(): Unit = {

    if (initWith == INIT_WITH_CONST) {
      w = DenseMatrix.fill(nextLayer.inputSize, inputSize) {
        initialValue
      }
      b = DenseVector.fill(nextLayer.inputSize) {
        initialValue
      }
    } else if ( initWith == INIT_WITH_RANDOM) {
      w = DenseMatrix.rand[Double](nextLayer.inputSize, inputSize) -
        DenseMatrix.fill(nextLayer.inputSize, inputSize){0.5}
      b = DenseVector.rand[Double](nextLayer.inputSize) -
        DenseVector.fill(nextLayer.inputSize) {0.5}
    }
    resetBatch()
  }


  private def resetBatch() : Unit = {
    avg_nabla_b = DenseVector.zeros[Double](nextLayer.inputSize)
    avg_nabla_w = DenseMatrix.zeros[Double](nextLayer.inputSize, inputSize)
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
    nextLayer.feedForward(w * inputActivation.fn(z_in) + b)
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
    val delta = nextLayer.feedForwardAndPropBack(z_out, y)


    avg_nabla_b += delta
    avg_nabla_w += delta * a_in.t

    (w.t * delta) :* inputActivation.deriv(z_in)
  }

  /**
    * update all weights and biases with the average of the most recently finished sample batch
    * @param eta the learning factor
    */
  def update ( eta: Double ): Unit = {
    assertReady()

    w :-= avg_nabla_w * ( eta / batchCounter)
    b :-= avg_nabla_b * ( eta / batchCounter)

    resetBatch()
    nextLayer.update(eta)
  }


  private def assertReady () : Unit =
    if ( w == null )
      throw new NetworkDesignException("Layer has not been forward-connected yet, and is not an output layer")
}
