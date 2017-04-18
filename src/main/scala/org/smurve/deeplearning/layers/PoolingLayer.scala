package org.smurve.deeplearning.layers
import breeze.linalg.DenseVector
import org.smurve.deeplearning.DV

/**
  * Created by wgiersche on 18.04.17.
  */
class PoolingLayer ( val stride: Int = 2, val poolWidth: Int = 2, val poolHeight: Int = 2, val outputWidth: Int,
                     val function: PoolingFunction = MAX_POOLING ) extends Layer {

  private var _inputSize: Int = _

  /**
    * Every Layer has a well defined input size that may, however, only be determined once the previous layer is known
    *
    * @return the size of the expected input vector
    */
  override def inputSize: Int = _inputSize

  /**
    * update the weights from the average corrections collected in previous learnings
    *
    * @return the recent average loss
    */
override def update(): Double = nextLayer.get.update()

  /**
    * just the forward feed, returns the final activations as a result
    *
    * @param x the vector to classify
    * @return the classification for the given vector according to the recent model
    */
  override def feedForward(x: DV): DV = {

    val z = DenseVector.tabulate(outputSize)(t => {
      val c_d = poolWidth * ( t % outputWidth )
      val r_d = poolHeight * ( t / outputWidth )
      pool.map(d => x(c_d + inputWidth * r_d + d)).max
    })

    nextLayer.get.feedForward(z)

  }

  /**
    *
    * @param x the input vector to learn from
    * @param y the desired classification for the given input
    * @return the back-propagated deltas
    */
  override def feedForwardAndPropBack(x: DV, y: DV): DV = {

    val maxAndIndices = DenseVector.tabulate(outputSize)(t => {
      val c_d = poolWidth * ( t % outputWidth )
      val r_d = poolHeight * ( t / outputWidth )
      val neutralElementForFolding = (x(0), 0, 0)
      pool.map(d1 => c_d + inputWidth * r_d + d1).map(d => (x(d),d,t))
        .fold(neutralElementForFolding)((l,r)=>if (l._1>r._1) l else r )
    })

    val delta = nextLayer.get.feedForwardAndPropBack(maxAndIndices.map(_._1), y)

    val res = DenseVector.fill(inputSize){0.0}
    maxAndIndices.foreach(mi => res(mi._2) = delta(mi._3))

    res
  }

  /**
    * initialize weights, will be called by the subsequent layer. Should continue until the input layer
    */
  override private[layers] def initialize() = {
    _inputSize = poolHeight * poolWidth * nextLayer.get.inputSize
    previousLayer.foreach(_.initialize())
  }

  private def inputWidth: Int = poolWidth * outputWidth

  /**
    * @return the index steps that make up a pool
    */
  private def pool: Array[Int] = Array(0, 1, inputWidth, inputWidth + 1)

  private def outputSize: Int = nextLayer.get.inputSize
}
