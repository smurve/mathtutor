package org.smurve.deeplearning.layers

import breeze.linalg.{DenseVector, sum}
import org.smurve.deeplearning._

/**
  * A convolutional layer
  * This implementation supports any number of LRFs of different sizes. These dimensions of these have to be specified
  * in the constructor argument as an array of LocalReceptiveFieldSpecs. The output comprises of a vector containing
  * the feature maps in the same order that their LRFs are specified in the constructor.
  *
  * @param lrfSpecs array of LRF specifications
  * @param eta the learning factor for this layer
  */
class ConvolutionalLayer(val lrfSpecs: Array[LocalReceptiveFieldSpec], val eta: Double)

  extends Layer {

  val n_features: Int = lrfSpecs.length

  val fmapSizes: Array[Int] = lrfSpecs.map(_.fmap_size)
  val fmapOffsets: Array[Int] = fmapSizes.zipWithIndex.map(x=>fmapSizes.slice(0,x._2).sum)

  private var avg_nabla_b: Array[Double] = _ // DenseVector.zeros(outputSize)
  private var avg_nabla_w: Array[DV] = _ // DenseMatrix.zeros(outputSize, inputSize)

  private var batchCounter = 0

  private def resetBatch() : Unit = {
    avg_nabla_b = Array.fill(lrfSpecs.length){0.0}

    avg_nabla_w = Array.tabulate(lrfSpecs.length)(m=>{
      DenseVector.zeros[Double](lrfSpecs(m).lrf_size)
    })

    batchCounter = 0
  }

  /**
    * update the weights from the average corrections collected in previous learnings
    *
    */
  override def update(): Double = {

    lrfSpecs.zipWithIndex.foreach(spec_and_index => {
      val spec = spec_and_index._1
      val m = spec_and_index._2

      spec.w :-= avg_nabla_w(m) * (eta / batchCounter)
      spec.b -= avg_nabla_b(m) * (eta / batchCounter)
    })

    resetBatch()
    nextLayer.get.update()
  }


  /**
    * calculate all feature maps and combine them to a single output vector
    *
    * @param x the vector to classify
    * @return the classification for the given vector according to the recent model
    */
  override def feedForward(x: DV): DV = {
    nextLayer.get.feedForward(z(x))
  }

  private def z ( x: DV): DV =
    DenseVector((0 until n_features).flatMap(calcFMap(_,x)).toArray)

  /**
    * calculate a single feature map
    *
    * @param m the index of the feature map to be produced
    * @param input this layer's input vector
    * @return the resulting feature map
    */
  def calcFMap(m: Int, input: DV): Array[Double] = {
    (0 until lrfSpecs(m).fmap_size).map(k => {
      (0 until lrfSpecs(m).lrf_size).map(j => {
        val d = lrfSpecs(m).dTF(k, j)
          try {
            lrfSpecs(m).w(j) * input(d)
          } catch {
            case e: Exception =>
              throw e
          }
      }).sum + lrfSpecs(m).b
    }).toArray
  }

  /**
    * Determine the feature map that belongs to the given target space index.
    * Note, that the target vector combines all feature maps of this layer.
    * @param t the index of a component of the output vector
    * @return the index of the feature map this component is determined by
    */
  def m_of_t ( t:Int ) : Int = fmapOffsets.zipWithIndex.find(_._1 > t)
    .map(_._2).getOrElse(fmapOffsets.length) - 1

  /**
    *
    * @param x the input vector to learn from
    * @param y the desired classification for the given input
    * @return the back-propagated deltas
    */
  override def feedForwardAndPropBack(x: DV, y: DV): DV = {
    val delta = nextLayer.get.feedForwardAndPropBack( z(x), y)

    // update weights and biases
    batchCounter += 1
    lrfSpecs.zipWithIndex.foreach(spec_and_index => {
      val size = spec_and_index._1.lrf_size
      val m = spec_and_index._2
      val nabla_wm = DenseVector.tabulate(size)(f=>dC_dwmf(x, delta, f, m))

      val nabla_bm = sum(delta.slice(fmapOffsets(m), fmapOffsets(m) + fmapSizes(m)))

      avg_nabla_w(m) :+= nabla_wm
      avg_nabla_b(m) += nabla_bm
    })

    DenseVector.tabulate(inputSize)(d => dC_dx_d ( delta, d))
  }

  def dC_dwmf( x: DV, delta: DV, f: Int, m: Int ): Double = {
    (0 until lrfSpecs(m).fmap_size).map(t=>
      delta(t+fmapOffsets(m)) * x(lrfSpecs(m).dTF(t,f))).sum
  }

  /**
    * calculate dC/dx_d for back propagation
    * @param delta the back propagated deltas
    * @param d the index of the input neuron
    * @return
    */
  def dC_dx_d(delta: DV, d: Int) : Double = {

    // sum over all features
    ( 0 until n_features ).map(m => {
      val lrfCols = lrfSpecs(m).lrf_cols
      val fmapCols = lrfSpecs(m).fmap_cols
      val targetIndices = lrfSpecs(m).lrfTargets(d)
      val lowerRight = targetIndices.last

      // sum over only the contributing deltas - not all
      targetIndices.map(t=>{
        val x = (lowerRight - t) % fmapCols
        val y = (lowerRight - t) / fmapCols
        val f = y * lrfCols + x
        val tm = t + fmapOffsets(m)
        val dt = delta(tm)
        val wf =
          try {
          lrfSpecs(m).w(f)
        } catch {
          case e: Exception =>
            throw e
        }
        dt * wf
      }).sum
    }).sum

  }


  /**
    * all lrf specs define the same input size, so ask any
    *
    * @return the size of the expected input vector
    */
  override def inputSize: Int = lrfSpecs(0).input_size

  /**
    * initialize weights, to be called by the next layer, should continue until the input layer
    */
  override def initialize(): Unit = {
    val requiredOutputSize = n_features * lrfSpecs(0).fmap_size
    assert(requiredOutputSize == nextLayer.get.inputSize, "Can't connect layers. Sizes don't match")
    previousLayer.foreach(_.initialize())
    resetBatch()

  }
}
