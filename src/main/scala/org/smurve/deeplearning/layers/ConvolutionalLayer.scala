package org.smurve.deeplearning.layers

import breeze.linalg.DenseVector
import org.smurve.deeplearning._

/**
  * A convolutional layer
  * This implementation supports any number of LRFs of different sizes. These dimensions of these have to be specified
  * in the constructor argument as an array of LocalReceptiveFieldSpecs. The output comprises of a vector containing
  * the feature maps in the same order that their LRFs are specified in the constructor.
  */
class ConvolutionalLayer(val lrfSpecs: Array[LocalReceptiveFieldSpec])

  extends Layer {

  val n_features: Int = lrfSpecs.length

  val fmapSizes: Array[Int] = lrfSpecs.map(_.fmap_size)
  val fmapOffsets: Array[Int] = fmapSizes.zipWithIndex.map(x=>fmapSizes.slice(0,x._2).sum)

  /**
    * update the weights from the average corrections collected in previous learnings
    *
    * @param eta : the learning factor
    */
  override def update(eta: Double): Double = ???

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
        lrfSpecs(m).w(j) * input(lrfSpecs(m).dTF(k, j))
      }).sum + lrfSpecs(m).b(k)
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

    DenseVector.tabulate(inputSize)(d => dC_dx_d ( delta, d))
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
      val targetIndices = lrfSpecs(m).lrfTargets(d)
      val lowerRight = targetIndices.last

      // sum over all contributing deltas - not all
      targetIndices.map(t=>{
        val x = (lowerRight - t) % lrfCols
        val y = (lowerRight - t) / lrfCols
        val f = y * lrfCols + x
        val tm = t + fmapOffsets(m)
        val dt = delta(tm)
        val wf = lrfSpecs(m).w(f)
        dt * wf
      }).sum
    }).sum


    /*
    val wd = weights_and_deltas (delta, d)
    wd.map(wd => wd._1 * wd._2).sum
    */
  }

  /* *
    * Determine the contributions of neuron d to the recent loss
    * @param delta the delta as arriving from the subsequent layer
    * @param d an index from the input layer
    * @return a list of the components of delta and the weights that determine the
    *         contribution of the d-th neuron to the loss function
    * /
  def wights_and_deltas ( delta: DV, d:Int): Array[(Double, Double, Int)] =
    delta.data.zipWithIndex.map(dt => (dt._1, w_eta(m_of_t(dt._2), d, dt._2), dt._2))


  / **
    * find the single weight of feature_m that contributes to dC/dx_d
    *  /
  def w_eta(m: Int, d: Int, t:Int) : Double = {
    val spec = lrfSpecs(m)

    val allds = (0 until spec.lrf_size).map(f => (f, spec.dTF(t, f)))


    val f_non_zero = (0 until spec.lrf_size).filter(f => d == spec.dTF(t, f)).head
    spec.w(f_non_zero)
  }
    */

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
  }
}
