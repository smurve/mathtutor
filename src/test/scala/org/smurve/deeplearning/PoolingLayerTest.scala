package org.smurve.deeplearning

import breeze.linalg.DenseVector
import org.scalatest.{FlatSpec, ShouldMatchers}
import org.smurve.deeplearning.layers._
import org.smurve.deeplearning.stats.OutputLayer

/**
  * Created by wgiersche on 18.04.17.
  */
class PoolingLayerTest extends FlatSpec with ShouldMatchers {

  val pl = new PoolingLayer("pool", stride = 2, poolWidth = 2, poolHeight = 2, outputWidth = 2, function = MAX_POOLING)
  val ol = new OutputLayer(size = 4)
  private val nn = pl || ol
  val x = DenseVector(
    1.0, 2.0, 5.0, 6.0,
    4.0, 3.0, 7.0, 8.0,
    1.0, 6.0, 8.0, -1.0,
    3.0, 5.0, 7.0, 6.0
  )

  val expected = DenseVector(4.0, 8.0, 6.0, 8.0)

  "A pooling layer" should "initialize correctly." in {

    nn.inputSize should be(16)
  }

  "A pooling layer" should "produce the reduced output vector consisting of pooled Doubles" in {
    nn.feedForward(x) should be(expected)

  }

  "The backpropagated deltas" should "show up in the max values' positions" in {

    val delta = nn.feedForwardAndPropBack(x, expected :- DenseVector(2.0,3,4,5))

    delta should be(DenseVector(
      0.0, 0, 0, 0,
      2.0, 0, 0, 3,
      0.0, 4, 5, 0,
      0.0, 0, 0, 0))
  }
}
