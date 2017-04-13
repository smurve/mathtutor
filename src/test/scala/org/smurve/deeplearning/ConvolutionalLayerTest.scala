package org.smurve.deeplearning

import breeze.linalg.DenseVector
import org.scalatest.{FlatSpec, ShouldMatchers}
import org.smurve.deeplearning.layers._

class ConvolutionalLayerTest extends FlatSpec with ShouldMatchers {

  /**
    * These constant LRFs detect the following two 3x2 features
    *
    * o   o             o
    *   o      and    o   o
    *
    * down            up
    */
  private val UP = 1
  private val DOWN = 0
  private val weights_down = DenseVector(1.0, -1, 1, -1, 1, -1)
  private val weights_up = DenseVector(-1.0, 1, -1, 1, -1, 1)
  private val symbols = Array(Array(0, 2, 6), Array(1, 5, 7))
  private val zeroBias = DenseVector.fill(16, 0.0)

  private val lrfSpecs = Array(weights_down, weights_up).map(weights =>
    LocalReceptiveFieldSpec(5, 5, 3, 2, weights = Some(weights), bias = Some(zeroBias)))

  private val cl = new ConvolutionalLayer(lrfSpecs = lrfSpecs)

  "helper function 'image'" should "generate a simple image" in {

    val img_down_arrow_at_1_1 = image(1, 1, DOWN)
    val img_up_arrow_at_2_2 = image(2, 2, UP)
    print(img_down_arrow_at_1_1)
    print(img_up_arrow_at_2_2)
  }

  "the down arrow" should "be identified at pos 4 on the first featuremap" in {

    val ol = new OutputLayer(24)

    val nn = cl || RELU || ol
    //cl.lrfSpecs.foreach(s => println(s.w))
    val res = nn.feedForward(image(1, 1, DOWN))
    //println(res)

    res(4) should be(3) // 3 is the max activation, indicating a full match
  }

  "the up arrow" should "be identified at pos 8 on the second featuremap (offset 12)" in {

    val ol = new OutputLayer(24)
    val nn = cl º RELU º ol
    //cl.lrfSpecs.foreach(s => println(s.w))
    val res = nn.feedForward(image(2, 2, UP))
    //println(res)

    res(12 + 8 ) should be(3) // 3 is the max activation, indicating a full match
  }

  "m_of_t" should "determine the feature map the output with index t is produced with" in {
    val ol = new OutputLayer(24)
    val nn = cl º RELU º ol
    val layer = nn.entry.asInstanceOf[ConvolutionalLayer]
    layer.m_of_t(0) should be (0)
    layer.m_of_t(11) should be (0)
    layer.m_of_t(12) should be (1)
    layer.m_of_t(23) should be (1)
  }


  "lrfs" should "determine those output indices that a particular input neuron contributes to" in {
    lrfSpecs(0).lrfTargets(0) should be(Array(0))
    lrfSpecs(0).lrfTargets(6) should be(Array(0,1,3,4))
    lrfSpecs(0).lrfTargets(12) should be(Array(3,4,5,6,7,8))
  }

  "the upper-left input" should "have a well-defined contribution to the loss function" in {
    val ol = new OutputLayer(24)
    val nn = cl º RELU º ol

    // we're making up a delta that would have arrived from the output layer. This allows us to verify the indexes
    val delta = DenseVector.tabulate(24)(_+1.0)
    val layer = nn.entry.asInstanceOf[ConvolutionalLayer]
    val wd = layer.dC_dx_d(delta, 0)
    wd should be (-12)
  }


  /**
    * We place the given symbol on a 5x5 plane
    *
    * @param pos_x  the x coordinate on the plane
    * @param pos_y  the y coordinate on the plane
    * @param symbol either UP or DOWN
    */
  private def image(pos_x: Int, pos_y: Int, symbol: Int): DenseVector[Double] = {
    assert(pos_x >= 0 && pos_x <= 2)
    assert(pos_y >= 0 && pos_y <= 3)

    DenseVector((0 until 25).map(paint(pos_x + pos_y * 5, symbols(symbol), _)).toArray)
  }

  private def paint(offset: Int, symbol: Array[Int], actual: Int): Double = {
    if (symbol.map(_ + offset).contains(actual)) 1.0 else 0.0
  }

  private def print(img: DV): Unit = {
    val str: String = img.map(v => if (v == 1.0) " o" else " .").toArray.
      zipWithIndex.map(p => p._1 + (if (p._2 % 5 == 4) "\n" else "")).mkString

    println(str)
  }
}
