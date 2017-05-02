package org.smurve.deeplearning

import breeze.linalg.{DenseVector, max, min}
import org.scalatest.{FlatSpec, ShouldMatchers}
import org.smurve.deeplearning.layers.{ConvolutionalLayer, LRFSpec}
import org.smurve.mnist.MNISTImage

/**
  * Created by wgiersche on 19/03/17.
  */
class ConvolutionTest extends FlatSpec with ShouldMatchers {

  "An LRF spec" should "calculate output sizes correctly" in {
    val frame = LRFSpec(input_cols = 28, input_rows = 28, lrf_cols = 4, lrf_rows = 3)
    frame.fmap_size should be((28 - 4 + 1) * (28 - 3 + 1))
  }


  "An LRF spec's dTF function" should
    "calculate the domain index from the feature map (f) and target (t) index correctly" in {

    val frame = LRFSpec(input_cols = 28, input_rows = 28, lrf_cols = 4, lrf_rows = 3)
    /**
      *     Domain: width 28
      *   _ _ _ _ _ _ _
      *  |
      *  |    o o o o       LRF:: 4x3
      *  |    o o o o
      *  |    o o o o
      *
      *
      *     Contributes to target: width 25
      *   _ _ _ _ _ _
      *  |
      *  |    o
      */

    /**      | target t |  lrf f   |           domain d          */
    frame.dTF(1 * 25 + 2, 0 * 4 + 0) should be(1 * 28 + 2)
    frame.dTF(1 * 25 + 2, 0 * 4 + 1) should be(1 * 28 + 2 + 1)
    frame.dTF(1 * 25 + 2, 0 * 4 + 3) should be(1 * 28 + 2 + 3)
    frame.dTF(1 * 25 + 2, 1 * 4 + 0) should be(2 * 28 + 2)
    frame.dTF(1 * 25 + 2, 1 * 4 + 3) should be(2 * 28 + 2 + 3)
    frame.dTF(1 * 25 + 2, 2 * 4 + 1) should be(3 * 28 + 2 + 1)
    frame.dTF(1 * 25 + 2, 2 * 4 + 3) should be(3 * 28 + 2 + 3)

    /** lrf index > 11 */
    an [AssertionError] should be thrownBy frame.dTF(1 * 25 + 2, 2 * 4 + 4)

    /** domain index >= 28 * 28 */
    an [AssertionError] should be thrownBy frame.dTF(785, 0)
  }


  "A convolutional layer" should "convolve the input vector" in {
    val input = DenseVector(
      1,  2,  3,  4,  5,
      6,  7,  8,  9,  10,
      11, 12, 13, 14, 15,
      16, 17, 18, 19, 20,
      21, 22, 23, 24, 25.0)

    val expectedResult = Array(
      27, 33, 39,
      57, 63, 69,
      87, 93, 99,
      117, 123, 129.0)

    val spec = LRFSpec(5, 5, 3, 2,
      /** this feature basically adds up all numbers within the LRF*/
      weights = Some(DenseVector(1, 1, 1, 1, 1, 1)),
      bias = Some(0.0))

    val anyEta = .1 // not used

    val layer = new ConvolutionalLayer(name="conv", Array(spec), anyEta)

    val features = layer.calcFMap(spec, input)

    features should be(expectedResult)
  }



  "A convolutional layer" should "localize given features in an image" in {

    val anyEta = .1 // not used

    /**
      * Two "houses" centered around (2,2) and (5,6) and a cross centered at (1,6)
      * */
    val input = DenseVector(
      0.0, 0, 0, 0, 0, 0, 0, 1,
      0.0, 0, 1, 0, 0, 0, 0, 0,
      0.0, 1, 0, 1, 0, 0, 1, 0,
      0.0, 1, 1, 1, 0, 0, 0, 0,
      0.0, 0, 0, 0, 0, 0, 0, 0,
      1.0, 0, 1, 0, 0, 1, 0, 0,
      0.0, 1, 0, 0, 1, 0, 1, 0,
      1.0, 0, 1, 0, 1, 1, 1, 0
    )

    /** resulting in dominant activations of the feature map for houses at (1,1) and (4,5) */
    val exp_h1 = (1,1)
    val exp_h2 = (4,5)
    /** and a dominant activation for a cross at (0,5)*/
    val exp_c = (0,5)

    val house = LRFSpec(8,8,3,3,
      weights = Some(DenseVector(
        0, 1, 0.0,
        1, 0, 1,
        1, 1, 1)
      ),
      bias = Some(0.0))

    val cross = LRFSpec(8,8,3,3,
      weights = Some(DenseVector(
        1, 0, 1.0,
        0, 1, 0,
        1, 0, 1)
      ),
      bias = Some(0.0))


    val layer = new ConvolutionalLayer(name="conv", lrfSpecs = Array(house, cross), anyEta)
    def imageFrom(spec: LRFSpec) = MNISTImage(toByteArray(DenseVector(layer.calcFMap(spec, input))), 6, 6)

    val res_house = imageFrom(house)
    println(s"Houses:\n$res_house")
    res_house.bytes(exp_h1._1 + 6 * exp_h1._2) should be ( -1 )
    res_house.bytes(exp_h2._1 + 6 * exp_h2._2) should be ( -1 )


    val res_cross = imageFrom(cross)
    println(s"Crosses:\n$res_cross")
    res_cross.bytes(exp_c._1 + 6 * exp_c._2) should be ( -1 )
  }


  def toByteArray(v: DV): Array[Byte] = {
    val maxV = max(v)
    val minV = min(v)

    def scale = 255 / (maxV - minV)

    v.toArray.map(x => ((x - minV) * scale).toByte)
  }

  "toByteArray" should "linearly scale any vector to elements with values between 0 and 255" in {
    val res = toByteArray(DenseVector(-4, -2, -1, 4.0))
    res should be(Array(0, 63, 95, -1))
  }

}
