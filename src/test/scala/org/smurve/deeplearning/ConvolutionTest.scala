package org.smurve.deeplearning

import breeze.linalg.{DenseVector, max, min}
import org.scalatest.{FlatSpec, ShouldMatchers}
import org.smurve.deeplearning.layers.LocalReceptiveFieldSpec
import org.smurve.mnist.{ConvNetworkLayer, MNISTImage}

/**
  * Created by wgiersche on 19/03/17.
  */
class ConvolutionTest extends FlatSpec with ShouldMatchers {

  "An LRF spec" should "calculate output sizes correctly" in {
    val frame = LocalReceptiveFieldSpec(input_cols = 28, input_rows = 28, lrf_cols = 4, lrf_rows = 3)
    frame.fmap_size should be((28 - 4 + 1) * (28 - 3 + 1))
  }

  "An LRF spec" should "calculate the domain index from the feature map index correctly" in {
    val frame = LocalReceptiveFieldSpec(input_cols = 28, input_rows = 28, lrf_cols = 4, lrf_rows = 3)
    frame.dTF(27, 0) should be(28 + 2)
  }

  "A convolution network" should "scan the input vector" in {
    val input = DenseVector(
      1.0, 2, 3, 4, 5,
      6, 7, 8, 9, 10,
      11, 12, 13, 14, 15,
      16, 17, 18, 19, 20,
      21, 22, 23, 24, 25)

    val frame = LocalReceptiveFieldSpec(5, 5, 3, 2)

    val layer = new ConvNetworkLayer(frame = frame, num_features = 1, activation = a_sigmoid, costDerivative = None)

    layer.setFeatures(Array(DenseVector(1, 1, 1, 1, 1, 1)))

    val features = layer.convolute(0, input)

    features should be(Array(27.0, 33, 39, 57, 63, 69, 87, 93, 99, 117, 123, 129))
  }

  "A convolutional network" should "localize certain features" in {
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

    val frame = LocalReceptiveFieldSpec(8, 8, 3, 3)

    val layer = new ConvNetworkLayer(frame = frame, num_features = 2, activation = a_sigmoid, costDerivative = None)

    layer.setFeatures(Array(
      DenseVector(
        0, 1, 0.0,
        1, 0, 1,
        1, 1, 1),
      DenseVector(
        1, 0, 1.0,
        0, 1, 0,
        1, 0, 1))
    )

    println("Circles:")
    println(MNISTImage(toByteArray(DenseVector(layer.convolute(0, input))), 6, 6))
    println("Crosses:")
    println(MNISTImage(toByteArray(DenseVector(layer.convolute(1, input))), 6, 6))

  }


  "toByteArray" should "linearly scale any vector to elements with values between 0 and 255" in {
    val res = toByteArray(DenseVector(-4, -2, -1, 4.0))
    res should be(Array(0, 63, 95, -1))
  }

  def toByteArray(v: DV): Array[Byte] = {
    val maxV = max(v)
    val minV = min(v)

    def scale = 255 / (maxV - minV)

    v.toArray.map(x => ((x - minV) * scale).toByte)
  }
}
