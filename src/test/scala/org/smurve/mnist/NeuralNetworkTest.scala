package org.smurve.mnist

import breeze.linalg.DenseVector
import org.scalatest.{FlatSpec, ShouldMatchers}
import org.smurve.deeplearning.a_sigmoid
import org.smurve.deeplearning.EUCLIDEAN
import org.smurve.deeplearning.INIT_WITH_RANDOM
import org.smurve.deeplearning.INIT_WITH_CONST

class NeuralNetworkTest extends FlatSpec with ShouldMatchers {

  "A simple network" should "feed forward from input directly to output" in {
    val nn = new NeuralNetwork(Array(4,2), Array(a_sigmoid), INIT_WITH_RANDOM)
    val output = nn.classify(DenseVector(-2.0, -1.0, 1.0, 2))
    output.length should be ( 2)

  }

  "A 3-layer network" should "feed forward from input via one hidden layer output" in {
    val nn = new NeuralNetwork(Array(4,8, 3), Array(a_sigmoid, a_sigmoid), INIT_WITH_CONST)
    val output = nn.classify(DenseVector(-2.0, -1.0, 1.0, 2))
    output.length should be ( 3)
  }

  "The Adamard product" should "multiply elementwise" in {
    val v1 = DenseVector(2.0, 4.0)
    val v2 = DenseVector(3.0, 5.0)
    v1 :* v2 should be (DenseVector(6.0, 20.0))
  }

  "The squared euclidian difference" should "make a good cost function" in {
    EUCLIDEAN.fn(DenseVector(2,1), DenseVector(2,1)) should be ( 0)
    EUCLIDEAN.fn(DenseVector(2,1), DenseVector(3,1)) should be ( .5)
    EUCLIDEAN.fn(DenseVector(2,1), DenseVector(2,0)) should be ( .5)
    EUCLIDEAN.deriv(DenseVector(2,1), DenseVector(2,1)) should be ( DenseVector(0.0,0.0))
    EUCLIDEAN.deriv(DenseVector(2,1), DenseVector(3,1)) should be ( DenseVector(-1.0, 0.0))
    EUCLIDEAN.deriv(DenseVector(2,1), DenseVector(2,0)) should be ( DenseVector(0.0, 1.0))
  }


}
