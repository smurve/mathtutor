package org.smurve.deeplearning

import breeze.linalg.{DenseVector, sum}
import org.scalatest.{FlatSpec, ShouldMatchers}
import org.smurve.deeplearning.layers._

/**
  */
class FullyConnectedLayerTest extends FlatSpec with ShouldMatchers {


  "a linear Networks" should "exhibit linear behaviour" in {

    val input = new FullyConnectedLayer ( _inputSize = 4, initialValue = 1.0 )
    val hidden1 = new FullyConnectedLayer ( _inputSize = 3, initialValue = 1.0 )
    val hidden2 = new FullyConnectedLayer ( _inputSize = 2, initialValue = 1.0 )
    val out = new OutputLayer( _inputSize = 2, EUCLIDEAN )

    // stack'em: Only now the weights are initialized
    val nn = input º hidden1 º hidden2 º out

    val outVector = nn.feedForward(v(1, 2, 3, 4))

    outVector should be ( v( 69, 69))

    nn.feedForwardAndPropBack(v(1,2,3,4), v(69, 67))

    nn.asInstanceOf[NeuralNetwork].recentLoss should be ( 2 )

  }

  "A linear network" should "easily learn the simple concept of linear separability" in {

    // We're using two hidden layers just to make it a bit more complex. A single layer would do.
    val input = new FullyConnectedLayer ( _inputSize = 4, initWith = INIT_WITH_RANDOM, inputActivation=a_identity)
    val hidden1 = new FullyConnectedLayer ( _inputSize = 3, initWith = INIT_WITH_RANDOM, inputActivation=a_relu )
    val hidden2 = new FullyConnectedLayer ( _inputSize = 2, initWith = INIT_WITH_RANDOM, inputActivation=a_relu )
    val out = new OutputLayer( _inputSize = 2, EUCLIDEAN )

    // stack'em: Only now the weights are initialized
    val nn = input º hidden1 º hidden2 º SIGMOID º out

    // train with 1000 randomly created samples
    for ( _ <- 0 to 1000 ) {
      val nextTest = rnd()
      nn.feedForwardAndPropBack(nextTest._1, nextTest._2)
      println(nn.asInstanceOf[NeuralNetwork].recentLoss)
      nn.update(0.1)
    }

    var sum_good = 0.0
    val N_SAMPLES = 100
    for ( _  <- 0 to N_SAMPLES ) {
      val nextTest = rnd()
      val res = nn.feedForward(nextTest._1)
      //println(nextTest._1)
      val actual = res.data(0) - res.data(1)
      val desired = nextTest._2.data(0) - nextTest._2.data(1)
      println(actual, desired)
      if ( desired * actual > 0 ) sum_good += 1
    }
    println(s"\nSuccess rate: ${sum_good / N_SAMPLES * 100}%")

  }

  private def rnd() = {
    val rndIn = DenseVector.rand[Double](4)
    val x = rndIn * 20.0 - DenseVector.fill[Double](4){10}
    ( x, desired ( x ))
  }

  private def desired ( in: DV ): DV = {
    val x = if ( sum(in) > 0 ) 1.0 else 0.0
    val y = 1.0 - x
    DenseVector(x,y)
  }

  private def v(x: Double*) = DenseVector(x.toArray)
}
