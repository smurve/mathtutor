package org.smurve.deeplearning

import breeze.linalg.{DenseVector, sum}
import org.scalatest.{FlatSpec, ShouldMatchers}

/**
  */
class FullyConnectedLayerTest extends FlatSpec with ShouldMatchers {


  "a linear Networks" should "exhibit linear behaviour" in {

    val input = new FCL ( inputSize = 4, initialValue = 1.0 )
    val hidden1 = new FCL ( inputSize = 3, initialValue = 1.0 )
    val hidden2 = new FCL ( inputSize = 2, initialValue = 1.0 )
    val out = new OL( inputSize = 2, activation = UNIT, EUCLIDEAN )

    // stack'em: Only now the weights are initialized
    val nn: NeuralNetwork = input º hidden1 º hidden2 º out

    val outVector = nn.feedForward(v(1, 2, 3, 4))

    outVector should be ( v( 69, 69))

    nn.feedForwardAndPropBack(v(1,2,3,4), v(69, 67))

    nn.recentLoss should be ( 2 )

  }

  "A linear network" should "learn a simple concept" in {

    val input = new FCL ( inputSize = 4, initWith = INIT_WITH_RANDOM)
    val hidden1 = new FCL ( inputSize = 3, initWith = INIT_WITH_RANDOM )
    val hidden2 = new FCL ( inputSize = 2, initWith = INIT_WITH_RANDOM )
    val out = new OL( inputSize = 2, activation = SIGMOID, EUCLIDEAN )

    // stack'em: Only now the weights are initialized
    val nn: NeuralNetwork = input º hidden1 º hidden2 º out

    for ( _ <- 0 to 1000 ) {
      val nextTest = rnd()
      nn.feedForwardAndPropBack(nextTest._1, nextTest._2)
      println(nn.recentLoss)
      nn.update(0.1)
    }

    for ( _  <- 0 to 10 ) {
      val nextTest = rnd()
      val res = nn.feedForward(nextTest._1)
      println(nextTest._1)
      println(res.data(0) - res.data(1), nextTest._2.data(0) - nextTest._2.data(1))
    }

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
