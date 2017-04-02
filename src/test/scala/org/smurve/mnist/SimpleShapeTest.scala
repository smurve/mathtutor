package org.smurve.mnist

import org.scalatest.{FlatSpec, ShouldMatchers}
import org.smurve.deeplearning.INIT_WITH_RANDOM
import org.smurve.deeplearning.SIGMOID

class SimpleShapeTest extends FlatSpec with ShouldMatchers {

  "A simple network" should "learn to tell horizontal from diag from vertical structures" in {
    val nn = new NeuralNetwork(Array(4,7, 3), Array(SIGMOID, SIGMOID), INIT_WITH_RANDOM)

    for ( n <- 0 to 500 ) {
      val sample = SimpleShapeHelper.nextSample
      nn.train( sample._1, sample._2)
      if ( n % 10 == 0 )
        nn.update(0.2)
    }

    for ( _ <- 0 to 50 ) {
      val sample = SimpleShapeHelper.nextSample
      val prediction = nn.classify(sample._1)
      println ( s"$sample -> $prediction")
    }

  }


}
