package org.smurve.mnist

import breeze.linalg.DenseVector
import org.scalatest.{FlatSpec, ShouldMatchers}
import org.smurve.deeplearning.INIT_WITH_RANDOM
import org.smurve.deeplearning.a_sigmoid

class SimpleBinaryClassifierTest extends FlatSpec with ShouldMatchers {

  //val nn = new NeuralNetwork(Array(2, 2), Array(SIGMOID), INIT_WITH_RANDOM)
  //val nn = new NeuralNetwork(Array(2, 5, 2), Array(SIGMOID, SIGMOID), INIT_WITH_RANDOM)
  val nn = new NeuralNetwork(Array(2, 5, 5, 2), Array(a_sigmoid,a_sigmoid,a_sigmoid), INIT_WITH_RANDOM)

  //val nn = new NeuralNetwork(Array(2, 5, 5, 5, 2), Array(SIGMOID, SIGMOID, SIGMOID, SIGMOID), INIT_WITH_RANDOM)

  val sbch = new SimpleBinaryClassifierHelper(x=>
    (x - 2 ) * ( x + 2 ) * x )
    //4* math.sin(x))

  "A simple binary classifier" should "learn to tell above from below" in {
    for ( n <- 1 to 1000000 ) {
      val sample = sbch.nextSample(maxX = 5)
      nn.train( sample )
      if ( n % 300 == 0 ) {
        nn.update(0.5)
        //println(nn)
      }
    }

    println(nn)

    val N = 20
    var n = 0
    for ( _ <- 0 until N ) {
      if ( checkSample()) {
        n += 1
      }
    }
    println( "" +  n * 100 / N + "% success.")
  }

  def checkSample() : Boolean = {
    val sample = sbch.nextSample(maxX = 5.0)
    val classification = nn.classify(sample._1)
    val class_clear = if ( classification.data(0) > classification.data(1))
      DenseVector(1.0, 0.0)
    else
      DenseVector(0.0, 1.0)
    val ok_nok = class_clear == sample._2
    println ( s"$sample -> $classification : $class_clear ==> $ok_nok" )
    ok_nok
  }
}
