package org.smurve.mnist

import breeze.linalg.DenseVector

/**
  * simply classify points as to be above/below a certain function
  */
class SimpleBinaryClassifierHelper( f: Double=> Double) {

  def nextSample : (DV, DV) = {

    val x: DV = (DenseVector.rand(2) * 10.0) - DenseVector.fill(2){5.0}
    (x, classify(x))
  }

  private def classify ( p: DV ) : DV = {
    val x = p.data(0)
    val y = p.data(1)

    if ( y > f(x) )
      DenseVector(1.0, 0.0)
    else
      DenseVector(0.0, 1.0)
  }

  def train ( nn: NeuralNetwork ) : Unit = {
    for ( n <- 1 to 1000000 ) {
      val sample = nextSample
      nn.train( sample )
      if ( n % 300 == 0 ) {
        nn.update(0.5)
        println(nn)
      }
    }
  }

  def validate ( nn: NeuralNetwork ) : Unit = {
    for {
      nx <- -100 to 100
      ny <- -100 to 100

    } yield {
      val x = nx * 0.05
      val y = -ny * 0.05
      (x,y, nn.classify(DenseVector(x, y)))
    }

  }
}

