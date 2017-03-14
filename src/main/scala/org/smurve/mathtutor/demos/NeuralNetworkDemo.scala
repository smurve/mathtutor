package org.smurve.mathtutor.demos

import java.awt.Color

import breeze.numerics.sigmoid
import org.smurve.mnist._

/**
  * Created by wgiersche on 14/03/17.
  */
object NeuralNetworkDemo {

  val N_VALIDATE = 100000

  def main(args: Array[String]): Unit = {
    val nn = new NeuralNetwork(Array(2, 5, 5, 2), Array(sigmoid(_), sigmoid(_), sigmoid(_)), INIT_WITH_RANDOM)

    val f = (x:Double) => 0.1 * (x - 3.5) * (x - 1) * (x + 3.5) * x

    val sbch = new SimpleBinaryClassifierHelper(f)
    //4* math.sin(x))

    for (n <- 1 to 1000000) {
      val sample = sbch.nextSample
      nn.train(sample)
      if (n % 300 == 0) {
        nn.update(0.5)
        //println(nn)
      }
    }

    val spd = ScatterPlotDemo
    spd.create()

    var above = Array[(Double, Double)]()
    var below = Array[(Double, Double)]()
    for (_ <- 1 to N_VALIDATE) {
      val sample = sbch.nextSample
      val p = (sample._1.data(0), sample._1.data(1))
      val y = nn.classify(sample._1)
      if ( y.data(0) > y.data(1))
        above = above :+ p
      else
        below = below :+ p
    }

    spd.addSeries("Above", above, Color.RED)
    spd.addSeries("Below", below, Color.DARK_GRAY)
  }
}