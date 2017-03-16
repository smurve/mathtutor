package org.smurve.mathtutor.demos

import java.awt.Color

import org.smurve.mnist._

/**
  * Created by wgiersche on 14/03/17.
  */
object NeuralNetworkDemo {

  private def now = System.currentTimeMillis

  val N_VALIDATE = 20000

  def main(args: Array[String]): Unit = {
    val nn = new NeuralNetwork(Array(2, 5, 5, 2), Array(SIGMOID, SIGMOID, SIGMOID), INIT_WITH_RANDOM)

    val f = (x:Double) => 0.1 * (x - 3.5) * (x - 1) * (x + 3.5) * x

    val fplot = (0 until 10000).map(i=>{
      val x = -5.0 + i/1000.0
      (x, f(x))
    }).filter(_._2<7).toArray

    val sbch = new SimpleBinaryClassifierHelper(f)
    //4* math.sin(x))

    val beforeTraining = now
    for (n <- 1 to 1000000) {
      // train only up to x = 3
      val sample = sbch.nextSample(maxX = 3.0)
      nn.train(sample)
      if (n % 300 == 0) {
        nn.update(0.5)
      }
    }
    val timeForTraining = now - beforeTraining
    println(s"Training duration: $timeForTraining")

    val spd = ScatterPlotDemo
    spd.create()

    var above = Array[(Double, Double)]()
    var below = Array[(Double, Double)]()
    for (_ <- 1 to N_VALIDATE) {
      val sample = sbch.nextSample(maxX = 5.0)
      val p = (sample._1.data(0), sample._1.data(1))
      val y = nn.classify(sample._1)
      if ( y.data(0) > y.data(1))
        above = above :+ p
      else
        below = below :+ p
    }

    spd.addSeries("Above", above, Color.RED)
    spd.addSeries("Below", below, Color.DARK_GRAY)
    spd.addSeries("The Function", fplot, Color.BLACK)
  }
}