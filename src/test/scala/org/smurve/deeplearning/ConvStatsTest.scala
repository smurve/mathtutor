package org.smurve.deeplearning

import breeze.linalg._
import org.scalatest.{FlatSpec, ShouldMatchers}
import org.smurve.deeplearning.layers._
import org.smurve.deeplearning.optimizers.{ConstantOptimizer, SignumBasedMomentum}
import org.smurve.deeplearning.stats.{ConvLayerStats, NNStats, OutputLayer}

class ConvStatsTest extends FlatSpec with ShouldMatchers {

  /**
    * These constant LRFs detect the following two 3x2 features
    *
    * o   o             o
    *   o      and    o   o
    *
    * down            up
    */
  private val UP = 1
  private val DOWN = 0
  private val weights_down = DenseVector(1.0, -1, 1, -1, 1, -1)
  private val weights_up = DenseVector(-1.0, 1, -1, 1, -1, 1)
  private val symbols = Array(Array(0, 2, 6), Array(1, 5, 7))

  private val lrfSpecs = Array(weights_down, weights_up).map(weights =>
    LRFSpec(5, 5, 3, 2, weights = None, bias = None))


  "A convolutional layer" should "be able to identify hand-crafted features anywhere on the image" in {

    val conv = new ConvolutionalLayer("conv", lrfSpecs = lrfSpecs, eta = 3)
    val hidden = new DenseLayer(_inputSize = 24, initWith = INIT_WITH_RANDOM,
      opt_b = new ConstantOptimizer(eta = 1), opt_w = new ConstantOptimizer(eta = 1) )

    val batch_size = 100
    val output = new OutputLayer(2, costFunction = CROSS_ENTROPY_B())
    val nn = conv || TAU() || hidden || SIGMOID() || output
    var stats: NNStats = null;
    for ( n <- 1 to 100000) {
      val (img, y) = rndImage
      nn.feedForwardAndPropBack(img, y)
      if ( n % batch_size == 0 ) {
        stats = nn.update()

        val convStats = stats.getStats("conv").get.asInstanceOf[ConvLayerStats]

        println("c=" + round2(stats.recentCost) , "w=" + round2(convStats.nw(0).head), "b=" + round2(convStats.nb(0).head))
      }
    }

    var success = 0
    for ( _ <- 1 to 1000 ) {
      val (img, y) = rndImage
      val res = nn.feedForward(img)
      if ( (y(0) - y(1)) * ( res(0) - res(1)) > 0 ) success += 1
    }
    success = success / 10
    success should be > 90
    println(s"Success: $success %")

    val row0 = hidden.dump._1(0,::)
    val row1 = hidden.dump._1(1,::)
    println (row0.t.toArray.toList.map(v=>(v*10).toInt))
    println (row1.t.toArray.toList.map(v=>(v*10).toInt))
  }

  private def round2(v: DV): DV = v.map(round2)
  private def round2(d: Double): Double = (d*100).toInt/100.0

  private def rndImage = {
    val s = (math.random * 2.0).toInt
    val desired = if ( s == 1 ) DenseVector(1.0,0) else DenseVector(0.0, 1)
    val x = (math.random * 3).toInt
    val y = (math.random * 4).toInt
    (image(x,y,s), desired)
  }

  /**
    * We place the given symbol on a 5x5 plane
    *
    * @param pos_x  the x coordinate on the plane
    * @param pos_y  the y coordinate on the plane
    * @param symbol either UP or DOWN
    */
  private def image(pos_x: Int, pos_y: Int, symbol: Int): DenseVector[Double] = {
    assert(pos_x >= 0 && pos_x <= 2)
    assert(pos_y >= 0 && pos_y <= 3)

    DenseVector((0 until 25).map(paint(pos_x + pos_y * 5, symbols(symbol), _)).toArray)
  }

  private def paint(offset: Int, symbol: Array[Int], actual: Int): Double = {
    if (symbol.map(_ + offset).contains(actual)) 1.0 else 0.0
  }

  private def print(img: DV): Unit = {
    val str: String = img.map(v => if (v == 1.0) " o" else " .").toArray
      .zipWithIndex.map(p => p._1 + (if (p._2 % 5 == 4) "\n" else "")).mkString

    println(str)
  }
}
