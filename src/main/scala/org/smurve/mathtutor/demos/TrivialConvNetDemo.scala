package org.smurve.mathtutor.demos

import breeze.linalg.DenseVector
import org.smurve.deeplearning._
import org.smurve.deeplearning.layers._
import org.smurve.deeplearning.optimizers.ConstantOptimizer
import org.smurve.deeplearning.stats.OutputLayer
import org.smurve.deeplearning.utilities.ImageGenerator

/**
  * The network is supposed to detect the following two 3x2 features anywhere on the test images
  *
  * o   o             o
  *   o      and    o   o
  *
  * down            up
  */
object TrivialConvNetDemo {

  private val upImg = DenseVector(0.0, 1, 0, 1.0, 0, 1)
  private val downImg = DenseVector(1.0, 0, 1, 0, 1.0, 0)

  def main(args: Array[String]): Unit = {

    val imgW = 4 // 12
    val imgH = 3 // 11
    val eta_c = 0.1 // 2
    val eta_d = 1 // 6

    val randSpecs = Array(
      LRFSpec(imgW, imgH, 3, 2),
      LRFSpec(imgW, imgH, 3, 2))

    val conv = new ConvolutionalLayer(name="conv", lrfSpecs = randSpecs, eta = eta_c)
    val pooling = new PoolingLayer(name="pool", outputWidth = 2)
    val inputSize = (imgW - 3 + 1) * (imgH - 2 + 1) * 2

    val dense = new DenseLayer(_inputSize = inputSize / 4, initWith = INIT_WITH_RANDOM,
      opt_b = new ConstantOptimizer(eta_d), opt_w = new ConstantOptimizer(eta_d))

    val ol = new OutputLayer(2, costFunction = CROSS_ENTROPY_B())

    val nn = conv || TAU() || pooling || dense || SIGMOID() || ol

    val N_train = 30000
    val gen = new ImageGenerator(imgW, imgH)
    for (n <- 1 until N_train) {
      val img = gen.rndImage(gen.UP, gen.DOWN)
      val y = if (img.symbol == gen.UP) DenseVector(1.0, 0) else DenseVector(0, 1.0)
      nn.feedForwardAndPropBack(img.asDV, y)
      if (n % 20 == 0) {
        val stats = nn.update()
        println(stats.recentCost)
      }
    }

    var success = 0.0
    val N_test = 1000
    for (_ <- 1 to N_test) {
      val img = gen.rndImage(gen.UP, gen.DOWN)
      val y = if (img.symbol == gen.UP) DenseVector(0.8, 0) else DenseVector(0, 0.8)
      val res = nn.feedForward(img.asDV)
      if ((y(0) - y(1)) * (res(0) - res(1)) > 0) {
        success += 1
      }
    }
    success /= ( N_test / 100 )
    println("==========================================================")
    println(s"Success: $success%")


    println("hidden layer")
    val row0 = dense.dump._1(0, ::)
    val row1 = dense.dump._1(1, ::)
    println(row0.t.toArray.toList.map(v => (v * 100).toInt / 100.0))
    println(row1.t.toArray.toList.map(v => (v * 100).toInt / 100.0))

    println("convolutional layer")
    val w0 = conv.lrfSpecs(0).w
    val b0 = conv.lrfSpecs(0).b
    val w1 = conv.lrfSpecs(1).w
    val b1 = conv.lrfSpecs(1).b

    val w0_u = w0.t * upImg + b0
    val w0_d = w0.t * downImg + b0
    val w1_u = w1.t * upImg + b1
    val w1_d = w1.t * downImg + b1

    val w0i = w0.toArray.map(v => (v * 100).toInt / 100.0).toList
    val w1i = w1.toArray.map(v => (v * 100).toInt / 100.0).toList

    println(s"w0 =  $w0i, b0 = $b0, w0*up = $w0_u, w0*down = $w0_d")
    println(s"w1 =  $w1i, b1 = $b1, w1*up = $w1_u, w1*down = $w1_d")

    // can't guarantee, mostly it works, sometimes gets stuck in a
    //success > 80 should be(true)
  }

}
