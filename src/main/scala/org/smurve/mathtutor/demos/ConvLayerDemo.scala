package org.smurve.mathtutor.demos

import breeze.linalg.DenseVector
import org.smurve.charts.{LineChart, PlotterSpec}
import org.smurve.mnist._
import org.smurve.deeplearning._
import org.smurve.deeplearning.layers.LRFSpec
/**
  * Created by wgiersche on 22/03/17.
  */
object ConvLayerDemo {

  val size_output = 16

  def go() : Unit = main(Array(""))

  var convLayer: MNISTLayer = _

  def main(args: Array[String]): Unit = {

    val frame = LRFSpec (16, 16, 5, 5)

    val num_features = 2

    val size_all_feature_maps = num_features

    val hidden1 = new FullyConnectedLayer(
      inputSize = size_all_feature_maps * frame.fmap_size,
      outputSize = size_output,
      activation = a_sigmoid
    )

    val


    convLayer = new ConvNetworkLayer(
      next = Some(hidden1),
      frame = frame,
      num_features = num_features)

    val sample = C_C_I_Generator.nextImage
    val y = DenseVector.tabulate(size_output){i=>if(i==sample._2) 1.0 else 0.0}
    println(sample)
    convLayer.feedForwardAndPropBack(sample._1.dv, y)

    for ( i <- 1 to 50000 ) {
      if ( i % 100 == 0)
        convLayer.update(0.0003)
      val t = nextTrainingRecord
      convLayer.feedForwardAndPropBack(t._1.dv, t._2)
    }


    val N_TEST = 5000
    var sumSuccess = 0.0
    var sumFailure = 0.0
    for (n <- 0 until N_TEST) {
      val sample = nextTrainingRecord
      val res = convLayer.feedForward(sample._1.dv)
      val result = asNumber(res)
      val desired = asNumber(sample._2)
      if ( desired == result ) sumSuccess += 1 else sumFailure += 1
    }

    println(s"sucess rate: $sumSuccess/$N_TEST = ${sumSuccess / N_TEST * 100}% ")



    val N_DEMO = 20
    sumSuccess = 0.0
    sumFailure = 0.0
    for (n <- 0 until N_TEST) {
      val sample = nextTrainingRecord
      val res = convLayer.feedForward(sample._1.dv)
      val result = asNumber(res)
      val desired = asNumber(sample._2)
      println(sample._1)
      println(s"Actual: $desired")
      println(s"result vector: $res")
      println(s"as number: $result")
      println
      if ( desired == result ) sumSuccess += 1 else sumFailure += 1
    }

    println(s"sucess rate: $sumSuccess/$N_TEST = ${sumSuccess / N_TEST * 100}% ")


    val chart = new LineChart()(PlotterSpec())

    chart.showData(hidden1.costByTime)
  }

  def nextTrainingRecord: (MNISTImage, DenseVector[Double]) = {
    val sample = C_C_I_Generator.nextImage()
    val y = DenseVector.tabulate(size_output){i=>if(i==sample._2) 1.0 else 0.0}
    (sample._1,y)
  }
}
