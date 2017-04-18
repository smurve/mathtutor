package org.smurve.mathtutor.demos

import org.smurve.deeplearning._
import org.smurve.deeplearning.layers.{AffineLayer, _}
import org.smurve.mnist._
import org.smurve.deeplearning

abstract class ImageFilter {
  val transform: (MNISTImage) => MNISTImage
  val width: Int
}


object MNISTDemo {

  val noFilter = new ImageFilter {
    override val transform: (MNISTImage) => MNISTImage = x => x
    override val width: Int = 28
  }

  val shrink2 = new ImageFilter {
    override val transform: (MNISTImage) => MNISTImage = MNISTHelper.shrink
    override val width: Int = 14
  }

  val shrink4 = new ImageFilter {
    override val transform: (MNISTImage) => MNISTImage = (x: MNISTImage) =>
      MNISTHelper.shrink(MNISTHelper.shrink(x))
    override val width: Int = 7
  }

  private val TRANSFORM = shrink2
  private val N_TRAINING = 60000
  private val N_BATCH = 2
  private val ETA = 0.15
  private val IMAGE_WIDTH = TRANSFORM.width
  private val IMAGE_HEIGHT = IMAGE_WIDTH
  private val IMAGE_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH

  val imgs: MNISTImageFile = new MNISTImageFile("train")
  val lbls: MNISTLabelFile = new MNISTLabelFile("train-labels")

  val input = new AffineLayer("INPUT", IMAGE_SIZE, INIT_WITH_RANDOM, eta = ETA)
  val hidden1 = new AffineLayer("FIRST HIDDEN", 30, INIT_WITH_RANDOM, eta = ETA)
  val hidden2 = new AffineLayer("SECOND HIDDEN", 30, INIT_WITH_RANDOM, eta = ETA)
  val output = new deeplearning.layers.OutputLayer(10, CROSS_ENTROPY)

  private val NN = input º SIGMOID º hidden1 º SIGMOID º hidden2 º SIGMOID º output

  def now: Long = System.currentTimeMillis

  def go(): Unit = main(Array(""))

  def main(args: Array[String]): Unit = {

    val beforeTraining = now

    val repeat = 3

    for {
      n <- 0 until N_TRAINING
      r <- 0 until repeat
    } {
      val variation =
        if (r == 1)
          MNISTHelper.shearHorizontal(imgs.img(n))
        else if (r == 2)
          MNISTHelper.squeeze(imgs.img(n))
        else
          imgs.img(n)

      val orig = MNISTHelper.sharpen(variation, 127)

      val img = TRANSFORM.transform(orig)
      val lbl = lbls.lblAtPos(n)
      NN.feedForwardAndPropBack(img.dv, lbl)
      if (n % N_BATCH == 0) {
        val avgCost = NN.update()
        println(avgCost)
      }
    }
    val timeForTraining = now - beforeTraining
    println(s"Training duration: $timeForTraining")

    var sumSuccess2 = 0.0
    var sumFailure2 = 0.0
    for (n <- 0 until N_TRAINING) {
      val img = TRANSFORM.transform(imgs.img(n))
      val res2 = NN.feedForward(img.dv)
      val resNumber2 = asNumber(res2)
      val desired = lbls.lv(n)
      if (desired == resNumber2) sumSuccess2 += 1 else sumFailure2 += 1
    }

    //println(s"sucess rate previously: $sumSuccess1 /$N_TRAINING = ${sumSuccess1 / N_TRAINING * 100}% ")
    println(s"sucess rate now: $sumSuccess2/$N_TRAINING = ${sumSuccess2 / N_TRAINING * 100}% ")


    imgs.imgs.zipWithIndex.foreach(p => {
      val img = shrink2.transform(p._1)
      val idx = p._2
      val desired = lbls.lv(idx)
      val res = valueOf(NN.feedForward(img.dv))
      if ( res != desired) {
        println(img)
        println(s"Labelled   as $desired")
        println(s"Classified as $res")
      }
    })


  }


  def valueOf(output: DV): Int = {
    output.toArray.zipWithIndex.foldLeft((0.0, 0))(
      (r, c) => if (r._1 > c._1) (r._1, r._2) else (c._1, c._2))._2
  }

}
