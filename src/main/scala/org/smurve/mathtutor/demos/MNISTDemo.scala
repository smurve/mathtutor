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
      MNISTHelper.shrink (MNISTHelper.shrink(x))
    override val width: Int = 7
  }

  private val TRANSFORM = shrink2
  private val N_TRAINING = 30000
  private val N_BATCH = 50
  private val ETA = 3.0
  private val INIT_WITH = INIT_WITH_RANDOM
  private val IMAGE_WIDTH = TRANSFORM.width
  private val IMAGE_HEIGHT = IMAGE_WIDTH
  private val IMAGE_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH

  val imgs: MNISTImageFile = new MNISTImageFile("train")
  val lbls: MNISTLabelFile = new MNISTLabelFile("train-labels")

  val input = new AffineLayer("INPUT", IMAGE_SIZE, INIT_WITH, initialValue = 0.001)
  val hidden1 = new AffineLayer("FIRST HIDDEN", 30, INIT_WITH, initialValue = 0.001 )
  val hidden2 = new AffineLayer("FIRST HIDDEN", 30, INIT_WITH, initialValue = 0.001 )
  val output = new deeplearning.layers.OutputLayer(10, EUCLIDEAN)

  private val NN = input º SIGMOID º hidden1 º SIGMOID º hidden2 º SIGMOID º output

  def now: Long = System.currentTimeMillis

  def go(): Unit = main(Array(""))

  def main(args: Array[String]): Unit = {

    val beforeTraining = now

    val repeat = 1

    for {
      n <- 0 until N_TRAINING
      _ <- 0 until repeat
    } {
      val img = TRANSFORM.transform(imgs.img(n))
      val lbl = lbls.lblAtPos(n)
      NN.feedForwardAndPropBack(img.dv, lbl)
      if (n % N_BATCH == 0) {
        NN.update(ETA)
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


    (0 until 0).foreach(n => {
      val img = TRANSFORM.transform(imgs.img(n))
      println(img)
      println(lbls.lblAtPos(n))
      //println(nn.classify(img.dv).toString() + " = " + valueOf (nn.classify(img.dv)))
      println(NN.feedForward(img.dv).toString() + " = " + valueOf (NN.feedForward(img.dv)))
    })

  }


  def valueOf ( output: DV ) : String = {
    output.toArray.zipWithIndex.foldLeft((0.0,0))(
      (r,c)=>if(r._1>c._1) (r._1,r._2) else (c._1,c._2))._2.toString
  }

}
