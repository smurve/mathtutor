package org.smurve.mathtutor.demos

import org.smurve.mnist._

object MNISTDemo {

  val N_TRAINING = 60000
  val N_BATCH = 100
  val ETA = 3.0

  // shrunk images
  //val transform : ( MNISTImage ) => MNISTImage = MNISTHelper.shrink
  //val IMAGE_WIDTH = 14

  // regular images
  val transform: (MNISTImage) => MNISTImage = x => x
  val IMAGE_WIDTH = 28


  val IMAGE_HEIGHT = IMAGE_WIDTH
  val IMAGE_SIZE: Int = IMAGE_HEIGHT * IMAGE_WIDTH
  val FIRST_HIDDEN_LAYER_SIZE = 49

  val imgs: MNISTImageFile = new MNISTImageFile("train")
  val lbls: MNISTLabelFile = new MNISTLabelFile("train-labels")

  //val nn = new NeuralNetwork(Array(IMAGE_SIZE, FIRST_HIDDEN_LAYER_SIZE, 100, 10), Array(sigmoid(_), sigmoid(_), sigmoid(_)),INIT_WITH_RANDOM)
  val nn = new NeuralNetwork(Array(IMAGE_SIZE, FIRST_HIDDEN_LAYER_SIZE, 10), Array(sigmoid(_), sigmoid(_)), INIT_WITH_RANDOM)
  //val nn = new NeuralNetwork(Array(IMAGE_SIZE, 10), Array(sigmoid(_)),INIT_WITH_RANDOM)

  def now: Long = System.currentTimeMillis


  /**
    * determine the number with the max confidence
    *
    * @param y the result vector of the network
    * @return the index with the largest value
    */
  def asNumber(y: DV): Int = {
    y.data.zipWithIndex.fold((0.0, 1))((l, r) => if (l._1 < r._1) r else l)._2
  }

  def go(): Unit = main(Array(""))

  def main(args: Array[String]): Unit = {

    val beforeTraining = now

    for (n <- 0 until N_TRAINING) {
      val img = transform(imgs.img(n))
      //println ( img )
      val lbl = lbls.lblAtPos(n)
      nn.train(img.dv, lbl)
      if (n % N_BATCH == 0) {
        nn.update(ETA)
      }
    }
    val timeForTraining = now - beforeTraining
    println(s"Training duration: $timeForTraining")


    var sumSuccess = 0.0
    var sumFailure = 0.0
    for (n <- 0 until 60000) {
      val img = transform(imgs.img(n))
      val res = nn.classify(img.dv)
      val resNumber = asNumber(res)
      val desired = lbls.lv(n)
      if (desired == resNumber) sumSuccess += 1 else sumFailure += 1
    }

    println(s"sucess rate: $sumSuccess/$N_TRAINING = ${sumSuccess / N_TRAINING * 100}% ")


    (0 to 2).foreach(n => {
      val img = transform(imgs.img(n))
      println(img)
      println(lbls.lblAtPos(n))
      println(nn.classify(img.dv))
    })

  }

  /**
    * provides the weights of the input layer as images
    * The weights of the input layer can be seen as immediate image filters. Thus the weights of the
    * input layer that are feeding into the first neuron of the subsequent layer can be seen as what that neuron
    * is actually looking at
    *
    * @param index of the weights
    * @return
    */
  def weightsAsImage(index: Int, width: Int, height: Int): MNISTImage = {
    val data = nn.layers(0).weights(index, ::).t.toArray //.data
    MNISTImage(data.map(w => (math.abs(2 * w) * 256).toByte), width, height)
  }


  def allWeights(): Unit = (0 until FIRST_HIDDEN_LAYER_SIZE).foreach(n => {
    println()
    println(MNISTDemo.weightsAsImage(n, IMAGE_WIDTH, IMAGE_HEIGHT))
  })

}
