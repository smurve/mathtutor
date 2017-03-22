package org.smurve.mnist

/**
  * cross-and-circle image generator
  * generates 16x16 pixel random images of 0 to 3 diamonds and 0 to 3 crosses
  */
object C_C_I_Generator {

  val cross = Array((-2,-2), (2,-2), (-1,-1), (1,-1), (0,0), (-1,1), (1,1), (-2,2), (2,2))
  val circle = Array(
    (-2, -1), (-2,0), (-2,1), (1,-2),(0,-2), (-1,-2),
    (2,  -1), (2, 0), (2, 1), (1, 2),(0, 2), (-1, 2) )


  def nextImage ( ): (MNISTImage, Int) = {

    val max_circles: Int = 3
    val max_crosses: Int = 3

    val bytes = Array.fill[Byte](256)(0)

    val ncircles = rand(max_circles + 1)
    val ncrosses = rand(max_crosses + 1)

    for ( _ <- 0 until ncircles ) paint ( bytes, circle )
    for ( _ <- 0 until ncrosses ) paint ( bytes, cross )

    val category = 4 * ncircles + ncrosses

    ( MNISTImage(bytes, 16, 16), category)
  }

  private def paint ( img: Array[Byte], pattern: Array[(Int, Int)]) : Unit = {
    val x = 2 + rand(12)
    val y = 2 + rand(12)
    for ( p <- pattern ) {
      img(x + p._1 + 16 * (y+p._2)) = 127
    }
  }

  private def rand ( range: Int) : Int = math.floor(math.random * range ).toInt
}
