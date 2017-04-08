package org.smurve.deeplearning

import breeze.linalg.DenseVector
import org.scalatest.{FlatSpec, ShouldMatchers}
import org.smurve.deeplearning.layers._

class ConvolutionalLayerTest extends FlatSpec with ShouldMatchers {

  /**
    *   These constant LRFs detect the following two 3x2 features
    *
    *   o   o       o
    *     o   and o   o
    *    down       up
    */
    private val UP = 1
  private val DOWN = 0
  private val constantLRF = Array ( DenseVector(1.0, -1, 1, -1, 1, -1 ), DenseVector(-1.0, 1, -1, 1, -1, 1 ))
  private val symbols = Array( Array(0,2,6) , Array(1,5,7))

  private val cl = new ConvolutionalLayer(
    initWith = INIT_WITH_CONST, initialValue = Some(constantLRF),
    inputActivation = a_identity,
    lrf = LocalReceptiveFieldSpec(5, 5, 3, 2), n_features = 2)

  private val hidden1 = new FullyConnectedLayer(
    _inputSize = 24,
    initWith = INIT_WITH_CONST, initialValue = .5,
    inputActivation = a_sigmoid)

  private val ol_1 = new OutputLayer(
    _inputSize = 24,
    costFunction = EUCLIDEAN)


  "the down arrow " should "be identified at pos 4 on the first featuremap" in {

    val nn = cl º RELU º ol_1
    cl.w.foreach(println)
    val res = nn.feedForward( image(1,1,DOWN))
    println(res)

    res(4) should be ( 3)
  }


  "helper function 'image'" should "generate a simple image" in {

    val img_down_arrow_at_1_1 = image(1,1,DOWN)
    val img_up_arrow_at_2_2 = image(2,2,UP)
    print(img_down_arrow_at_1_1)
    print(img_up_arrow_at_2_2)
  }

  /**
    * We place the given symbol on a 5x5 plane
    * @param pos_x the x coordinate on the plane
    * @param pos_y the y coordinate on the plane
    * @param symbol either UP or DOWN
    */
  private def image ( pos_x: Int, pos_y: Int, symbol: Int ) = {
    assert(pos_x >= 0 && pos_x <= 2)
    assert(pos_y >= 0 && pos_y <= 3)

    DenseVector((0 until 25 ).map(paint(pos_x + pos_y * 5, symbols(symbol), _)).toArray)
  }

  private def paint( offset: Int, symbol: Array[Int], actual: Int): Double = {
    if ( symbol.map(_ + offset).contains(actual)) 1.0 else 0.0
  }

  private def print( img: DV ): Unit = {
    val str: String = img.map(v=>if(v==1.0) " o" else " .").toArray.
      zipWithIndex.map(p=>p._1 + (if(p._2%5==4) "\n" else "")).mkString

    println(str)
  }
}
