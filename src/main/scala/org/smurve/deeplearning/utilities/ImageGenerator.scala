package org.smurve.deeplearning.utilities

/**
  * Create images containing a given number of simple 3x2 symbols
  */
class ImageGenerator(val width: Int, val height: Int ) {

  val downArrow = Array(1,0,1,0,1,0)
  val upArrow = Array(0,1,0,1,0,1)
  val square = Array(1,1,0,1,1,0)
  val line = Array(1,1,1,0,0,0)

  private val W_SYMBOL = 3
  private val H_SYMBOL = 2
  private val S_SYMBOL = W_SYMBOL * H_SYMBOL

  val symbols = Array ( downArrow, upArrow, square, line )

  val DOWN = 0
  val UP = 1
  val SQUARE = 2
  val LINE = 3

  def singleSymbolImage ( symbol: Int, posX: Int, posY: Int ): SingleSymbolImage = {

    assert ( posX + W_SYMBOL <= width && posX >= 0, "x position out of range")
    assert ( posY + H_SYMBOL <= height && posY >= 0, "y position out of range")
    val imgArray = Array.fill(width * height){0.0}
    val symbolArray = symbols(symbol)
    for ( i <- 0 until S_SYMBOL ) {
      val px = i % W_SYMBOL
      val py = i / W_SYMBOL
      val pixelX = posX + px
      val pixelY = posY + py
      imgArray(pixelX + width * pixelY) = symbolArray(i)
    }
    new SingleSymbolImage(width, height, imgArray, symbol)
  }

  def rndImage ( possibleSymbols: Int*) : SingleSymbolImage = {
    val posX = (math.random * (width - W_SYMBOL + 1)).toInt
    val posY = (math.random * (height - H_SYMBOL + 1)).toInt
    val symbol = possibleSymbols ((math.random * possibleSymbols.length ).toInt)
    singleSymbolImage(symbol, posX, posY)
  }
}
