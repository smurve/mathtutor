package org.smurve.algebra

import breeze.linalg.{ DenseVector => DV }

/**
  * NOTE: NOT THREAD-SAFE (not meant to be)
  *
  * A screen is a class that creates string representations of 2-dimensional fields
  * It will always re-calculate the boundaries and adapt the displayed region if necessary
  *
  * @param screenWidth: The desired width in screen coordinates (pixels, char,...)
  * @param margin: the margin between outermost points and the screen boundary, expressed as a fraction of the screen width
  * @param initialBoundary: A box describing the smallest rectangle containing all the values
  */
class Screen(val screenWidth: Int, val margin: Double, initialBoundary: Box ) {

  private var currentMetaData = new MetaData(initialBoundary)

  private class MetaData ( val boundary: Box ) {

    // We'll provide some margin on each side of the data box
    val dataWidth: Double = (1 + 2 * margin) * (boundary.upperRight.data(0) - boundary.lowerLeft.data(0))
    val dataHeight: Double = (1 + 2 * margin) * (boundary.upperRight.data(1) - boundary.lowerLeft.data(1))
    assert(dataHeight > 0, "Cannot calculate screen height from that sample.")
    assert(dataWidth > 0, "Cannot calculate screen width from that sample.")

    // screen height is proportional
    private[algebra] val screenHeight = screenWidth

    private[algebra] val scale = dataWidth / screenWidth max dataHeight / screenHeight

    // the margin in screen coordinates
    private[algebra] val screenMarginY: Int = (screenHeight * margin).toInt
    private[algebra] var screenMarginX: Int = (screenWidth * margin).toInt

    // lower data boundaries
    private[algebra] val xOffset = boundary.lowerLeft.data(0)
    private[algebra] val yOffset = boundary.lowerLeft.data(1)

    // screen coordinates of both axis
    private[algebra] val screenYAxisX = (screenMarginX - xOffset / scale).toInt
    // on-screen x of the y-axis
    private[algebra] val screenXAxisY = (screenMarginY - yOffset / scale).toInt // on-screen y of the x-axis
  }


  /**
    * A function that can be displayed on the screen
    */
  private var func: (Double) => Double = { (x: Double) => -x }

  /**
    * set the function to be displayed
    * @param fun the function
    */
  def display ( fun: Double => Double ) : Unit = func = fun

  private def newMetaDataIfNecessary ( boundary: Box ) = {
    if (boundary != currentMetaData.boundary) {
      currentMetaData = new MetaData( boundary )
    }
    currentMetaData
  }

  /**
    * @param dx data cordinates
    * @return Screen x coordinates as a function of the data coordinates
    */
  def scx ( dx: Double ) : Int =
    ((dx - currentMetaData.xOffset) / currentMetaData.scale).toInt + currentMetaData.screenMarginX


  /**
    * @param dy data cordinates
    * @return Screen y coordinates as a function of the data coordinates
    */
  def scy ( dy: Double ) : Int =
    ((dy - currentMetaData.yOffset) / currentMetaData.scale).toInt + currentMetaData.screenMarginY


  /**
    *
    * @param sample     the data as a list of DenseVectors
    * @param displayAsX the index of the x-coordinates of the sample Vectors
    * @param displayAsY the index of the y-coordinates of the sample Vectors
    * @return A String displaying the data in the coordinate system
    */
  def frame(sample: List[DV[Double]], displayAsX: Int, displayAsY: Int): String = {

    val m: MetaData = newMetaDataIfNecessary(AlgebraUtils.boundaryBox(sample.toArray))

    val fields: Array[String] = Array.fill(screenWidth * m.screenHeight)("   ")


    val s = sample.head
    val symbol: Char = if (s.data(2) == 1) 'X' else 'O'
    draw(fields, s.data(displayAsX), s.data(displayAsY), symbol)

    for (s <- sample.tail) {
      val symbol: Char = if (s.data(2) == 1) 'x' else 'o'
      draw(fields, s.data(displayAsX), s.data(displayAsY), symbol)
    }




    val writableWidth = screenWidth / ( 1 + 2 * margin )
    val deltaX = (m.boundary.upperRight.data(0) - m.xOffset) / writableWidth

    for ( i <- (m.screenMarginX - 1 )until (screenWidth - m.screenMarginX - 2) ) {
      val x = m.xOffset + deltaX * i
      val y = func(x)
      draw(fields, x, y, '.')
    }




    val res = new Array[String](m.screenHeight)
    (0 until m.screenHeight ).foreach(y => {
      var line = ""
      (0 until screenWidth ).foreach(x => {
        val symbol =
          if (x == m.screenYAxisX && y == m.screenXAxisY) {
            " + "
          } else if (x == m.screenYAxisX) {
            " | "
          } else if ( y == m.screenXAxisY) {
            "---"
          } else  fields(screenWidth * y + x)
        line = line + symbol
      })
      res(y) = line
    })
    res.mkString("\n")
  }

  /**
    * draws a point on the field. Silently ignores points beyond the border
    * @param field the field to draw on
    * @param x the x coordinate of the point
    * @param y the y coordinate of the point
    * @param symbol the symbol to draw
    */
  private def draw ( field: Array[String], x: Double, y: Double, symbol: Char ) = {
    val ss = " " + symbol + " "
    val sx = scx(x)
    val sy = scy(y)

    val pos = screenWidth * (currentMetaData.screenHeight - sy) + sx - 1
    if ( pos < field.length && pos >= 0 ) {
      field(pos) = ss
    }
  }
}
