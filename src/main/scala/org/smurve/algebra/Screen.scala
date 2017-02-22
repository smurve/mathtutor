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
  * @param dataBoundary: A box describing the smallest rectangle containing all the values
  */
class Screen(val screenWidth: Int, val margin: Double, dataBoundary: Box ) {

  private class MetaData ( val boundary: Box ) {

    // We'll provide some margin on each side of the data box
    val dataWidth: Double = (1 + 2 * margin) * (dataBoundary.upperRight.data(0) - dataBoundary.lowerLeft.data(0))
    val dataHeight: Double = (1 + 2 * margin) * (dataBoundary.upperRight.data(1) - dataBoundary.lowerLeft.data(1))
    assert(dataHeight > 0, "Cannot calculate screen height from that sample.")
    assert(dataWidth > 0, "Cannot calculate screen width from that sample.")

    private[algebra] val ratio: Double = dataHeight / dataWidth

    private[algebra] val scale = dataWidth / screenWidth

    // screen height is proportional
    private[algebra] val screenHeight = (screenWidth * ratio).toInt

    // the margin in screen coordinates
    private[algebra] val screenMarginY = screenHeight * margin * ratio
    private[algebra] var screenMarginX = screenWidth * margin

    // lower data boundaries
    private[algebra] val xOffset = dataBoundary.lowerLeft.data(0)
    private[algebra] val yOffset = dataBoundary.lowerLeft.data(1)

    // screen coordinates of both axis
    private[algebra] val screenYAxisX = (screenMarginX - xOffset / scale).toInt
    // on-screen x of the y-axis
    private[algebra] val screenXAxisY = (screenMarginY - yOffset / scale).toInt // on-screen y of the x-axis
  }

  private var currentMetaData = new MetaData(dataBoundary)


  private def newMetaDataIfNecessary ( boundary: Box ) = {
    if (boundary != currentMetaData.boundary) {
      currentMetaData = new MetaData( boundary )
    }
    currentMetaData
  }

  /**
    *
    * @param sample     the data as a list of DenseVectors
    * @param displayAsX the index of the x-coordinates of the sample Vectors
    * @param displayAsY the index of the y-coordinates of the sample Vectors
    * @return A String displaying the data in the coordinate system
    */
  def frame(sample: List[DV[Double]], displayAsX: Int, displayAsY: Int): String = {

    val m: MetaData = newMetaDataIfNecessary(AlgebraUtils.boundaryBox(sample.toArray))

    val fields: Array[String] = Array.fill(screenWidth * m.screenHeight)(" . ")

    for (s <- sample) {
      val x = ((s.data(displayAsX) - m.xOffset) / m.scale).toInt + m.screenMarginX
      val y = ((s.data(displayAsY) - m.yOffset) / m.scale).toInt + m.screenMarginY
      fields(screenWidth * (m.screenHeight - y.toInt) + x.toInt - 1) = if (s.data(2) == 1) " X " else " O "
    }

    val res = new Array[String](m.screenHeight)
    (0 until m.screenHeight ).foreach(y => {
      var line = ""
      (0 until screenWidth ).foreach(x => {
        var symbol = " . "
        if (x == m.screenYAxisX) symbol = " | "
        if (y == m.screenXAxisY) symbol = " - "
        if (x == m.screenYAxisX && y == m.screenXAxisY) symbol = " + "
        symbol = fields(screenWidth * y + x)
        line = line + symbol
      })
      res(y) = line
    })
    res.mkString("\n")
  }
}
