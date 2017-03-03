package org.smurve.charts

import java.awt.RenderingHints

import org.jfree.chart.axis.NumberAxis
import org.jfree.chart.plot.{FastScatterPlot, PlotOrientation, XYPlot}
import org.jfree.chart.renderer.xy.XYDotRenderer
import org.jfree.chart.{ChartFactory, ChartPanel, JFreeChart}
import org.jfree.data.xy.{XYSeries, XYSeriesCollection}
import org.jfree.ui.{ApplicationFrame, RefineryUtilities}
import org.jfree.util.ShapeUtilities


class FastScatterPlotApplication extends ApplicationFrame("Fast Scatter Plot") {

  /** A constant for the number of items in the sample dataset. */
  private val COUNT = 100

  /** The data. */
  private val data = populateData2()

  private val chart: JFreeChart = ChartFactory.createScatterPlot(
    "Title", "X-Axis","Y-Axis",
    data, PlotOrientation.VERTICAL,
    false, false, false)

  val domainAxis = new NumberAxis("X")
  domainAxis.setAutoRangeIncludesZero(false)
  val rangeAxis = new NumberAxis("Y")
  rangeAxis.setAutoRangeIncludesZero(false)

  //val plot = new FastScatterPlot(this.data, domainAxis, rangeAxis)

  //val chart = new JFreeChart("Fast Scatter Plot", plot)

  val xYDotRenderer = new XYDotRenderer
  val cross = ShapeUtilities.createDiamond(2f) // createDiagonalCross(1, 1)
  xYDotRenderer.setBaseShape(cross)

  private val plot = chart.getPlot.asInstanceOf[XYPlot]
  //plot.setRenderer(new XYDotRenderer)
  plot.getRenderer(0).setSeriesShape(0, cross)



  //        chart.setLegend(null);


  // force aliasing of the rendered content..
  chart.getRenderingHints.put(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)

  val panel = new ChartPanel(chart, true)
  panel.setPreferredSize(new java.awt.Dimension(1024, 768))
  //      panel.setHorizontalZoom(true);
  //    panel.setVerticalZoom(true);
  panel.setMinimumDrawHeight(10)
  panel.setMaximumDrawHeight(2000)
  panel.setMinimumDrawWidth(20)
  panel.setMaximumDrawWidth(2000)

  setContentPane(panel)


  // ****************************************************************************
  // * JFREECHART DEVELOPER GUIDE                                               *
  // * The JFreeChart Developer Guide, written by David Gilbert, is available   *
  // * to purchase from Object Refinery Limited:                                *
  // *                                                                          *
  // * http://www.object-refinery.com/jfreechart/guide.html                     *
  // *                                                                          *
  // * Sales are used to provide funding for the JFreeChart project - please    *
  // * support us so that we can continue developing free software.             *
  // ****************************************************************************

  /**
    * Populates the data array with random values.
    */
  private def populateData(): Array[Array[Float]] = {

    (0 until COUNT).map(_=>
      Array((100 * math.random).toFloat, (100 * math.random).toFloat)
    ).toArray
  }
  /**
    * Populates the data array with random values.
    */
  private def populateData2(): XYSeriesCollection = {

    val series = new XYSeries("Random")
    (0 until COUNT).foreach(_=>
      series.add(100 * math.random, 100 * math.random)
    )
    new XYSeriesCollection(series)
  }
}

object FastScatterPlotDemo {

  /**
    * Starting point for the demonstration application.
    *
    * @param args  ignored.
    */
  def main( args: Array[String]) {

    val app:  FastScatterPlotApplication = new FastScatterPlotApplication()
    app.pack()
    RefineryUtilities.centerFrameOnScreen(app)
    app.setVisible(true)
  }

}