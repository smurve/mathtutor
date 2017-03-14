package org.smurve.charts

import java.awt.{Color, RenderingHints}

import org.jfree.chart.axis.NumberAxis
import org.jfree.chart.plot.{PlotOrientation, XYPlot}
import org.jfree.chart.{ChartFactory, ChartPanel, JFreeChart}
import org.jfree.data.xy.{XYSeries, XYSeriesCollection}
import org.jfree.ui.ApplicationFrame
import org.jfree.util.ShapeUtilities


class FastScatterPlot extends ApplicationFrame("Fast Scatter Plot") {

  /** The data. */
  private val dataSet : XYSeriesCollection = new XYSeriesCollection()

  private var nSeries = 0

  private val diamond = ShapeUtilities.createDiamond(1f)

  private val chart: JFreeChart =
    ChartFactory.createScatterPlot(
      "Some Data", "X-Axis", "Y-Axis",
      dataSet, PlotOrientation.VERTICAL,
      false, false, false)

  private val plot =chart.getPlot.asInstanceOf[XYPlot]

  construct ()

  private def construct() = {

    val domainAxis = new NumberAxis("X")
    domainAxis.setAutoRangeIncludesZero(false)
    val rangeAxis = new NumberAxis("Y")
    rangeAxis.setAutoRangeIncludesZero(false)


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
  }


  def addSeries ( data: Array[(Double, Double)], name: String, color: Color = null) : Unit = {
    val series = new XYSeries( name )
    data.foreach(p=>series.add(p._1, p._2))
    dataSet.addSeries(series)
    plot.getRenderer(0).setSeriesShape(nSeries, diamond)
    plot.getRenderer(0).setSeriesPaint(nSeries, color)
    nSeries += 1
  }

  def cls() : Unit = {
    dataSet.removeAllSeries()
    nSeries = 0
  }


}

