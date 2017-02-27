package org.smurve.charts

import org.jfree.chart.axis.NumberAxis
import org.jfree.chart.plot.PlotOrientation
import org.jfree.chart.{ChartFactory, ChartPanel, JFreeChart}
import org.jfree.data.xy.{XYSeries, XYSeriesCollection}
import org.jfree.ui.{ApplicationFrame, RefineryUtilities}
import org.smurve.algebra.fun.Fun

class ScPlot {

}

class PlotterSpec (
                    val name: String
                  ) {}

class FastScatterPlotDemo ( fun: Fun)( implicit spec: PlotterSpec )  extends ApplicationFrame("Chart") {

  val dataSet = new XYSeriesCollection
  dataSet.addSeries(createSeries ( spec, fun ))

  val chart: JFreeChart = ChartFactory.createXYLineChart(
    "XY Chart", // Title
    "x-axis", // x-axis Label
    "y-axis", // y-axis Label
    dataSet, // Dataset
    PlotOrientation.VERTICAL, // Plot Orientation
    true, // Show Legend
    true, // Use tooltips
    false // Configure chart to generate URLs?
  )


  val domainAxis = new NumberAxis("X")
  domainAxis.setAutoRangeIncludesZero(false)
  val rangeAxis = new NumberAxis("Y")
  rangeAxis.setAutoRangeIncludesZero(false)

  // force aliasing of the rendered content..
  //chart.getRenderingHints.put (RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)

  val panel = new ChartPanel(chart, true)
  panel.setPreferredSize(new java.awt.Dimension(500, 270))
  //      panel.setHorizontalZoom(true);
  //    panel.setVerticalZoom(true);
  panel.setMinimumDrawHeight(10)
  panel.setMaximumDrawHeight(2000)
  panel.setMinimumDrawWidth(20)
  panel.setMaximumDrawWidth(2000)

  setContentPane(panel)

  pack()
  RefineryUtilities.centerFrameOnScreen(this)
  setVisible(true)


  private def createSeries (spec: PlotterSpec, fun: Fun ) : XYSeries = {
    val series = new XYSeries(spec.name)

    series.add(10.0, 10.0)
    series.add(-12.0, 3.0)
    series.add(7.0, -8.0)
    series.add(-15.0, -12.0)
    series
  }
}