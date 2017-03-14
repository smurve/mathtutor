package org.smurve.mathtutor.demos

import java.awt.Color

import org.jfree.data.xy.XYSeries
import org.jfree.ui.RefineryUtilities
import org.smurve.charts.FastScatterPlot

/**
  * Demonstrate the FastScatterPlot API
  */
object ScatterPlotDemo {

  private var scp = new FastScatterPlot()
  /**
    * run as standalone application
    * @param args unused
    */
  def main(args: Array[String]) {

    addSeries("Random", someRandom(400), Color.RED)
    create()
  }

  /**
    * Show the plotter window
    */
  def create() : FastScatterPlot = {
    scp = new FastScatterPlot()
    scp.pack()
    RefineryUtilities.centerFrameOnScreen(scp)
    scp.setVisible(true)
    scp
  }

  /**
    * @param name the display name of the series
    * @param series the data series as an array of Double pairs
    */
  def addSeries ( name: String, series: Array[(Double, Double)], color: java.awt.Color):  Unit = {
    scp.addSeries(series,name, color)
  }

  /**
    * Clear all data from the display
    */
  def cls (): Unit = {
    scp.cls()
  }

  /**
    * Populates the data array with random values.
    * @param n: number of values
    */
  def someRandom(n: Integer): Array[(Double, Double)] = {

    (0 until n).map(_=>
      (100 * math.random, 100 * math.random)
    ).toArray
  }


}
