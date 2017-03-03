package org.smurve.charts

import java.awt.{Color, RenderingHints}

import breeze.linalg.{DenseVector => DV}
import org.jfree.chart.axis.NumberAxis
import org.jfree.chart.plot.XYPlot
import org.jfree.chart.{ChartFactory, ChartPanel, JFreeChart}
import org.jfree.data.xy.{XYSeries, XYSeriesCollection}
import org.jfree.ui.{ApplicationFrame, RefineryUtilities}
import org.jfree.util.ShapeUtilities
import org.smurve.algebra.fun.{Fun, _}
import org.smurve.complex._

import scala.collection.immutable.IndexedSeq

/**
  * parallel projection for an observing plane from default direction:
  * phi =   -110° from x-axis,
  * theta = 30° from z-plane
  */
class Chart3D(phi_deg: Double = -110, theta_deg: Double = 30)(implicit spec: PlotterSpec)
  extends ApplicationFrame("3D Chart") {

  // some reasonable defaults for the three core components (fun and fp belong together)
  private var domain = Domain(-1, 1, -1, 1, 100, 100)
  private var fun: Fun = x*x
  private var fp : Cpx=>Double = z=>z.r
  private var projector = new PlaneProjector(phi_deg, theta_deg)

  private val dataSet = new XYSeriesCollection // = sampleData(100) //new XYSeriesCollection


  createPanel()

  private def createPanel(): ChartPanel = {


    Set("Axis", "Shadow", "Data").foreach{n=>dataSet.addSeries(new XYSeries(n))}

    val chart: JFreeChart = ChartFactory.createScatterPlot(
      spec.title, spec.xLabel, spec.yLabel,
      dataSet, spec.orientation, false, false, false)
    chart.getRenderingHints.put(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)

    val domainAxis = new NumberAxis("X")
    domainAxis.setAutoRangeIncludesZero(false)
    val rangeAxis = new NumberAxis("Y")
    rangeAxis.setAutoRangeIncludesZero(false)

    //val xYDotRenderer = new XYDotRenderer
    val diamond = ShapeUtilities.createDiamond(1f)
    val slimDiamond = ShapeUtilities.createDiamond(0.5f)
    //xYDotRenderer.setBaseShape(cross)

    val plot = chart.getPlot.asInstanceOf[XYPlot]
    plot.setDomainGridlinesVisible(false)
    plot.setRangeGridlinesVisible(false)
    plot.getDomainAxis.setVisible(false)
    plot.getRangeAxis.setVisible(false)

    plot.getRenderer(0).setSeriesShape(0, slimDiamond)
    plot.getRenderer(0).setSeriesPaint(0, Color.RED)

    plot.getRenderer(0).setSeriesShape(1, diamond)
    plot.getRenderer(0).setSeriesPaint(1, Color.DARK_GRAY)

    plot.getRenderer(0).setSeriesShape(2, slimDiamond)
    plot.getRenderer(0).setSeriesPaint(2, Color.RED)

    new ChartPanel(chart, true)

    val panel = new ChartPanel(chart, true)
    panel.setPreferredSize(new java.awt.Dimension(1024, 768))
    panel.setMinimumDrawHeight(10)
    panel.setMaximumDrawHeight(2000)
    panel.setMinimumDrawWidth(20)
    panel.setMaximumDrawWidth(2000)

    setContentPane(panel)

    pack()
    RefineryUtilities.centerFrameOnScreen(this)
    setVisible(true)

    panel
  }


  /**
    * creates a new projector reflecting the given angles
    * @param phi_deg rotation around the z-axis
    * @param theta_deg inclination towards the z-axis
    * @return
    */
  def fromAngle ( phi_deg: Double, theta_deg: Double ) : Chart3D= {
    projector = new PlaneProjector(phi_deg, theta_deg)
    display()
    this
  }

  def setFun ( fun: Fun, fp: Cpx=>Double) : Chart3D = {
    this.fun = fun
    this.fp = fp
    display()
    this
  }

  def setDomain(domain: Domain): Chart3D = {
    this.domain = domain
    display()
    this
  }

  /**
    * clear all data from the screen
    */
  def cls(): Unit = {
    dataSet.removeAllSeries()
  }

  /**
    * Populates the data array with random values.
    */
  private def axesData(name: String, domain: Domain, minF: Double, maxF: Double, f: Cpx=>Double): XYSeries = {

    val series = new XYSeries(name)

    val dx = (domain.right - domain.left) / 400
    (0 until 400).foreach(n => {
      val x = domain.left + dx * n
      val p = projector.project(x, 0, f(x))
      series.add(p._1, p._2)
    })

    val dy = (domain.upper - domain.lower) / 400
    (0 until 400 ).foreach(n => {

      val y = domain.lower + dy * n
      val p = projector.project(0, y, f(i*y))
      series.add(p._1, p._2)
    })

    //val numZ = domain.numX.min(domain.numY)
    val dz = (maxF - minF) / 400
    ( 0 until 400).foreach(n=>{
      val z = minF + n * dz
      val p = projector.project(0,0,z)
      series.add(p._1, p._2)
    })
    series
  }



  /**
    */
  def display() : Unit = {

    cls()

    assert(domain != null)
    assert(fun != null)

    val series = new XYSeries("Data")
    val z: (Double, Double) = domain.values(0)
    var minf = fp(fun(Cpx(z._1, z._2)))
    var maxf = minf

    //*
    domain.values.foreach ( p => {
      val z = Cpx(p._1, p._2)

      val optionalF = evalIfVisible(fun, _.r, z, domain.dx, domain.dy )

      optionalF.foreach (f => {
        maxf = maxf max f
        minf = minf min f
        val (x, y) = projector.project(p._1, p._2, f)
        series.add(x, y)
      })

    }) // */
    dataSet.addSeries(axesData("Shadow", domain, minf, maxf, fun(_).r))
    dataSet.addSeries(axesData("Axis", domain, minf, maxf, _=>0))
    dataSet.addSeries(series)
  }

  /**
    * A point x,y,f is considered visible, if the upwards-pointing normal vector on the tangent plane of f(x,y)
    * points towards the observing vector zp
    * @param dx: delta x
    * @param dy: delta y
    * @return
    */
  def evalIfVisible ( fun: Fun, pf: Cpx=>Double, z: Cpx, dx: Double, dy: Double ): Option[Double] = {
    val f = pf(fun(z))
    val fx = (pf(fun(z+dx)) - f ) / dx
    val fy = (pf(fun(z+i*dy)) - f ) / dy

    if ( projector.zp.t * DV(-fx, -fy, 1) < 0 )
      Option(f)
    else
      None
  }
}


object Chart3D {

  def main(args: Array[String]): Unit = {

    val f: Fun = exp(-x*conj(x)) * ((x-1)°2) * (x+1)°2
    val chart = new Chart3D()(PlotterSpec(title = f.toString))

    chart.cls()
    //chart.showFun(0.4* sin(2 * x) * exp(-conj(x) * x), _.i)(Domain(-4, 4, -4, 4, 200, 200))
    //chart.showFun(.2 * x°3, _.r)(Domain(-2, 2, -2, 2, 200, 200))
    chart
        .setFun(f, _.r)
        .setDomain(Domain(-2, 2, -2, 2, 200, 200))
        .fromAngle(-120, 50)
  }
}
