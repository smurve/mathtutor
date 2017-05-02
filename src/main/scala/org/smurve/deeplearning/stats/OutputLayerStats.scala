package org.smurve.deeplearning.stats

class OutputLayerStats () extends LayerStats {

  val name: String = "Output"

  private var avgCostByTime: List[Double] = List[Double]()

  def registerCost ( cost: Double ): Unit =
    avgCostByTime = cost :: avgCostByTime

  def recentCost: Double = avgCostByTime.head
}
