package org.smurve.deeplearning.stats

import scala.collection.mutable

/**
  * Statistics and parameters of the network
  * @param NL the number of learning units of the network.
  * @param NS the number of steps expected to reach the minimum
  */
class NNStats ( val NL: Int = 10, val NS: Int = 100 ) {

  def getStats(key: String) : Option[LayerStats] = layerStats.get(key)

  private val layerStats:  mutable.Map[String, LayerStats] = new mutable.HashMap[String, LayerStats]()

  def registerStats(layer: LayerStats ): Unit = layerStats.put(layer.name, layer)

  def outputStats: OutputLayerStats = layerStats("Output").asInstanceOf[OutputLayerStats]

  def recentCost: Double = outputStats.recentCost

}
