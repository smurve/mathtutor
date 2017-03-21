package org.smurve.mnist

trait Layer {

  /**
    * update the weights from the average corrections collected in previous learnings
    * @param eta: the learning factor
    */
  def update ( eta: Double ): Unit

  /**
    * just the forward feed, returns the final activations as a result
    * @param x the vector to classify
    * @return the classification for the given vector according to the recent model
    */
  def feedForward ( x: DV) : DV

  /**
    *
    * @param x the input vector to learn from
    * @param y the desired classification for the given input
    * @return the back-propagated deltas
    */
  def feedForwardAndPropBack(x: DV, y: DV): DV

  }
