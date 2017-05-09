package org.smurve.deeplearning.layers

import org.smurve.deeplearning.stats.{NNStats, OutputLayer}
import org.smurve.deeplearning.{DV, NeuralNetwork}

abstract class Layer()  {

  /**
    * Every Layer has a well defined input size that may, however, only be determined once the previous layer is known
    * @return the size of the expected input vector
    */
  def inputSize : Int

  /**
    * a readable name for diagnostic purposes
    * @return
    */
  def name: String

  /**
    * update the weights from the average corrections collected in previous learnings
    * @return the recent average loss
    */
  def update (nnstats: NNStats): NNStats

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

  /**
    * Alternative to the "||" stacking operator,
    * in case you prefer the typical math 'apply' symbol "º" (Alt-j on a Mac).
    * @param next the inner layer
    * @return
    */
  def º ( next: Layer ): Layer = this || next
  def º ( next: OutputLayer ): NeuralNetwork = this || next

  /**
    * Layer stacking operator.
    * @param next the subsequent layer
    * @return this, now representing the entire expression, or a NN, if the next layer is the output layer
    */
  def || (next: Layer ): Layer = {

    val thisExit = this.exit
    val nextEntry = next.entry
    thisExit.nextLayer = Some(nextEntry)
    nextEntry.previousLayer = Some(thisExit)
    this
  }

  def || (output: OutputLayer ): NeuralNetwork = {

    val thisExit = this.exit
    val nextEntry = output.entry
    thisExit.nextLayer = Some(nextEntry)
    nextEntry.previousLayer = Some(thisExit)

    output.initialize()
    new NeuralNetwork(this.entry, output)
  }


  private[deeplearning] var previousLayer: Option[Layer] = None
  private[deeplearning] var nextLayer: Option[Layer] = None

  /**
    * the first fundamental layer, if in a compound layer, otherwise return just the layer itself
    * @return
    */
  def entry: Layer = if (previousLayer.isDefined) previousLayer.get.entry else this

  /**
    * the last fundamental layer, if in a compound layer, otherwise return just the layer itself
    * @return
    */
  def exit: Layer = if (nextLayer.isDefined) nextLayer.get.exit else this

  /**
    * initialize weights, will be called by the subsequent layer. Should continue until the input layer
    */
  private [deeplearning] def initialize () : Unit
}
