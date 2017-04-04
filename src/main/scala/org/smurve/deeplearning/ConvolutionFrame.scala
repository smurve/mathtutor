package org.smurve.deeplearning

case class ConvolutionFrame(input_cols: Int, input_rows: Int, window_cols: Int, window_rows: Int ) {
  /**
    * the width of the resulting feature matrix
    */
  val feature_cols: Int = input_cols - window_cols + 1
  val feature_rows: Int = input_rows - window_rows + 1
  val size_featureMap: Int = feature_cols * feature_rows
  val size_window: Int = window_rows * window_cols


  /**
    * domain's index with offset = k and index target index j
    */
  @inline
  def tau ( k: Int, j: Int ) : Int = phi ( j ) + xi ( k )

  /**
    * @param k offset
    * @return the tau function applied to the entire window
    */
  @inline
  def tau ( k: Int) : Array[Int] = (0 until size_window).map (tau (_, k)).toArray

  @inline
  private def phi ( j: Int ) : Int = j % window_cols + (j / window_cols) * input_cols

  @inline
  private def xi ( k: Int ) : Int = (k / feature_cols) * input_cols + k % feature_cols
}
