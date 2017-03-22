package org.smurve.mnist

case class ConvolutionFrame(input_cols: Int, input_rows: Int, window_cols: Int, window_rows: Int ) {
  val feature_cols: Int = input_cols - window_cols + 1
  val size_featureMap: Int = feature_cols * ( input_rows - window_rows + 1)
  val size_window: Int = window_rows * window_cols

  @inline
  def tau ( k: Int, j: Int ) : Int = phi ( j ) + xi ( k )

  @inline
  def tau ( k: Int) : Array[Int] = (0 until size_window).map(phi (_) + xi ( k )).toArray

  @inline
  def phi ( j: Int ) : Int = j % window_cols + (j / window_cols) * input_cols

  @inline
  def xi ( k: Int ) : Int = (k / feature_cols) * input_cols + k % feature_cols
}
