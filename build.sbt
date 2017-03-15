name := "mathtutor"

version := "1.0"

scalaVersion := "2.11.8"

libraryDependencies ++= Seq("org.specs2" %% "specs2-core" % "3.8.8" % "test")

libraryDependencies  ++= Seq(
  // other dependencies here
  "org.apache.spark" %% "spark-mllib" % "2.1.0",
  //"jcommon" % "jcommon" % "1.0.16",
  "org.jfree" % "jfreechart-swt" % "1.0"


  // Check comments on ParallelTest for a discussion on the effects of this library
  //"com.github.fommil.netlib" % "all" % "1.1.2"
)

resolvers ++= Seq(
  // other resolvers here
  "Scala Tools Snapshots" at "https://oss.sonatype.org/content/groups/scala-tools/",
  "ScalaNLP Maven2" at "http://repo.scalanlp.org/repo"
)

scalacOptions in Test ++= Seq("-Yrangepos")
