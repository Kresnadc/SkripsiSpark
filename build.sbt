name := "SparkMLlibTest"

version := "0.1"

scalaVersion := "2.11.11"

libraryDependencies ++= Seq(
  "org.apache.spark" % "spark-core_2.11" % "2.2.0",
  "org.apache.spark" % "spark-mllib_2.11" % "2.2.0",
)
//// https://mvnrepository.com/artifact/org.apache.spark/spark-core
//libraryDependencies += "org.apache.spark" % "spark-core_2.11" % "2.2.0"
//// https://mvnrepository.com/artifact/org.apache.spark/spark-mllib
//libraryDependencies += "org.apache.spark" % "spark-mllib_2.11" % "2.2.0" % "provided"
//// https://mvnrepository.com/artifact/org.apache.hadoop/hadoop-client
//libraryDependencies += "org.apache.hadoop" % "hadoop-client" % "2.7.4"
