# ProgramSpark

Project perangkat lunak demo Apache Spark MLlib. Perangkat lunak dibuat dalam memenuhi kebutuhan Tugas Akhir prodi Informatika Unpar

## Getting Started
Instruksi berikut adalah untuk menjalankan proyek perangkat lunak pada cluster hadoop. Lihat bagian deployment untuk melihat cara deploy pada cluster hadoop.

### Prerequisites

Persyaratan lingkungan untuk menjalankan perangkat lunak demo

```
Java JDK 1.8.x
SBT 1.x.x
Hadoop 2.7.x
Apache Spark 2.2.0
```
### Create jar File

Membuat file .jar pada project perangkat lunak dengan menjalankan SBT.

```
/path/to/project/folder>sbt
```

Command line akan menjalankan SBT.

Build .jar dengan

```
sbt:project>assembly
```

File .jar berada dibawah directory
```
target\scala-2.11\projectSkripsi2014730048-assembly-1.0.jar
```

## Deployment

Langkah-langkah menjalankan perangkat lunak pada master node cluster hadoop

Buka command line atau terminal pada directory

```
/path/to/apache-spark-folder/bin/ 
```

Jalankan perintah berikut

```
./spark-submit --master yarn --deploy-mode client --executor-memory 5G /path/to/jar_file.jar
```

Perangkat lunak demo telah berjalan diatas cluster hadoop


## Built With

* [SBT](https://www.scala-sbt.org/) - Scala built tool
* [Maven](https://maven.apache.org/) - Dependency Management
* [Spark](https://spark.apache.org/) - Cluster computing framework


## Authors

* **Kresna Dwi Cahyo** - *2014730048 - Parahyangan University* - [Kresnadc](https://github.com/Kresnadc)

## Acknowledgments
Thanks
