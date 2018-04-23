import java.io.{File, PrintWriter}

object TextToFile {

  /**
    * Method untuk membuat file dari sebuah teks
    * @param path direktori lokasi penyimpanan file
    * @param text input text yang akan disimpan
    */
  def saveResultToTextFile(path: String, text: String): Unit ={
    val pw = new PrintWriter(new File(path))
    pw.write(text)
    pw.close
  }
}
