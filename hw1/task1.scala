import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.storage.StorageLevel
import org.json4s._
import org.json4s.jackson.JsonMethods.parse
import java.io.PrintWriter

object nan_zheng_task1 {
    def main(args: Array[String]): Unit = {

    val Array(srcFile, desFile) = args

    val sparkConf = new SparkConf().setAppName("inf553_hw1_scala").setMaster("local[*]")
    val sc = new SparkContext(sparkConf)

    val file = sc.textFile(srcFile)

    val data = file.map(x=>parse(x))
       .map(x=>((x\"review_id").values.asInstanceOf[String], (x\"user_id").values.asInstanceOf[String], (x\"business_id").values.asInstanceOf[String], (x\"useful").values.asInstanceOf[BigInt], (x\"stars").values.asInstanceOf[Double], (x\"text").values.asInstanceOf[String].length()))
       .persist(StorageLevel.MEMORY_AND_DISK_SER)

    val museful = data.filter( x => (x._4 > 0) ).count()

    val mfivestar = data.filter( x => x._5 == 5.0 ).count()

    val mlongestreview = data.map( x => (x._6, 1) ).top(1)

    val muser = data.map(x=>(x._2, 1)).reduceByKey(_+_).sortBy(x=>x._1).persist(StorageLevel.MEMORY_AND_DISK_SER)

    val musernum = muser.count()

    val muserreview = muser.sortBy(x=>x._2, false).take(20).map(x=>"["+"\""+x._1+"\", "+x._2+"]")

    val mbusiness = data.map(x=>(x._3, 1)).reduceByKey(_+_).sortBy(x=>x._1).persist(StorageLevel.MEMORY_AND_DISK_SER)

    val mbusinessnum = mbusiness.count()

    val mbusinessreview = mbusiness.sortBy(x=>x._2, false).take(20).map(x=>"[\""+x._1+"\", "+x._2+"]")


    var s:String = "{\"n_review_useful\": "+museful+", \"n_review_5_star\": "+mfivestar+", \"n_characters\": "+mlongestreview(0)._1+", \"n_user\": "+musernum+", \"top20_user\": ["
    for (x <- muserreview)  s += x
    s += "], \"n_business\": "+mbusinessnum+", \"top20_business\": ["
    for (x <- mbusinessreview)  s += x
    s += "]}"
    s = s.replace("][","], [")

    val writer = new PrintWriter(desFile)
    writer.println(s)
    writer.close()

    sc.stop()

  }


}