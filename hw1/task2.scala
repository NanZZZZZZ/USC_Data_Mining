import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.storage.StorageLevel
import org.json4s._
import org.json4s.jackson.JsonMethods.parse
import java.io.PrintWriter
import java.util.Calendar

object nan_zheng_task2 {
  def main(args: Array[String]): Unit = {

    val Array(srcFiler, srcFileb, desFile1, desFile2) = args

    val sparkConf = new SparkConf().setAppName("inf553_hw1_scala").setMaster("local[*]")
    val sc = new SparkContext(sparkConf)

    val filer = sc.textFile(srcFiler)
    val datar = filer.map(x=>parse(x))
      .map(x=>((x\"business_id").values.asInstanceOf[String], (x\"stars").values.asInstanceOf[Double]))

    val fileb = sc.textFile(srcFileb)
    val datab = fileb.map(x=>parse(x))
      .map(x=>((x\"business_id").values.asInstanceOf[String], (x\"state").values.asInstanceOf[String]))

    val data = datar.join(datab)
      .map(x=>(x._2._2, (x._2._1, 1)))
      .reduceByKey((x,y)=>(x._1+y._1, x._2+y._2))
      .sortByKey()

    val mavgstatestar = data.map(x=> (x._1, (x._2._1/x._2._2)))
      .sortBy(x=>x._2,false)
      .persist(StorageLevel.MEMORY_AND_DISK_SER)


//  1-collect
    val begin_time_1 = Calendar.getInstance().getTimeInMillis.toInt

    val mavgstatestarc = mavgstatestar.collect()
    for(i <- Range(0, 5))
        print(mavgstatestarc(i))

    val end_time_1 = Calendar.getInstance().getTimeInMillis.toInt

//  2-take
    val begin_time_2 = Calendar.getInstance().getTimeInMillis.toInt

    val m2 = mavgstatestar.take(5)
    for(i <- Range(0, 5))
        print(m2(i))

    val end_time_2 = Calendar.getInstance().getTimeInMillis.toInt

    val writer1 = new PrintWriter(desFile1)
    writer1.print("states,stars")
    for(e <- mavgstatestarc)
      writer1.print("\n"+e._1+","+e._2)
    writer1.close()

    val writer2 = new PrintWriter(desFile2)
    writer2.print("{\"m1\": "+(end_time_1-begin_time_1).toFloat/1000 +", \"m2\": "+(end_time_2-begin_time_2).toFloat/1000 +", \"explanation\": \"")
    writer2.println("The Method 2 needs less time, because the quantities of the data need to be dealed with are different. In M1, collect function transfers all the data, but in M2, the take function only takes first 5 and finishes the transfermation.\"}")
    writer2.close()

    sc.stop()

  }

}

