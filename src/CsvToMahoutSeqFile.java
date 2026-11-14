import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;

public class CsvToMahoutSeqFile {

    private static final String CSV_REGEX = ",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)";

    public static class SeqMapper extends Mapper<LongWritable, Text, Text, Text> {
        private final Text docKey = new Text();
        private final Text docVal = new Text();

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.isEmpty() || line.startsWith("review_id")) return;

            String[] cols = line.split(CSV_REGEX, -1);
            if (cols.length >= 5) {
                String reviewId = cols[0].replace("\"", "").trim();
                String ratingStr = cols[3].replace("\"", "").trim();
                String reviewText = cols[4].replace("\"", "").trim();

                if (reviewText.length() < 15) return;
                try {
                    double rating = Double.parseDouble(ratingStr);
                    String label = (rating >= 4.0) ? "Positive" : (rating == 3.0 ? "Neutral" : "Negative");
                    docKey.set("/" + label + "/" + reviewId + "_" + key.get());
                    docVal.set(reviewText);
                    context.write(docKey, docVal);
                } catch (NumberFormatException ignored) {}
            }
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Amazon 2023 CSV to Mahout SequenceFile");
        job.setJarByClass(CsvToMahoutSeqFile.class);
        job.setMapperClass(SeqMapper.class);
        job.setNumReduceTasks(0);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        job.setOutputFormatClass(SequenceFileOutputFormat.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
