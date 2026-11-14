import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Collections;
import java.util.Map;
import java.util.TreeMap;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class CsvReviewAnalytics {

    private static final String CSV_REGEX = ",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)";

    // Compact 16-byte Binary Writable for Sum + Count shuffle optimization
    public static class SumCountWritable implements Writable {
        private double sum;
        private long count;

        public SumCountWritable() {}
        public void set(double sum, long count) { this.sum = sum; this.count = count; }
        public double getSum() { return sum; }
        public long getCount() { return count; }

        @Override
        public void write(DataOutput out) throws IOException {
            out.writeDouble(sum);
            out.writeLong(count);
        }

        @Override
        public void readFields(DataInput in) throws IOException {
            sum = in.readDouble();
            count = in.readLong();
        }
    }

    // =========================================================================
    // JOB 1: AVERAGE RATING & VOLUME PER PRODUCT
    // =========================================================================
    public static class ProductRatingMapper extends Mapper<LongWritable, Text, Text, SumCountWritable> {
        private final Text productKey = new Text();
        private final SumCountWritable outVal = new SumCountWritable();

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.isEmpty() || line.startsWith("review_id")) return;

            String[] cols = line.split(CSV_REGEX, -1);
            if (cols.length >= 4) {
                String productId = cols[1].replace("\"", "").trim();
                String ratingStr = cols[3].replace("\"", "").trim();
                try {
                    double rating = Double.parseDouble(ratingStr);
                    if (!productId.isEmpty() && rating >= 1.0 && rating <= 5.0) {
                        productKey.set(productId);
                        outVal.set(rating, 1L);
                        context.write(productKey, outVal);
                    }
                } catch (NumberFormatException ignored) {}
            }
        }
    }

    public static class SumCountCombiner extends Reducer<Text, SumCountWritable, Text, SumCountWritable> {
        private final SumCountWritable partial = new SumCountWritable();

        @Override
        public void reduce(Text key, Iterable<SumCountWritable> values, Context context)
                throws IOException, InterruptedException {
            double sum = 0.0;
            long count = 0L;
            for (SumCountWritable v : values) {
                sum += v.getSum();
                count += v.getCount();
            }
            partial.set(sum, count);
            context.write(key, partial);
        }
    }

    public static class ProductRatingReducer extends Reducer<Text, SumCountWritable, Text, Text> {
        private final Text result = new Text();

        @Override
        public void reduce(Text key, Iterable<SumCountWritable> values, Context context)
                throws IOException, InterruptedException {
            double sum = 0.0;
            long count = 0L;
            for (SumCountWritable v : values) {
                sum += v.getSum();
                count += v.getCount();
            }
            if (count > 0) {
                result.set(String.format("AvgRating=%.4f\tTotalReviews=%d", sum / count, count));
                context.write(key, result);
            }
        }
    }

    // =========================================================================
    // JOB 2: CATEGORY STAR-RATING DISTRIBUTION MATRIX
    // =========================================================================
    public static class CategoryDistMapper extends Mapper<LongWritable, Text, Text, LongWritable> {
        private static final LongWritable ONE = new LongWritable(1L);
        private final Text catRatingKey = new Text();

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.isEmpty() || line.startsWith("review_id")) return;

            String[] cols = line.split(CSV_REGEX, -1);
            if (cols.length >= 4) {
                String category = cols[2].replace("\"", "").trim();
                String rating = cols[3].replace("\"", "").trim();
                if (!category.isEmpty() && rating.matches("[1-5]")) {
                    catRatingKey.set(category + "\t" + rating + "_Star");
                    context.write(catRatingKey, ONE);
                }
            }
        }
    }

    public static class LongSumReducer extends Reducer<Text, LongWritable, Text, LongWritable> {
        private final LongWritable out = new LongWritable();

        @Override
        public void reduce(Text key, Iterable<LongWritable> values, Context context)
                throws IOException, InterruptedException {
            long sum = 0L;
            for (LongWritable v : values) sum += v.get();
            out.set(sum);
            context.write(key, out);
        }
    }

    // =========================================================================
    // JOB 3: TWO-STAGE TOP-K MOST REVIEWED PRODUCTS
    // =========================================================================
    public static class ProductCountMapper extends Mapper<LongWritable, Text, Text, LongWritable> {
        private static final LongWritable ONE = new LongWritable(1L);
        private final Text productKey = new Text();

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.isEmpty() || line.startsWith("review_id")) return;

            String[] cols = line.split(CSV_REGEX, -1);
            if (cols.length >= 2) {
                String productId = cols[1].replace("\"", "").trim();
                if (!productId.isEmpty()) {
                    productKey.set(productId);
                    context.write(productKey, ONE);
                }
            }
        }
    }

    public static class TopKMapper extends Mapper<LongWritable, Text, LongWritable, Text> {
        private final TreeMap<Long, String> localTop = new TreeMap<>();
        private int k = 10;

        @Override
        protected void setup(Context context) { k = context.getConfiguration().getInt("analytics.topk", 10); }

        @Override
        public void map(LongWritable key, Text value, Context context) {
            String[] p = value.toString().split("\t");
            if (p.length == 2) {
                localTop.put(Long.parseLong(p[1]), p[0]);
                if (localTop.size() > k) localTop.remove(localTop.firstKey());
            }
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            for (Map.Entry<Long, String> e : localTop.entrySet()) {
                context.write(new LongWritable(e.getKey()), new Text(e.getValue()));
            }
        }
    }

    public static class TopKReducer extends Reducer<LongWritable, Text, Text, LongWritable> {
        private final TreeMap<Long, String> globalTop = new TreeMap<>(Collections.reverseOrder());
        private int k = 10;

        @Override
        protected void setup(Context context) { k = context.getConfiguration().getInt("analytics.topk", 10); }

        @Override
        public void reduce(LongWritable key, Iterable<Text> values, Context context) {
            for (Text v : values) {
                globalTop.put(key.get(), v.toString());
                if (globalTop.size() > k) globalTop.remove(globalTop.lastKey());
            }
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            for (Map.Entry<Long, String> e : globalTop.entrySet()) {
                context.write(new Text(e.getValue()), new LongWritable(e.getKey()));
            }
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.err.println("Usage: CsvReviewAnalytics <rating|distribution|topk> <inputDir> <outputDir> [topK]");
            System.exit(1);
        }

        Configuration conf = new Configuration();
        conf.setBoolean("mapreduce.map.output.compress", true);
        conf.setInt("mapreduce.task.io.sort.mb", 256);

        String mode = args[0].toLowerCase();
        Path inPath = new Path(args[1]);
        Path outPath = new Path(args[2]);

        if (mode.equals("rating")) {
            Job job = Job.getInstance(conf, "Amazon 2023 Product Rating Aggregation");
            job.setJarByClass(CsvReviewAnalytics.class);
            job.setMapperClass(ProductRatingMapper.class);
            job.setCombinerClass(SumCountCombiner.class);
            job.setReducerClass(ProductRatingReducer.class);
            job.setMapOutputKeyClass(Text.class);
            job.setMapOutputValueClass(SumCountWritable.class);
            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(Text.class);
            FileInputFormat.addInputPath(job, inPath);
            FileOutputFormat.setOutputPath(job, outPath);
            System.exit(job.waitForCompletion(true) ? 0 : 1);

        } else if (mode.equals("distribution")) {
            Job job = Job.getInstance(conf, "Amazon 2023 Category Star Distribution");
            job.setJarByClass(CsvReviewAnalytics.class);
            job.setMapperClass(CategoryDistMapper.class);
            job.setCombinerClass(LongSumReducer.class);
            job.setReducerClass(LongSumReducer.class);
            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(LongWritable.class);
            FileInputFormat.addInputPath(job, inPath);
            FileOutputFormat.setOutputPath(job, outPath);
            System.exit(job.waitForCompletion(true) ? 0 : 1);

        } else if (mode.equals("topk")) {
            int k = (args.length >= 4) ? Integer.parseInt(args[3]) : 10;
            conf.setInt("analytics.topk", k);
            Path stage1 = new Path(outPath.toString() + "_stage1");

            Job job1 = Job.getInstance(conf, "Amazon 2023 Top-K Stage 1");
            job1.setJarByClass(CsvReviewAnalytics.class);
            job1.setMapperClass(ProductCountMapper.class);
            job1.setCombinerClass(LongSumReducer.class);
            job1.setReducerClass(LongSumReducer.class);
            job1.setOutputKeyClass(Text.class);
            job1.setOutputValueClass(LongWritable.class);
            FileInputFormat.addInputPath(job1, inPath);
            FileOutputFormat.setOutputPath(job1, stage1);
            if (!job1.waitForCompletion(true)) System.exit(1);

            Job job2 = Job.getInstance(conf, "Amazon 2023 Top-K Stage 2");
            job2.setJarByClass(CsvReviewAnalytics.class);
            job2.setMapperClass(TopKMapper.class);
            job2.setReducerClass(TopKReducer.class);
            job2.setNumReduceTasks(1);
            job2.setMapOutputKeyClass(LongWritable.class);
            job2.setMapOutputValueClass(Text.class);
            job2.setOutputKeyClass(Text.class);
            job2.setOutputValueClass(LongWritable.class);
            FileInputFormat.addInputPath(job2, stage1);
            FileOutputFormat.setOutputPath(job2, outPath);
            System.exit(job2.waitForCompletion(true) ? 0 : 1);
        }
    }
}
