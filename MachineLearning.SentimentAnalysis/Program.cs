using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using static Microsoft.ML.DataOperationsCatalog;

namespace MachineLearning.SentimentAnalysis
{
    class Program
    {
        private static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "amazon_cells_labelled.txt");

        static void Main(string[] args)
        {
            var ctx = new MLContext();

            var splitDataView = LoadData(ctx);

            var model = BuildAndTrainModel(ctx, splitDataView.TrainSet);

            Evaluate(ctx, model, splitDataView.TestSet);

            Predict(ctx, model, "This was a horrible meal", "I hate this", "I love this spaghetti");

            Console.ReadKey();
        }

        private static TrainTestData LoadData(MLContext ctx)
        {
            var dataView = ctx.Data.LoadFromTextFile<SentimentData>(_dataPath, hasHeader: false);

            var splitDataView = ctx.Data.TrainTestSplit(dataView, testFraction: 0.2);

            return splitDataView;
        }

        private static ITransformer BuildAndTrainModel(MLContext ctx, IDataView splitTrainSet)
        {
            var estimator = ctx.Transforms.Text
                .FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText)) // convert each word into a feature
                .Append(ctx.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

            Console.WriteLine("=============== Create and Train the Model ===============");
            var model = estimator.Fit(splitTrainSet);
            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();

            return model;
        }

        private static void Evaluate(MLContext ctx, ITransformer model, IDataView splitTestSet)
        {
            Console.WriteLine("=============== Evaluating Model accuracy with Test data ===============");
            var predictions = model.Transform(splitTestSet);

            var metrics = ctx.BinaryClassification.Evaluate(predictions, "Label");

            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}"); // accuracy of a model
            Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}"); // how confident the model is correctly classifying the positive and negative classes.
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}"); // measure of balance between precision and recall.
            Console.WriteLine("=============== End of model evaluation ===============");
        }

        private static void Predict(MLContext ctx, ITransformer model, params string[] text)
        {
            IEnumerable<SentimentData> sentiments = text.Select(t => new SentimentData
            {
                SentimentText = t
            });

            IDataView batchComments = ctx.Data.LoadFromEnumerable(sentiments);

            IDataView predictions = model.Transform(batchComments);

            var predictedResults = ctx.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);

            Console.WriteLine();

            Console.WriteLine("=============== Predictions ==============="); foreach (SentimentPrediction prediction in predictedResults)
            {
                Console.WriteLine($"Sentiment: {prediction.SentimentText} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} | Probability: {prediction.Probability} ");

            }
            Console.WriteLine("=============== End of predictions ===============");
        }
    }

    public class SentimentData
    {
        [LoadColumn(0)]
        public string SentimentText;

        [LoadColumn(1), ColumnName("Label")]
        public bool Sentiment;
    }

    public class SentimentPrediction : SentimentData
    {

        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }
    }
}
