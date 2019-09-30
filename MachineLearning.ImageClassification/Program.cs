using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using System.Collections.Generic;

namespace MachineLearning.ImageClassification
{
    class Program
    {
        private static readonly string _trainingFolder = Path.Combine(@"..\..\..\data", "Train");
        private static readonly string _validationFolder = Path.Combine(@"..\..\..\data", "Validate");
        private static readonly string _modelLocation = Path.Combine(@"..\..\..\data", "model.zip");

        static void Main(string[] args)
        {
            var ctx = new MLContext();

            var start = DateTime.Now;

            Train(ctx);

            Console.WriteLine($"Took to train {(DateTime.Now - start).TotalSeconds.ToString()}s");
            Console.WriteLine($"Saved model to \"{_modelLocation}\"");

            Validation(ctx);
        }

        private static void Train(MLContext ctx)
        {
            var data = ctx.Data.LoadFromEnumerable(ImageData.ReadFromFolder(_trainingFolder));
            var tensorflowModel = ctx.Model.LoadTensorFlowModel("tensorflow_inception_graph.pb");

            var pipeline = ctx.Transforms.Conversion.MapValueToKey(outputColumnName: "Label", inputColumnName: "Label")
                .Append(ctx.Transforms.LoadImages(outputColumnName: "input",
                                                      imageFolder: "",
                                                      inputColumnName: nameof(ImageData.Location)))
                .Append(ctx.Transforms.ResizeImages(outputColumnName: "input",
                                                        imageWidth: ImageNetSettings.imageWidth,
                                                        imageHeight: ImageNetSettings.imageHeight,
                                                        inputColumnName: "input"))
                .Append(ctx.Transforms.ExtractPixels(outputColumnName: "input",
                                                         interleavePixelColors: ImageNetSettings.channelsLast,
                                                         offsetImage: ImageNetSettings.mean))
                .Append(tensorflowModel.ScoreTensorFlowModel(outputColumnNames: new[] { "softmax2_pre_activation" },
                                                     inputColumnNames: new[] { "input" },
                                                     addBatchDimensionInput: true))
                .Append(ctx.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: "Label",
                                                                                      featureColumnName: "softmax2_pre_activation"))
                .Append(ctx.Transforms.Conversion.MapKeyToValue("Prediction", "PredictedLabel"));

            var model = pipeline.Fit(data);
            ctx.Model.Save(model, data.Schema, _modelLocation);
        }

        private static void Validation(MLContext ctx)
        {
            var model = ctx.Model.Load(_modelLocation, out var schema);
            var data = ctx.Data.LoadFromEnumerable(ImageData.ReadFromFolder(_validationFolder));

            // do the thing!
            var valData = model.Transform(data);

            // print out results
            var predictions = ctx.Data.CreateEnumerable<ImageDataPrediction>(valData, false, true);
            foreach (var pr in predictions)
                Console.WriteLine($"{pr.Location}, {pr.Prediction}, {pr.Score.Max()}");

            // print metrics
            var classificationContext = ctx.MulticlassClassification;
            var metrics = classificationContext.Evaluate(valData, labelColumnName: "Label", predictedLabelColumnName: "PredictedLabel");
            Console.WriteLine($"LogLoss is: {metrics.LogLoss}");
            Console.WriteLine($"PerClassLogLoss is: {String.Join(" , ", metrics.PerClassLogLoss.Select(c => c.ToString()))}");
        }
    }

    public struct ImageNetSettings
    {
        public const int imageHeight = 224;
        public const int imageWidth = 224;
        public const float mean = 117;
        public const float scale = 1;
        public const bool channelsLast = true;
    }

    public class ImageData
    {
        public string Location { get; set; }
        public string Label { get; set; }

        public static IEnumerable<ImageData> ReadFromFolder(string folder)
        {
            foreach (var name in Directory.EnumerateDirectories(folder))
            {
                var label = Path.GetFileName(name);
                foreach (var f in Directory.EnumerateFiles(name, "*.jpg"))
                    yield return new ImageData { Location = Path.GetFullPath(f), Label = label };
            }
        }
    }

    public class ImageDataPrediction
    {
        public string Location { get; set; }
        public float[] Score { get; set; }
        public string Prediction { get; set; }
    }
}
