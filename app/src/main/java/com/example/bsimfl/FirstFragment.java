package com.example.bsimfl;

import android.content.res.AssetManager;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.fragment.app.Fragment;
import androidx.navigation.fragment.NavHostFragment;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.InputStreamInputSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.io.InputStream;
import java.util.List;

public class FirstFragment extends Fragment {

    private final String dataFile = "a7ddfb7f-c221-4d5b-a5c2-9f5e289269e1.csv";
    private final String ordinalDataFile = "data/ordinal/" + dataFile;
    private final String oneHotDataFile = "data/one_hot/" + dataFile; // Labels empiezan en col 55
    private final String file_x_train = "data/x_train/" + dataFile;
    private final String file_x_test = "data/x_test/" + dataFile;
    private final String file_y_train = "data/y_train/" + dataFile;
    private final String file_y_test = "data/y_test/" + dataFile;
    DataSet trainingData;
    DataSet testData;
    private final int batchSize = 100 ; // Nº de elementos cargados en cada iteración. 100 para tomar el dataset completo
    private final int features = 54;
    private final int classes = 10;
    private final double fractionTrain = 0.5;
    private final int epochs = 50;
    private final double lr = 0.015;
    private final double momentum = 0.9;


    @Override
    public View onCreateView(
            LayoutInflater inflater, ViewGroup container,
            Bundle savedInstanceState
    ) {
        // Inflate the layout for this fragment
        return inflater.inflate(R.layout.fragment_first, container, false);
    }

    public void onViewCreated(@NonNull View view, Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        view.findViewById(R.id.button_first).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                NavHostFragment.findNavController(FirstFragment.this)
                        .navigate(R.id.action_FirstFragment_to_SecondFragment);
            }
        });

        // Async task para la NN
        AsyncTask.execute(new Runnable() {
            @Override
            public void run() {
                // Obtener datos
                try {
                    createDataSource(ordinalDataFile); //TODO Cambiar este fichero
                } catch (IOException | InterruptedException e) {
                    Log.e("CARGA", "Algo ha ido mal en la carga de datos.");
                    e.printStackTrace();
                }
                // TODO probar la creación de la red
                createAndUseNetwork();
            }
        });
    }

    private void createAndUseNetwork() {
        // Variables
        int seed = 9;

        Log.i("createAndUseNetwork: ", "1) Creating layers");
        // Layers
        DenseLayer inputLayer = new DenseLayer.Builder()
                .nIn(features)
                .nOut(54)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.RELU)
                .build();

        DenseLayer hiddenLayer = new DenseLayer.Builder()
                .nIn(54)
                .nOut(54)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.RELU)
                .build();

        OutputLayer outputLayer = new OutputLayer.Builder(LossFunctions.LossFunction.MSE) // Mean Square Error obligada con activacion ReLU
                .nIn(54)
                .nOut(classes)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.RELU)
                .build();

        // Configuracion
        Log.i("createAndUseNetwork: ", "2) Creating configuration");
        MultiLayerConfiguration multiLayerConf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Sgd(lr))
                .list()
                .layer(0, inputLayer)
                .layer(1, hiddenLayer)
                .layer(2, outputLayer)
                .build();
        // TODO Como activar la backpropagation?
        // Model
        Log.i("createAndUseNetwork: ", "3) Creating model");
        MultiLayerNetwork model = new MultiLayerNetwork(multiLayerConf);
        model.init();

        // Resumen
        Log.i("createAndUseNetwork: ", "4) Summary");
        Log.i("NETWORK", "createAndUseNetwork: " + model.summary());

        // Training loop
        for(int l=0; l<=epochs; l++) {
            model.fit(trainingData);
        }

    }

    // TODO CSV con CSVwriter
    private void createDataSource(String dataFile) throws IOException, InterruptedException {
        //Pre: Create input stream
        Log.i("createDataSource: ", "0) Creando input stream");
        Log.i("createDataSource: ", "Ruta: "+dataFile);
        AssetManager am = this.getContext().getAssets();
        InputStream is = am.open(dataFile);
        Log.i("createDataSource", is.toString());

        //First: get the dataset using the record reader. CSVRecordReader handles loading/parsing
        Log.i("createDataSource: ", "1) Creando record reader");
        int numLinesToSkip = 1; // Cabecera
        char delimiter = ',';
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip, delimiter);
        recordReader.initialize(new InputStreamInputSplit(is));

        //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
        int labelIndex = 0; // Las acciones estan en la primera columna
        Log.i("createDataSource: ", "2) Creando iterador");
//        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, labelIndex, true);
//        DataSet allData = iterator.next();

        DataSetIterator iterator = new RecordReaderDataSetIterator.Builder(recordReader, batchSize)
                //Label index (first arg): Always value 1 when using ImageRecordReader. For CSV etc: use index of the column
                //  that contains the label (should contain an integer value, 0 to nClasses-1 inclusive). Column indexes start
                // at 0. Number of classes (second arg): number of label classes (i.e., 10 for MNIST - 10 digits)
                .classification(labelIndex, classes)
                .build();
        DataSet allData = iterator.next();

        // Dividir en train test
        Log.i("createDataSource: ", "3) Test y Train");
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(fractionTrain);  //Use 50% of data for training

        trainingData = testAndTrain.getTrain();
        testData = testAndTrain.getTest();

        Log.i("createDataSource: ", "Features train: " + trainingData.getFeatures().toString());
        Log.i("createDataSource: ", "Labels train: " + trainingData.getLabels().toString());
        Log.i("createDataSource: ", "N samples train: " + trainingData.numExamples());
        Log.i("createDataSource: ", "Label Names: " + trainingData.getLabelNames().toString());
        Log.i("createDataSource: ", "N Examples train: " + trainingData.numExamples());

        Log.i("createDataSource: ", "DATASET DE ENTRENAMIENTO: " + trainingData.toString());
        Log.i("createDataSource: ", "DATASET DE TESTEO: " + testData.toString());

        Log.i("createDataSource: ", "Fin de la carga.");

        // No haria falta normalizar, ya se hizo en el preprocesado en Python.
        //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
//        DataNormalization normalizer = new NormalizerStandardize();
//        normalizer.fit(trainingData);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
//        normalizer.transform(trainingData);     //Apply normalization to the training data
//        normalizer.transform(testData);         //Apply normalization to the test data. This is using statistics calculated from the *training* set
    }


    // TODO APLAZADO Leer el CSV y devolver lista de arrays (contenido de cada columna)
//    private static void get_train_data() {
//        List<Integer[]> questionList = new ArrayList<String[]>();
//        AssetManager assetManager = context.getAssets();
//
//        try {
//            InputStream csvStream = assetManager.open(CSV_PATH);
//            InputStreamReader csvStreamReader = new InputStreamReader(csvStream);
//            CSVReader csvReader = new CSVReader(csvStreamReader);
//            String[] line;
//
//            // throw away the header
//            csvReader.readNext();
//
//            while ((line = csvReader.readNext()) != null) {
//                questionList.add(line);
//            }
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//        return questionList;
//    }
}
//Example 1: Image classification, batch size 32, 10 classes
//        RecordReader rr = new ImageRecordReader(28,28,3); //28x28 RGB images
//        rr.initialize(new FileSplit(new File("/path/to/directory")));
//
//        DataSetIterator iter = new RecordReaderDataSetIterator.Builder(rr, 32)
//        //Label index (first arg): Always value 1 when using ImageRecordReader. For CSV etc: use index of the column
//        //  that contains the label (should contain an integer value, 0 to nClasses-1 inclusive). Column indexes start
//        // at 0. Number of classes (second arg): number of label classes (i.e., 10 for MNIST - 10 digits)
//        .classification(1, nClasses)
//        .preProcessor(new ImagePreProcessingScaler())      //For normalization of image values 0-255 to 0-1
//        .build()