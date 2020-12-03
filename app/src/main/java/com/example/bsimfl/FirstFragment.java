package com.example.bsimfl;

import android.content.res.AssetManager;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

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
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
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
    MultiLayerNetwork model;
    private final int batchSize = 100 ; // Nº de elementos cargados en cada iteración. 100 para tomar el dataset completo
    private final int features = 54;
    private final int classes = 10;
    private final double fractionTrain = 0.50;
    private final int epochs = 100;
    private final double lr = 0.015;
    private final double momentum = 0.9;
    private final List<String> labelList = new ArrayList<String>(Arrays.asList("turn off","turn on","Antena3","pop","FDF","70W","TeleCinco","20C","rock","18C"));

    // Views
    TextView tvSteps;
    TextView tvLog;
    TextView tvMetrics;

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

        // Initialize views
        tvSteps = view.findViewById(R.id.textView_steps);
        tvLog = view.findViewById(R.id.textView_log);
        tvMetrics = view.findViewById(R.id.textView_metrics);

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
//                    tvSteps.append("A) Create data source.");
                    createDataSource(ordinalDataFile);
                } catch (IOException | InterruptedException e) {
                    Log.e("CARGA", "Algo ha ido mal en la carga de datos.");
                    e.printStackTrace();
                }
                // Create and train network
//                tvSteps.append("B) Create and fit network.");
                createAndUseNetwork();

                // Evaluates network
//                tvSteps.append("C) Evaluate.");
                evaluateNetwork();
            }
        });
    }

    private void createAndUseNetwork() {
        // Variables
        int seed = 9;

//        tvLog.append("B.1 - Creating layers.");
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

        // TODO Cambiar la configuración para mejorar el modelo
        // Configuracion
//        tvLog.append("B.2 - Creating configuration.");
        Log.i("createAndUseNetwork: ", "2) Creating configuration");
        MultiLayerConfiguration multiLayerConf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Sgd(lr))
                //.weightDecay(momentum)
                .list()
                .layer(0, inputLayer)
                .layer(1, hiddenLayer)
                .layer(2, outputLayer)
                .backpropType(BackpropType.Standard) // For most MLPs
                .build();

        // Model
//        tvLog.append("B.3 - Creating model.");
        Log.i("createAndUseNetwork: ", "3) Creating model");
        model = new MultiLayerNetwork(multiLayerConf);
        model.init();

        // Summary
//        tvLog.append("B.4 - Generating summary.");
        Log.i("createAndUseNetwork: ", "4) Summary");
        Log.i("NETWORK", "createAndUseNetwork: " + model.summary());
//        tvLog.append(model.summary());

        // Training loop
//        tvLog.append("B.5 - Entering training loop...");
        Log.i("NETWORK", "Entering training loop...");
        for(int l=0; l<=epochs; l++) {
            model.fit(trainingData);
        }
        Log.i("NETWORK", "Training loop finished.");
//        tvLog.append("B.6 - Training loop finished.");
    }

    private void evaluateNetwork() {
//        tvLog.append("C.1 - Starting evaluation process.");
        Log.i("evaluateNetwork: ", "Starting evaluation process");
        Evaluation eval = new Evaluation(classes);
        eval.setLabelsList(labelList); // Set label literals

        // TODO Revisar esta entrada: actualIdx y
        INDArray predicted = model.output(testData.getFeatures());
        INDArray actual = testData.getLabels();
        eval.eval(actual, predicted);
//        tvLog.append(eval.stats());
        Log.i("evaluateNetwork: ", eval.stats());
        Log.i("evaluateNetwork: ", "Confusion matrix:\n" + eval.confusionToString());

//        tvLog.append("C.2 - End of evaluation.");
        Log.i("evaluateNetwork: ", "End of evaluation");
    }

    private void createDataSource(String dataFile) throws IOException, InterruptedException {
        //Pre: Create input stream
//        tvLog.append("A.0 - Creating input stream.");
        Log.i("createDataSource: ", "0) Creando input stream");
        Log.i("createDataSource: ", "Ruta: "+dataFile);
        AssetManager am = this.getContext().getAssets();
        InputStream is = am.open(dataFile);

        //First: get the dataset using the record reader. CSVRecordReader handles loading/parsing
//        tvLog.append("A.1 - Creating record reader.");
        Log.i("createDataSource: ", "1) Creando record reader");
        int numLinesToSkip = 1; // Cabecera
        char delimiter = ',';
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip, delimiter);
        recordReader.initialize(new InputStreamInputSplit(is));

        //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
        int labelIndex = 0; // Las acciones estan en la primera columna

//        tvLog.append("A.2 - Creating iterator.");
        Log.i("createDataSource: ", "2) Creando iterador");

        DataSetIterator iterator = new RecordReaderDataSetIterator.Builder(recordReader, batchSize)
                //Label index (first arg): Always value 1 when using ImageRecordReader. For CSV etc: use index of the column
                //  that contains the label (should contain an integer value, 0 to nClasses-1 inclusive). Column indexes start
                // at 0. Number of classes (second arg): number of label classes (i.e., 10 for MNIST - 10 digits)
                .classification(labelIndex, classes)
                .build();
        DataSet allData = iterator.next();

        // Dividir en train test
//        tvLog.append("A.3 - Perform train test split.");
        Log.i("createDataSource: ", "3) Test y Train");
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(fractionTrain);

        trainingData = testAndTrain.getTrain();
        testData = testAndTrain.getTest();

//        tvLog.append("A.4 - Train samples: " + trainingData.numExamples());
//        tvLog.append("A.4 - Test samples: " + testData.numExamples());

        Log.i("createDataSource: ", "Features train: " + trainingData.getFeatures().toString());
        Log.i("createDataSource: ", "Labels train: " + trainingData.getLabels().toString());
        Log.i("createDataSource: ", "N samples train: " + trainingData.numExamples());

        Log.i("createDataSource: ", "DATASET DE ENTRENAMIENTO: " + trainingData.toString());
        Log.i("createDataSource: ", "DATASET DE TESTEO: " + testData.toString());

        Log.i("createDataSource: ", "Fin de la carga.");

    }

    private void showInStepsView (String msg) {
        getActivity().runOnUiThread(new Runnable() {
            @Override
            public void run() {
                tvSteps.append(msg);
            }
        });
    }
}
