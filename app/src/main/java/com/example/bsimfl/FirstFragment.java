package com.example.bsimfl;

import android.content.res.AssetManager;
import android.os.AsyncTask;
import android.os.Bundle;
import android.text.format.Formatter;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.fragment.app.Fragment;
import androidx.navigation.fragment.NavHostFragment;

import com.example.bsimfl.utils.FileUploadService;

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
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import okhttp3.ResponseBody;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

public class FirstFragment extends Fragment {

    private final String apiBaseUrl = "http://192.168.1.38:5000/"; // TODO poner la API real
    private final String globalModelFile = "model_global";
    private final String dataFileName = "a7ddfb7f-c221-4d5b-a5c2-9f5e289269e1";
    private final String dataFile = dataFileName + ".csv";
    private final String ordinalDataFile = "data/ordinal/" + dataFile;
    private final String oneHotDataFile = "data/one_hot/" + dataFile; // Labels empiezan en col 55
    private final String file_x_train = "data/x_train/" + dataFile;
    private final String file_x_test = "data/x_test/" + dataFile;
    private final String file_y_train = "data/y_train/" + dataFile;
    private final String file_y_test = "data/y_test/" + dataFile;
    private final String modelDir = "model/";
    DataSet trainingData;
    DataSet testData;
    MultiLayerNetwork model;
    MultiLayerNetwork globalModel;
    private final int batchSize = 100 ; // Nº de elementos cargados en cada iteración. 100 para tomar el dataset completo
    private int saveModelInterval = 10; // Cada cuantas epochs se guarda el modelo global para su envío al server
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

        // Load data source part
        view.findViewById(R.id.button_load).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Ejecutarla al pulsar el botón
                // Async task para la NN
                AsyncTask.execute(new Runnable() {
                    @Override
                    public void run() {
                        long startTime = System.nanoTime();

                        // Obtener datos
                        try {
                            showInStepsView("A) Create data source.");
                            createDataSource(ordinalDataFile);
                        } catch (IOException | InterruptedException e) {
                            Log.e("CARGA", "Algo ha ido mal en la carga de datos.");
                            e.printStackTrace();
                        }
                    }
                });
            }
        });

        // Training part
        view.findViewById(R.id.button_first).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Ejecutarla al pulsar el botón
                // Async task para la NN
                AsyncTask.execute(new Runnable() {
                    @Override
                    public void run() {
                        long startTime = System.nanoTime();

                        // Obtener datos
                        try {
                            showInStepsView("A) Create data source.");
                            createDataSource(ordinalDataFile);
                        } catch (IOException | InterruptedException e) {
                            Log.e("CARGA", "Algo ha ido mal en la carga de datos.");
                            e.printStackTrace();
                        }
                        // Create and train network
                        showInStepsView("B) Create and fit network.");
                        createAndUseNetwork();

                        // Evaluates network
                        showInStepsView("C) Evaluate local network.");
                        evaluateNetwork(model);
                        showInStepsView("D) Evaluate global network.");
                        evaluateNetwork(globalModel);

                        // Calculate time spent
                        long endTime = System.nanoTime();
                        long duration = (endTime - startTime)/1000000 ;  //milliseconds.
                        showInMetricsView("Time spent: " + duration + "ms (~"+ duration/1000 + " s)" );
                    }
                });
//                // Old behaviour
//                NavHostFragment.findNavController(FirstFragment.this)
//                        .navigate(R.id.action_FirstFragment_to_SecondFragment);
            }
        });


    }

    private void createAndUseNetwork() {
        // Variables
        int seed = 9;

        showInLogView("B.1 - Creating layers.");
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
        showInLogView("B.2 - Creating configuration.");
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

        // Local model
        showInLogView("B.3.1 - Creating local model.");
        Log.i("createAndUseNetwork: ", "3) Creating local model");
        model = new MultiLayerNetwork(multiLayerConf);
        model.init();

        // Global model (first initialization
        showInLogView("B.3.2 - Creating global model.");
        Log.i("createAndUseNetwork: ", "4) Creating global model");
        globalModel = new MultiLayerNetwork(multiLayerConf);
        globalModel.init();

        // Summary
        showInLogView("B.4 - Generating summary.");
        Log.i("createAndUseNetwork: ", "5) Summary");
        Log.i("NETWORK", "createAndUseNetwork: " + model.summary());
//        tvLog.append(model.summary());

        // Training loop for local model
        showInLogView("B.5 - Entering training loop...");
        Log.i("NETWORK", "Entering training loop...");
        for(int l=0; l<=epochs; l++) {
            // Train local model as usual
            model.fit(trainingData);

            // Train global model
            globalModel.fit(trainingData);
            // Saves model every N epochs
            if (l>0 && l%saveModelInterval==0){
                //File file = new File(modelDir + dataFile + "_" + l);
                String filename = dataFileName + "_" + l + "_global";
                serializeModel(globalModel, filename);
                // Upload file to server
                Log.i("NETWORK", "Uploading model");
                uploadFile(filename);
                // TODO eliminar ficheros no usados en la memoria
                // Downloading global model
                Log.i("NETWORK", "Downloading new model");
                downloadGlobalModel();
                // Create new instance of global model using Transfer Learning
                Log.i("NETWORK", "Deserialize and transfer learning");
                globalModel = transferLearning(multiLayerConf);
            }
        }
        Log.i("NETWORK", "Training loop finished.");
        showInLogView("B.6 - Training loop finished.");

    }

    /***
     * Performs Transfer Learning and generates a new model.
     * @param config
     * @return
     */
    private MultiLayerNetwork transferLearning(MultiLayerConfiguration config){
        MultiLayerNetwork oldModel = deserializeModel(globalModelFile);
        return new TransferLearning.Builder(oldModel)
                .setFeatureExtractor(0)
                .build();
    }

    private void evaluateNetwork(MultiLayerNetwork network) {
        showInLogView("Starting evaluation process.");
        Log.i("evaluateNetwork: ", "Starting evaluation process");
        Evaluation eval = new Evaluation(classes);
        eval.setLabelsList(labelList); // Set label literals

        // TODO Revisar esta entrada: Pide los idx, se están pasando realmente los idx o sólo las predicciones? Habria que calcular cada idx?
        //  Se supone que está bien implementado, aunque la precisión del modelo es pesima (culpa de librería?)
        INDArray predicted = network.output(testData.getFeatures());
        INDArray actual = testData.getLabels();
        eval.eval(actual, predicted);
        showInLogView(eval.stats());
        Log.i("evaluateNetwork: ", eval.stats());
        Log.i("evaluateNetwork: ", "Confusion matrix:\n" + eval.confusionToString());

        showInLogView("C.2 - End of evaluation.");
        Log.i("evaluateNetwork: ", "End of evaluation");
    }

    private void createDataSource(String dataFile) throws IOException, InterruptedException {
        //Pre: Create input stream
        showInLogView("A.0 - Creating input stream.");
        Log.i("createDataSource: ", "0) Creando input stream");
        Log.i("createDataSource: ", "Ruta: "+dataFile);
        AssetManager am = this.getContext().getAssets();
        InputStream is = am.open(dataFile);

        //First: get the dataset using the record reader. CSVRecordReader handles loading/parsing
        showInLogView("A.1 - Creating record reader.");
        Log.i("createDataSource: ", "A.1 - Creating record reader.");
        int numLinesToSkip = 1; // Cabecera
        char delimiter = ',';
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip, delimiter);
        recordReader.initialize(new InputStreamInputSplit(is));

        //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
        int labelIndex = 0; // Las acciones estan en la primera columna

        showInLogView("A.2 - Creating iterator.");
        Log.i("createDataSource: ", "2) Creando iterador");

        DataSetIterator iterator = new RecordReaderDataSetIterator.Builder(recordReader, batchSize)
                //Label index (first arg): Always value 1 when using ImageRecordReader. For CSV etc: use index of the column
                //  that contains the label (should contain an integer value, 0 to nClasses-1 inclusive). Column indexes start
                // at 0. Number of classes (second arg): number of label classes (i.e., 10 for MNIST - 10 digits)
                .classification(labelIndex, classes)
                .build();
        DataSet allData = iterator.next();

        // Dividir en train test
        showInLogView("A.3 - Perform train test split.");
        Log.i("createDataSource: ", "3) Test y Train");
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(fractionTrain);

        trainingData = testAndTrain.getTrain();
        testData = testAndTrain.getTest();

        showInLogView("A.4 - Train samples: " + trainingData.numExamples());
        showInLogView("A.5 - Test samples: " + testData.numExamples());

        Log.i("createDataSource: ", "Features train: " + trainingData.getFeatures().toString());
        Log.i("createDataSource: ", "Labels train: " + trainingData.getLabels().toString());
        Log.i("createDataSource: ", "N samples train: " + trainingData.numExamples());

        Log.i("createDataSource: ", "DATASET DE ENTRENAMIENTO: " + trainingData.toString());
        Log.i("createDataSource: ", "DATASET DE TESTEO: " + testData.toString());

        Log.i("createDataSource: ", "Fin de la carga.");

    }

    /***
     * Serializes a model and save it in internal storage (accessible only for this app)
     * @param model
     */
    private void serializeModel(MultiLayerNetwork model, String filename){

        try {
            // Save file
            File file = new File(this.getContext().getFilesDir().getPath() + "/"+ filename);
            model.save(file);
            // Show file/size on log
            String file_size = Formatter.formatShortFileSize(this.getContext(),file.length());
            showInLogView("--Saved: " + filename + " -> " + file_size);
        } catch (IOException e) {
            Log.e("serializeModel: ", "Error in model serialization.");
            e.printStackTrace();
        }

    }

    /***
     * Deserializes a model from internal storage into a variable
     * If newModel is null, it means something went wrong during deserialization.
     * @param filename path to file
     */
    private MultiLayerNetwork deserializeModel(String filename){
        MultiLayerNetwork newModel = null;
        try {
            // Load file
            File file = new File(this.getContext().getFilesDir().getPath() + "/"+ filename);
            newModel = MultiLayerNetwork.load(file, true);
            // Show file/size on log
            String file_size = Formatter.formatShortFileSize(this.getContext(),file.length());
            showInLogView("--Loaded: " + filename + " -> " + file_size);
        } catch (IOException e) {
            Log.e("deserializeModel: ", "Error in model deserialization.");
            e.printStackTrace();
        }
        return newModel;
    }

    private void showInStepsView (String msg) {
        getActivity().runOnUiThread(new Runnable() {
            @Override
            public void run() {
                tvSteps.append(msg + "\n");
            }
        });
    }

    private void showInLogView (String msg) {
        getActivity().runOnUiThread(new Runnable() {
            @Override
            public void run() {
                tvLog.append(msg + "\n");
            }
        });
    }

    private void showInMetricsView (String msg) {
        getActivity().runOnUiThread(new Runnable() {
            @Override
            public void run() {
                tvMetrics.append(msg + "\n");
            }
        });
    }

    /***
     * Uploads local model to server
     * @param filename
     */
    private void uploadFile(String filename) {

        // https://github.com/iPaulPro/aFileChooser/blob/master/aFileChooser/src/com/ipaulpro/afilechooser/utils/FileUtils.java
        // use the FileUtils to get the actual file by uri

        //File file = FileUtils.getFile(this, fileUri);
        File originalFile = new File(this.getContext().getFilesDir().getPath() + "/"+ filename);

        // create RequestBody instance from file
        RequestBody filePart =
                RequestBody.create(
                        MultipartBody.FORM,
                        originalFile
                );

        // File Key - Value. MultipartBody.Part is used to send also the actual file name (multipart necesita un wrapper adicional)
        MultipartBody.Part body =
                MultipartBody.Part.createFormData("file", originalFile.getName(), filePart);

        // Description part. add another part within the multipart request
        String descriptionString = "hello, this is description speaking";
        RequestBody description =
                RequestBody.create(
                        okhttp3.MultipartBody.FORM, descriptionString);

        // Create Retrofit instance (service)
        Retrofit.Builder builder = new Retrofit.Builder()
                .baseUrl(apiBaseUrl)
                .addConverterFactory(GsonConverterFactory.create());

        Retrofit retrofit = builder.build();

        // Apply our interface
        FileUploadService service = retrofit.create(FileUploadService.class);

        // finally, execute the request
        Call<ResponseBody> call = service.upload(description, body);
        call.enqueue(new Callback<ResponseBody>() {
            @Override
            public void onResponse(Call<ResponseBody> call,
                                   Response<ResponseBody> response) {
                Log.i("uploadFile:", "Upload success");
            }

            @Override
            public void onFailure(Call<ResponseBody> call, Throwable t) {
                Log.e("uploadFile:", "Upload error: " + t.getMessage());
            }
        });
    }

    /***
     * Downloads global model from server
     */
    private void downloadGlobalModel() {

        // Create Retrofit instance (service)
        Retrofit.Builder builder = new Retrofit.Builder()
                .baseUrl(apiBaseUrl)
                .addConverterFactory(GsonConverterFactory.create());

        Retrofit retrofit = builder.build();

        // Apply our interface
        FileUploadService service = retrofit.create(FileUploadService.class);

        // Request
        Call<ResponseBody> call = service.downloadGlobalModel();

        Response<ResponseBody> response = null;
        try {
            response = call.execute();
            if (response.isSuccessful()) {
                Log.i("downloadGlobalModel: ", "server contacted and has file");
                boolean writtenToDisk = writeResponseBodyToDisk(response.body());
                Log.i("downloadGlobalModel: ", "file download was a success? " + writtenToDisk);
            }else{
                Log.e("downloadGlobalModel: ", "server contact failed");
            }
        } catch (IOException e) {
            Log.e("downloadGlobalModel: ", "error");
            e.printStackTrace();
        }
//         // Metodo asincrono con el enqueue (en FL debe ser sincrono)
//        call.enqueue(new Callback<ResponseBody>() {
//            @Override
//            public void onResponse(Call<ResponseBody> call, Response<ResponseBody> response) {
//                if (response.isSuccessful()) {
//                    Log.i("downloadGlobalModel: ", "server contacted and has file");
//
//                    boolean writtenToDisk = writeResponseBodyToDisk(response.body());
//
//                    Log.i("downloadGlobalModel: ", "file download was a success? " + writtenToDisk);
//                } else {
////                    Log.e("downloadGlobalModel: ", "server contact failed");
////                }
//            }
//
//            @Override
//            public void onFailure(Call<ResponseBody> call, Throwable t) {
//                Log.e("downloadGlobalModel: ", "error");
//            }
//        });
    }

    /***
     * Writes a response body to internal storage
     * @param body
     * @return
     */
    private boolean writeResponseBodyToDisk(ResponseBody body) {
        try {
            // File destination
            File file = new File(this.getContext().getFilesDir().getPath() + "/" + globalModelFile);

            InputStream inputStream = null;
            OutputStream outputStream = null;

            try {
                byte[] fileReader = new byte[4096];

                long fileSize = body.contentLength();
                long fileSizeDownloaded = 0;

                inputStream = body.byteStream();
                outputStream = new FileOutputStream(file);

                while (true) {
                    int read = inputStream.read(fileReader);

                    if (read == -1) {
                        break;
                    }

                    outputStream.write(fileReader, 0, read);

                    fileSizeDownloaded += read;

                    Log.i("writeResponseBodyToDisk:", "file download: " + fileSizeDownloaded + " of " + fileSize);
                }

                outputStream.flush();

                return true;
            } catch (IOException e) {
                e.printStackTrace();
                return false;
            } finally {
                if (inputStream != null) {
                    inputStream.close();
                }

                if (outputStream != null) {
                    outputStream.close();
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }
    }
}
