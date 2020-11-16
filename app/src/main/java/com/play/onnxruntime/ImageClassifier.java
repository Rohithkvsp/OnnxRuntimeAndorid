package com.play.onnxruntime;

import android.app.Activity;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;

import android.graphics.Bitmap;
import android.util.Log;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.SessionOptions;
import ai.onnxruntime.OrtSession.SessionOptions.OptLevel;
import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.TensorInfo;

import static java.util.Collections.*;



class Pair implements Comparable<Pair> {
    public final float first;
    public final int second;

    public Pair(float first, int second) {
        this.first = first;
        this.second = second;
    }

    @Override
    public int compareTo(Pair pair) {
        if(first > pair.first) {
            return 1;
        } else
            return -1;
    }
}

public class ImageClassifier {

    protected String ModelFile = "mobilenetv2-7.onnx";
    protected String labelFile = "labels.txt";

    protected String ModelFilePath = "/data/local/tmp/mobilenetv2-7.onnx";

    static String TAG = "OnnxRuntimeImageClassifier";
    protected ByteBuffer imgData = null;

    private static final int DIM_BATCH_SIZE = 1;
    private static final int DIM_CHANNEL_SIZE =3;
    private static final int  DIM_HEIGHT =224;
    private static final int DIM_WIDTH = 224;
    private static final int BYTES = 4;
    private static final float[] mean = {0.485f, 0.456f, 0.406f};
    private static final float[] vairance =  {0.229f, 0.224f, 0.225f};


    private byte[] modeldata;
    private OrtSession session = null;
    private OrtEnvironment env;
    private int[] intValues = new int[DIM_WIDTH *DIM_HEIGHT];


    private OnnxTensor input_tensor;

    private long[] input_node_dims;
    private long[] output_node_dims;

    private long output_size;
    //private float[] output_array;
    private float[] normalized;

    private float[] softmax_output;
    private PriorityQueue< Pair> pQueue = new PriorityQueue<Pair>();
    private ArrayList<Pair> results = new ArrayList<Pair>();



    private TensorInfo.OnnxTensorType input_node_type;
    private Map<String, OnnxTensor> container = new HashMap<>();
    private Set<String> requestedOutput =  new HashSet<>();
    private ArrayList<String> labels;



    ImageClassifier(Activity activity) {
        try {
            modeldata = loadModelFile(activity);
        } catch (IOException e) {
            Log.e(TAG, e.getMessage());
        }

        try {
            labels = loadLabelFile(activity);
        } catch (IOException e) {
            Log.e(TAG, e.getMessage());
        }


        Log.d(TAG, "ImageClassifier constructor");


        env = OrtEnvironment.getEnvironment();
        SessionOptions opts = new SessionOptions();
        try {
            opts.setOptimizationLevel(OptLevel.BASIC_OPT);
            //opts.addNnapi();
        } catch (OrtException e) {
            Log.e(TAG, e.getMessage());
        }


        try {
            session = env.createSession(modeldata, opts);

            Log.d(TAG, "----------Inputs-----------");
            for (NodeInfo i : session.getInputInfo().values()) {
                Log.d(TAG, i.getName());
                Log.d(TAG, i.toString());
                TensorInfo inputInfo = (TensorInfo) i.getInfo();
                input_node_type = inputInfo.onnxType;
                Log.d(TAG, String.valueOf(input_node_type));

                input_node_dims = inputInfo.getShape();

                for(int j = 0;j<inputInfo.getShape().length;j++)
                {
//                    Log.d(TAG, "Inputs dims "+inputInfo.getShape()[j]);
                    Log.d(TAG, "Input dims "+input_node_dims[j]);
                }
            }

            Log.d(TAG, "---------Outputs--------");

            for (NodeInfo i : session.getOutputInfo().values()) {
                Log.d(TAG, i.getName());
                Log.d(TAG, i.toString());
                TensorInfo outputInfo = (TensorInfo) i.getInfo();
                Log.d(TAG, String.valueOf(outputInfo.onnxType));

                output_node_dims = outputInfo.getShape();
                for(int j = 0;j<outputInfo.getShape().length;j++)
                {
//                    Log.d(TAG, "Output dims "+outputInfo.getShape()[j]);
                    Log.d(TAG, "Output dims "+output_node_dims[j]);
                }

            }
            output_size = output_node_dims[output_node_dims.length-1];
            Log.d(TAG, "Output size "+output_size);
            createInputOutputBuffer();

        } catch (OrtException e) {
            Log.e(TAG, e.getMessage());
        }


    }


    private byte[] loadModelFile(Activity activity) throws IOException{

        InputStream inputStream = activity.getAssets().open(ModelFile);
        byte[] byteArray = new byte[inputStream.available()];
        inputStream.read(byteArray);
        inputStream.close();
        Log.d(TAG,"total byte "+Long.toString(byteArray.length));
        return byteArray;
    }


    private ArrayList<String> loadLabelFile(Activity activity) throws IOException{

        ArrayList<String> labelList = new ArrayList<String>();
        InputStream inputStream = activity.getAssets().open(labelFile);
        BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
        String line = reader.readLine();
        while(line != null) {
            Log.d(TAG,line);
            line  = reader.readLine();
            labelList.add(line);
        }
        return labelList;
    }




    private void normalize()
    {
        int pixel = 0;
        for (int i = 0; i <DIM_HEIGHT ; ++i) {
            for (int j = 0; j < DIM_WIDTH; ++j) {
                final int val = intValues[pixel++];

                int r = (val >> 16) & 0xFF;
                int g = (val >> 8) & 0xFF;
                int b = val & 0xFF;

                normalized [(i*DIM_WIDTH+j)*DIM_CHANNEL_SIZE] = ((r/255.0f) - mean[0])/vairance[0]; //r
                normalized [(i*DIM_WIDTH+j)*DIM_CHANNEL_SIZE+1] = ((g/255.0f) - mean[1])/vairance[2]; //g
                normalized [(i*DIM_WIDTH+j)*DIM_CHANNEL_SIZE+2] = ((b/255.0f) - mean[2])/vairance[2];  //b

//                imgData.putFloat( ((r/255.0f) - mean[0])/vairance[0] );
//                imgData.putFloat( ((g/255.0f) - mean[1])/vairance[2] );
//                imgData.putFloat( ((b/255.0f) - mean[2])/vairance[2] );


//                imgData.putFloat((((val >> 16) & 0xFF) - mean[0])/vairance[0]); //r
//                imgData.putFloat((((val >> 8) & 0xFF) - mean[1])/vairance[1]);//g
//                imgData.putFloat(((val & 0xFF) - mean[2])/vairance[2]);//b

//                normalized [(i*DIM_WIDTH+j)*3] = (((val >> 16) & 0xFF) - mean[0])/vairance[0]; //r
//                normalized [(i*DIM_WIDTH+j)*3+1] = (((val >> 8) & 0xFF) - mean[1])/vairance[1]; //g
//                normalized [(i*DIM_WIDTH+j)*3+2] = ((val& 0xFF) - mean[2])/vairance[2];  //b


            }
        }
    }

    private  void HWCtoHWC()
    {
        imgData.rewind();
        for (int c = 0; c < 3; ++c) {
            for (int i = 0; i <DIM_HEIGHT ; ++i) {
                for (int j = 0; j < DIM_WIDTH; ++j) {

                    imgData.putFloat(normalized[(i*DIM_WIDTH+j)*DIM_CHANNEL_SIZE+c]);

                }
            }
        }

    }

    private void createInputOutputBuffer()
    {
        imgData = ByteBuffer.allocateDirect(DIM_BATCH_SIZE * DIM_HEIGHT * DIM_WIDTH * DIM_CHANNEL_SIZE * BYTES);
        imgData.order(ByteOrder.nativeOrder());
        Log.d(TAG,"ByteBuffer allocates "+imgData.capacity());

        try {
            input_tensor  = OnnxTensor.createTensor(env, imgData.asFloatBuffer(), input_node_dims);
            Log.d(TAG,"Input_tensor created");

            String inputName = session.getInputNames().iterator().next();
            Log.d(TAG,"InputName : "+inputName);
            container.put(inputName, input_tensor);

            String outputName = session.getOutputNames().iterator().next();
            requestedOutput.add(outputName);
            Log.d(TAG,"OutputName : "+outputName);


        } catch (OrtException e) {
            Log.e(TAG, e.getMessage());
        }

        normalized  = new float[DIM_BATCH_SIZE * DIM_HEIGHT * DIM_WIDTH * DIM_CHANNEL_SIZE];
        //output_array = new float[(int) output_size];
        softmax_output = new float[(int) output_size];
    }

    private void softmax(float[] output_array)
    {
        float max_val = output_array[0];
        for(int i=1;i<output_array.length;i++)
            max_val = Math.max(max_val, output_array[i]);


        float sum = 0.0f;
        for(int i=0;i<output_array.length;i++){
            float exp_val = (float) Math.exp(output_array[i] - max_val);
            softmax_output[i] = exp_val;
            sum += exp_val;
        }

        for(int i=0;i<output_array.length;i++){
            softmax_output[i] = softmax_output[i]/sum;
        }

        float max = getMax(softmax_output);
        float min = getMin(softmax_output);
        Log.d(TAG, "max softmax : "+max);
        Log.d(TAG, "min softmax : "+min);

        return;

    }

    private void postprocess(int k)
    {
        for(int i=0;i<softmax_output.length;i++)
        {
            pQueue.add(new Pair(softmax_output[i],i));
            while(pQueue.size()>k)
            {
                pQueue.poll();
            }
        }

        while(pQueue.size()>0)
        {
//            Log.d(TAG, "pQueue.size() "+pQueue.size());
            Pair pair = pQueue.poll();
//            Log.d(TAG, "pair"+String.valueOf(pair.first)+" , "+String.valueOf(pair.second));

            results.add(pair);
        }

        Collections.reverse(results);
        pQueue.clear();


    }

    public void run(Bitmap bitmap)
    {

        if (imgData == null) {
            return;
        }
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        normalize();
        HWCtoHWC();

        try (OrtSession.Result res = session.run(container)) {
            Log.d(TAG,"computed Output size : "+res.size());
            for(Map.Entry<String, OnnxValue> output_node: res)  {
                Log.d(TAG,"computed OutputName : "+output_node.getKey());
                Log.d(TAG,"computed Output type : "+output_node.getValue().getType());
                OnnxTensor outputTensor = (OnnxTensor) output_node.getValue();
                Log.d(TAG,"computed Output size : "+outputTensor.getFloatBuffer().capacity());
//                Log.d(TAG, String.valueOf(outputTensor.getFloatBuffer().array()[0]));
//                for(int i = 0;i <(int) output_size;i++)
//                {
//                    output_array[i] = outputTensor.getFloatBuffer().get();
//                }
                float [] output_array = outputTensor.getFloatBuffer().array();
                //Log.d(TAG,"output_arrays length "+output_arrays.length);

                softmax(output_array);
                postprocess(5);
                float max = getMax(output_array);
                float min = getMin(output_array);
                Log.d(TAG, "max : "+max);
                Log.d(TAG, "min : "+min);
                for(int i =0;i<3;i++)
                {

                    Log.d(TAG, "Output : " + String.valueOf(results.get(i).first)+ " , "+ String.valueOf(results.get(i).second)+" , "+labels.get(results.get(i).second));
                }

                results.clear();


            }

        } catch (OrtException e) {
            Log.e(TAG, e.getMessage());
        }


    }

    public static float getMax(float[] inputArray){
        float maxValue = inputArray[0];
        for(int i=1;i < inputArray.length;i++){
            if(inputArray[i] > maxValue){
                maxValue = inputArray[i];
            }
        }
        return maxValue;
    }

    public static float getMin(float[] inputArray){
        float minValue = inputArray[0];
        for(int i=1;i < inputArray.length;i++){
            if(inputArray[i] < minValue){
                minValue = inputArray[i];
            }
        }
        return minValue;
    }

}


