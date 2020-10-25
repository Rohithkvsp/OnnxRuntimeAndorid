package com.play.onnxruntime;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import android.app.Activity;
import android.util.Log;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.SessionOptions;
import ai.onnxruntime.OrtSession.SessionOptions.OptLevel;
import ai.onnxruntime.NodeInfo;


public class ImageClassifier {

    protected String ModelFile = "mobilenetv2-7.onnx";
    protected String ModelFilePath = "/data/local/tmp/mobilenetv2-7.onnx";

    static String TAG = "OnnxRuntimeImageClassifier";

    private byte[] modeldata;
    private OrtSession session = null;
    ImageClassifier(Activity activity) {
        try {
            modeldata = loadModelFile(activity);
        } catch (IOException e) {
            Log.e(TAG, e.getMessage());
        }


        Log.d(TAG, "ImageClassifier constructor");


        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions opts = new SessionOptions();
        try {
            opts.setOptimizationLevel(OptLevel.BASIC_OPT);
            opts.addNnapi();
        } catch (OrtException e) {
            Log.e(TAG, e.getMessage());
        }


        try {
            session = env.createSession(ModelFilePath, opts);
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

}


/**

 try (OrtEnvironment env = OrtEnvironment.getEnvironment();
 OrtSession.SessionOptions opts = new SessionOptions()) {

 opts.setOptimizationLevel(OptLevel.BASIC_OPT);

 try (OrtSession session = env.createSession(modeldata, opts)) {

 Log.d(TAG, "Inputs ");
 for (NodeInfo i : session.getInputInfo().values()) {
 Log.d(TAG, i.toString());
 }

 Log.d(TAG, "Outputs ");

 for (NodeInfo i : session.getOutputInfo().values()) {
 Log.d(TAG, i.toString());

 }
 }
 }**/