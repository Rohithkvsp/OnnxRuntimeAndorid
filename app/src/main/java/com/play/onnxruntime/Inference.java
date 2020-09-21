package com.play.onnxruntime;

import android.graphics.Bitmap;

public class Inference {


    private long selfAddr;

    static {
        System.loadLibrary("native-lib");
    }

    /**
     * makes jni call to create c++ reference
     */

    public Inference(String model_path, String label_file_path, int img_height, int img_width)
    {
        selfAddr = newSelf(model_path, label_file_path, img_height, img_width); //jni call to create c++ reference and returns address
    }

    /**
     * makes jni call to delete c++ reference
     */

    public void delete()
    {
        deleteSelf(selfAddr);//jni call to delete c++ reference
        selfAddr = 0;//set address to 0
    }

    @Override
    protected void finalize() throws Throwable {
        delete();
    }

    /**
     * return address of c++ reference
     */
    public long getselfAddr() {

        return selfAddr; //return address
    }

    /**
     * //makes jni call to proces frames
     */

    public String run(Bitmap input) {
        return run(selfAddr, input);//jni call to proces frames
    }


    private static native long newSelf(String model_path, String label_file_path, int img_height, int img_width);
    private static native void deleteSelf(long selfAddr);
    private static native String run(long selfAddr, Bitmap inbitmap);



}
