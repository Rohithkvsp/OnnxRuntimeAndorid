package com.play.onnxruntime;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Point;
import android.graphics.RectF;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.CaptureResult;
import android.hardware.camera2.TotalCaptureResult;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.os.Build;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.view.Surface;
import android.view.TextureView;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    // Used to load the 'native-lib' library on application startup.

    private int CAMERA = 10;

    private TextureView mTextureView;

    private  TextView mTextView;

    private Button mButton;

    static String TAG = "onnxruntime_inference";

    private static final int MAX_PREVIEW_WIDTH = 1920;

    private static final int MAX_PREVIEW_HEIGHT = 1080;

    private Size mPreviewSize;

    private String mCameraId;

    private HandlerThread mBackgroundThread;

    private Handler mBackgroundHandler;

    private CameraDevice mCameraDevice;

    private CameraCaptureSession mCaptureSession;

    private CaptureRequest.Builder captureRequestBuilder;

    private CaptureRequest captureRequest;

    private final Object lock = new Object();

    private boolean runClassifier = false;


    private int img_height = 224;
    private int img_width = 224;
    private Inference inference=null;

    private Bitmap bitmap = null;
    private Bitmap resizedbitmap = null;
    private boolean checkedPermissions =false;
    private String output_text;

    String model_path = "/data/local/tmp/mobilenetv2-7.onnx";
    String label_file_path = "/data/local/tmp/labels.txt";



    private final TextureView.SurfaceTextureListener textureListener =
            new TextureView.SurfaceTextureListener() {

                @Override
                public void onSurfaceTextureAvailable(SurfaceTexture texture, int width, int height) {
                    openCamera(width, height);
                }

                @Override
                public void onSurfaceTextureSizeChanged(SurfaceTexture texture, int width, int height) {
                    configureTransform(width, height);
                }

                @Override
                public boolean onSurfaceTextureDestroyed(SurfaceTexture texture) {
                    return true;
                }

                @Override
                public void onSurfaceTextureUpdated(SurfaceTexture texture) {}
            };




    private final CameraDevice.StateCallback mStateCallback =
            new CameraDevice.StateCallback() {

                @Override
                public void onOpened(@NonNull CameraDevice cameraDevice) {
                    // This method is called when the camera is opened.  We start camera preview here.
                    //cameraOpenCloseLock.release();
                    Log.d(TAG,"Camera Opened");
                    mCameraDevice = cameraDevice;
                    createPreviewSession();

                }

                @Override
                public void onDisconnected(@NonNull CameraDevice currentCameraDevice) {
                    //cameraOpenCloseLock.release();
                    Log.d(TAG,"Camera Disconnected");
                    mCameraDevice.close();
                    mCameraDevice = null;
                }

                @Override
                public void onError(@NonNull CameraDevice currentCameraDevice, int error) {
                    //cameraOpenCloseLock.release();
                    Log.d(TAG,"Camera Error");
                    mCameraDevice.close();
                    mCameraDevice = null;
                }
            };


    private CameraCaptureSession.CaptureCallback captureCallback =
            new CameraCaptureSession.CaptureCallback() {

                @Override
                public void onCaptureProgressed(
                        @NonNull CameraCaptureSession session,
                        @NonNull CaptureRequest request,
                        @NonNull CaptureResult partialResult) {}

                @Override
                public void onCaptureCompleted(
                        @NonNull CameraCaptureSession session,
                        @NonNull CaptureRequest request,
                        @NonNull TotalCaptureResult result) {}
            };


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mTextureView = (TextureView)findViewById(R.id.texture_view);
        mTextView = (TextView)findViewById(R.id.text_view);
        mTextView.setTextColor(Color.RED);


        Bitmap.Config conf = Bitmap.Config.ARGB_8888; // see other conf types

        inference = new Inference(model_path, label_file_path, img_height, img_width);


    }


    @Override
    protected void onResume() {
        super.onResume();
        // hideSystemUI(); //hide UI
        Log.d(TAG,"on Resume");

        openBackgroundThread();//start the backgroundthread


        if(mTextureView.isAvailable()) {
            Log.d(TAG, "mTexture ias available");
            openCamera(mTextureView.getWidth(), mTextureView.getHeight());
        }
        else
            mTextureView.setSurfaceTextureListener(textureListener);

    }

    @Override
    protected void onPause() {
        closeCamera();
        closeBackgroundThread();
        super.onPause();

    }


    @Override
    protected void onDestroy() {
        super.onDestroy();

        Log.d(TAG,"on  Destory");
        if(inference!=null)
            inference.delete();
        if(bitmap!=null)
            bitmap.recycle();
        if(resizedbitmap!=null)
            resizedbitmap.recycle();
    }



    private static Size chooseOptimalSize(Size[] choices, int textureViewWidth, int textureViewHeight, int maxWidth, int maxHeight, Size aspectRatio) {
        // Collect the supported resolutions that are at least as big as the preview Surface
        List<Size> bigEnough = new ArrayList<>();
        // Collect the supported resolutions that are smaller than the preview Surface
        List<Size> notBigEnough = new ArrayList<>();
        int w = aspectRatio.getWidth();
        int h = aspectRatio.getHeight();
        for (Size option : choices) {
            if (option.getWidth() <= maxWidth && option.getHeight() <= maxHeight && option.getHeight() == option.getWidth() * h / w) {
                if (option.getWidth() >= textureViewWidth && option.getHeight() >= textureViewHeight) {
                    bigEnough.add(option);
                } else {
                    notBigEnough.add(option);
                }
            }
        }
        // Pick the smallest of those big enough. If there is no one big enough, pick the
        // largest of those not big enough.
        if (bigEnough.size() > 0) {
            return Collections.min(bigEnough, new CompareSizesByArea());
        } else if (notBigEnough.size() > 0) {
            return Collections.max(notBigEnough, new CompareSizesByArea());
        } else {
            Log.d(TAG, "Couldn't find any suitable preview size");
            return choices[0];
        }
    }

    static class CompareSizesByArea implements Comparator<Size> {
        @Override
        public int compare(Size lhs, Size rhs) {
            // We cast here to ensure the multiplications won't overflow
            return Long.signum((long) lhs.getWidth() * lhs.getHeight() - (long) rhs.getWidth() * rhs.getHeight());
        }
    }

    private void configureTransform(int viewWidth, int viewHeight) {
        if (null == mTextureView || null == mPreviewSize) {
            return;
        }
        int rotation = MainActivity.this.getWindowManager().getDefaultDisplay().getRotation();
        Matrix matrix = new Matrix();
        RectF viewRect = new RectF(0, 0, viewWidth, viewHeight);
        RectF bufferRect = new RectF(0, 0, mPreviewSize.getHeight(), mPreviewSize.getWidth());
        float centerX = viewRect.centerX();
        float centerY = viewRect.centerY();

        if (Surface.ROTATION_90 == rotation || Surface.ROTATION_270 == rotation) {
            bufferRect.offset(centerX - bufferRect.centerX(), centerY - bufferRect.centerY());
            matrix.setRectToRect(viewRect, bufferRect, Matrix.ScaleToFit.FILL);
            float scale = Math.max(
                    (float) viewHeight / mPreviewSize.getHeight(),
                    (float) viewWidth / mPreviewSize.getWidth());
            Log.e("Scale", Float.toString(scale));
            matrix.postScale(scale, scale, centerX, centerY);
            matrix.postRotate(90 * (rotation - 2), centerX, centerY);

        } else if (Surface.ROTATION_180 == rotation) {
            matrix.postRotate(180, centerX, centerY);
        }
        mTextureView.setTransform(matrix);
    }



    void setupcamera(int width, int height)
    {

        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.LOLLIPOP) {
            CameraManager cameraManager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
            try
            {
                for(String cameraID: cameraManager.getCameraIdList())
                {
                    Log.d(TAG,"cameraID "+cameraID);
                    CameraCharacteristics characteristics = cameraManager.getCameraCharacteristics(cameraID);
                    Integer facing = characteristics.get(CameraCharacteristics.LENS_FACING);
                    if(facing != null && facing == CameraCharacteristics.LENS_FACING_FRONT)
                    {
                        continue;
                    }

                    int displayRotation = MainActivity.this.getWindowManager().getDefaultDisplay().getRotation();
                    int sensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION);
                    Log.e(TAG,"sensorOrientation "+Integer.toString(sensorOrientation));
                    Log.e(TAG, "Display rotation " + displayRotation);

                    boolean swappedDimensions = false;
                    switch (displayRotation) {
                        case Surface.ROTATION_0:
                        case Surface.ROTATION_180:
                            if (sensorOrientation == 90 || sensorOrientation == 270) {
                                swappedDimensions = true;
                            }
                            break;
                        case Surface.ROTATION_90:
                        case Surface.ROTATION_270:
                            if (sensorOrientation == 0 || sensorOrientation == 180) {
                                swappedDimensions = true;
                            }
                            break;
                        default:
                            Log.e(TAG, "Display rotation is invalid: " + displayRotation);
                    }


                    StreamConfigurationMap map = characteristics.get(characteristics.SCALER_STREAM_CONFIGURATION_MAP);

                    Size largest = Collections.max(Arrays.asList(map.getOutputSizes(ImageFormat.JPEG)),new CompareSizesByArea());
                    Log.d(TAG, "largest ImageFormat.JPEG Width, height "+largest.getWidth()+" x "+largest.getHeight());

                    Point displaysize = new Point();
                    MainActivity.this.getWindowManager().getDefaultDisplay().getSize(displaysize);

                    int rotatedPreviewWidth = width;
                    int rotatedPreviewHeight = height;
                    int maxPreviewWidth = displaysize.x;
                    int maxPreviewHeight = displaysize.y;

                    if (swappedDimensions) {
                        rotatedPreviewWidth = height;
                        rotatedPreviewHeight = width;
                        maxPreviewWidth = displaysize.y;
                        maxPreviewHeight = displaysize.x;
                    }


                    Log.d(TAG,"Display Width , Height "+maxPreviewWidth+" , "+maxPreviewHeight);
                    Log.d(TAG,"Surface Width , Height "+rotatedPreviewWidth+" , "+rotatedPreviewHeight);
                    Log.d(TAG, "Thresold  Display Width, Height " + MAX_PREVIEW_WIDTH+" , "+ MAX_PREVIEW_HEIGHT);

                    if(maxPreviewWidth >  MAX_PREVIEW_WIDTH)
                    {
                        maxPreviewWidth = MAX_PREVIEW_WIDTH;
                    }
                    if(maxPreviewHeight > MAX_PREVIEW_HEIGHT)
                    {
                        maxPreviewHeight = MAX_PREVIEW_HEIGHT;
                    }
                    Log.d(TAG,"Thresolded Display Width, Height "+ maxPreviewWidth+" , "+ maxPreviewHeight );
                    mPreviewSize = chooseOptimalSize(map.getOutputSizes(SurfaceTexture.class), rotatedPreviewWidth, rotatedPreviewHeight, maxPreviewWidth, maxPreviewHeight, largest);
                    Log.d(TAG, "Optimal preview Size Width, Height "+mPreviewSize.getWidth()+" , "+mPreviewSize.getHeight());
                    mCameraId = cameraID;

                }
            }
            catch (CameraAccessException e)
            {
                e.printStackTrace();
            }
        }


    }

    private void openCamera(int width, int height) {

        if(Build.VERSION.SDK_INT >= 23)
            if(!askForPermission(Manifest.permission.CAMERA,CAMERA)) //ask for camera permission
                return ;

        setupcamera(width, height);
        configureTransform(width, height);
        CameraManager cameraManager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
            try {
            cameraManager.openCamera(mCameraId,mStateCallback,mBackgroundHandler);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    private boolean askForPermission(String permission, Integer requestCode) {
        if (ContextCompat.checkSelfPermission(MainActivity.this, permission) != PackageManager.PERMISSION_GRANTED) {
            if (ActivityCompat.shouldShowRequestPermissionRationale(MainActivity.this, permission)) {

                ActivityCompat.requestPermissions(MainActivity.this, new String[]{permission}, requestCode);
                return false;

            } else {

                ActivityCompat.requestPermissions(MainActivity.this, new String[]{permission}, requestCode);
            }
        } else {
            //OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_6, this, mLoaderCallback);
            Log.v(TAG, "permission  is already granted.");
            return true;
        }
        return false;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (ActivityCompat.checkSelfPermission(this, permissions[0]) == PackageManager.PERMISSION_GRANTED) {
            Toast.makeText(this, "Permission granted", Toast.LENGTH_SHORT).show();
        } else{

            Toast.makeText(this, "Permission denied", Toast.LENGTH_SHORT).show();
        }
    }



    private void createPreviewSession()
    {
        try {
            SurfaceTexture surfaceTexture = mTextureView.getSurfaceTexture();
            surfaceTexture.setDefaultBufferSize(mPreviewSize.getWidth(),mPreviewSize.getHeight());
            Surface previewSurface = new Surface(surfaceTexture);

            captureRequestBuilder = mCameraDevice.createCaptureRequest(mCameraDevice.TEMPLATE_PREVIEW);
            captureRequestBuilder.addTarget(previewSurface);

            mCameraDevice.createCaptureSession(Collections.singletonList(previewSurface),

                    new CameraCaptureSession.StateCallback() {
                        @Override
                        public void onConfigured(@NonNull CameraCaptureSession cameraCaptureSession) {
                            if(mCameraDevice == null)
                            {
                                return;
                            }

                            try {
                                mCaptureSession = cameraCaptureSession;
                                captureRequestBuilder.set(
                                        CaptureRequest.CONTROL_AF_MODE,
                                        CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);
                                captureRequest = captureRequestBuilder.build();

                                mCaptureSession.setRepeatingRequest(captureRequest,captureCallback,mBackgroundHandler);
                            } catch (CameraAccessException e) {
                                e.printStackTrace();
                            }

                        }

                        @Override
                        public void onConfigureFailed(@NonNull CameraCaptureSession cameraCaptureSession) {

                        }
                    },mBackgroundHandler);




        } catch (CameraAccessException e) {
            e.printStackTrace();
            Log.e(TAG, "Failed to preview Camera", e);
        }


    }


    private void classifyFrame() {
        if (inference == null) {
            Log.e(TAG,"Uninitialized Classifier or invalid context.");
            //return;
        }

        if (mCameraDevice == null) {
            Log.e(TAG,"Uninitialized Classifier or invalid context.");
            return;
        }


       //// resizedbitmap = mTextureView.getBitmap(out_width, out_height);
        bitmap = mTextureView.getBitmap();
        Log.v(TAG,"bitmap.getWidth() "+ bitmap.getWidth() + "bitmap.getHeight" + bitmap.getHeight());

        if (bitmap.getWidth() <= bitmap.getHeight()) { //portrait
            bitmap = Bitmap.createBitmap(bitmap, 0, bitmap.getHeight()/2 - bitmap.getWidth()/2, bitmap.getWidth(), bitmap.getWidth()); //center crop
            resizedbitmap = Bitmap.createScaledBitmap(bitmap, img_width, img_height, true);// resize

            long startTime = SystemClock.uptimeMillis();
            output_text = inference.run(resizedbitmap);

            long endTime = SystemClock.uptimeMillis();
            Log.e(TAG, " inference " + Long.toString(endTime - startTime) + " in mills ");

            runOnUiThread(new Runnable() {

                @Override
                public void run() {
                    mTextView.setText(output_text);
                }

            });

        }
        else //land scape
        {
            Log.e(TAG, "landscape");
            //Matrix matrix = new Matrix();
            //matrix.postRotate(270); //rotate


           // Bitmap dstBmp = Bitmap.createBitmap( bitmap,  0,  0,  bitmap.getWidth(),  bitmap.getHeight(),matrix, true);
            ///Log.v(TAG,"dstBmp.getWidth() "+ dstBmp.getWidth() + " dstBmp.getHeight " + dstBmp.getHeight());


           /// resizedbitmap  = Bitmap.createScaledBitmap (dstBmp,224, 224,true);
            bitmap = Bitmap.createBitmap( bitmap, 0,  bitmap.getHeight()/2 - bitmap.getWidth()/2,  bitmap.getWidth(),  bitmap.getWidth()); //center crop
            resizedbitmap = Bitmap.createScaledBitmap(bitmap, img_width, img_height, true);//resize

            output_text = inference.run(resizedbitmap);

            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    mTextView.setText(output_text);
                }

            });


        }




    }

    private Runnable periodicClassify;

    {
        periodicClassify = new Runnable() {
            @Override
            public void run() {

                synchronized (lock) {
                    if (runClassifier) {
                        classifyFrame();
                        Log.d(TAG, "peroidic classify in background thread");

                    }

                }
                //mBackgroundHandler.postDelayed(periodicClassify,10);// (no load)

                mBackgroundHandler.post(periodicClassify);//if load
            }
        };
    }


    private void openBackgroundThread()
    {
        mBackgroundThread = new HandlerThread("camera background thread");
        mBackgroundThread.start();
        mBackgroundHandler = new Handler( mBackgroundThread.getLooper());
        synchronized (lock) {
            runClassifier = true;
        }
        mBackgroundHandler.post(periodicClassify);

    }






    private void closeBackgroundThread()
    {

        mBackgroundThread.quitSafely();
        try {
            mBackgroundThread.join();
            mBackgroundThread = null;
            mBackgroundHandler = null;
            synchronized (lock) {
                runClassifier = false;
            }
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

    }

    private void closeCamera()
    {
        if (null != mCaptureSession) {
            mCaptureSession.close();
            mCaptureSession = null;
        }
        if (mCameraDevice != null) {
            mCameraDevice.close();
            mCameraDevice = null;
        }
    }



}
