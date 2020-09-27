# OnnxRuntimeAndorid

This app uses ONNXRuntime (with NNAPI enabled) for Android C/C++ library to run MobileNet-v2 ONNX model. Android camera pixels are passed to ONNXRuntime using JNI

  <table>
    <tr>
      <td>
        <img src="https://github.com/Rohithkvsp/OnnxRuntimeAndorid/blob/master/imgs/orange.png" width="300" height="500">
      </td>
      <td>
        <img src="https://github.com/Rohithkvsp/OnnxRuntimeAndorid/blob/master/imgs/laptop.png" width="300" height="500">
      </td>
      <td>
        <img src="https://github.com/Rohithkvsp/OnnxRuntimeAndorid/blob/master/imgs/lamp.png" width="300" height="500">
      </td>
    </tr>
  </table>
 

# Implemtation Details

ONNXRuntime shared object (libonnxruntime.so) was created by following instructions in ONNXRuntime Docs.
https://github.com/microsoft/onnxruntime/blob/master/BUILD.md#android-build-instructions

    ./build.sh --android --android_sdk_path /Users/username/Library/Android/sdk --android_ndk_path /Users/username/Documents/Android/android-ndk-r21d --android_abi arm64-v8a --android_api 28 --use_nnapi --parallel --build_shared_lib


libonnxruntime.so is copied to distribution/lib/arm64-v8a/libonnxruntime.so and ONNXRuntime headers are copied to distribution/include/onnxruntime

 - app/src/main/cpp path contains ONNXRuntime C++ inference, preprocessing input, postprocessing output, JNI wrappers code
 - app/src/main/java/com/play/onnxruntime contains java code to open camera, display image stream, access bitmap, UI, JNI calls to run inference..
 - app/CMakeLists.txt is the cmake file to compile the c++ code

# Run the prebuilt apk 
copy model/mobilenetv2-7.onnx and model/labels.txt to /data/local/tmp directoy on device

    adb push model/mobilenetv2-7.onnx /data/local/tmp
    adb push model/labels.txt /data/local/tmp

Install the prebuilt apk file

    adb install apk/app-debug.apk


# Build Instructions

#Requirements
 - Android SDK 29
 - Android NDK r21d

Open the local.properties file and set ndk.dir to the path of Android NDK folder.

    sdk.dir = /Users/Name/Library/Android/sdk
    ndk.dir = /Users/Name/Documents/Android/android-ndk-r21d

Run the app after making the above changes.

#Note
This app was test on Google pixel3 and build for arm64-v8a.

